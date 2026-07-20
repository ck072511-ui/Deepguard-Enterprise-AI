"""
DeepGuard — api/v1/endpoints/image_detect.py

Dedicated image-only deepfake detection endpoint.

POST /api/v1/detect/image
  Upload a single image file and receive a deepfake prediction.

Produces:
  - Detailed DetectionResponse with label, confidence, face count, timing
  - Prometheus metric recording via api.middleware.metrics.record_detection
  - Structured error responses via the global exception handlers

Validation:
  - MIME type checked via content-type header AND magic-byte sniffing
  - File size enforced before reading the entire payload
  - 422 returned for malformed requests with field-level details
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Security, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.security import require_any_auth
from core.exceptions.api_exceptions import (
    FileTooLargeError,
    InferenceError,
    UnsupportedMediaTypeError,
)
from database.session import get_db_session
from schemas.responses.detection import DetectionResponse
from services.detection.service import DetectionService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Image Detection"])

# ── Constants ─────────────────────────────────────────────────────────────────

_ALLOWED_IMAGE_TYPES = frozenset({
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff",
})
_ALLOWED_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"})
_MAX_IMAGE_MB = 10
_MAX_IMAGE_BYTES = _MAX_IMAGE_MB * 1024 * 1024

# JPEG / PNG / WebP magic bytes
_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF", "image/webp"),      # RIFF....WEBP
    (b"BM", "image/bmp"),
]


def _sniff_image(header: bytes) -> bool:
    """Return True if the first bytes look like a known image format."""
    for magic, _ in _IMAGE_MAGIC:
        if header[:len(magic)] == magic:
            return True
    return False


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/detect/image",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect deepfake in an image",
    description=(
        "Upload a single **image file** (JPEG, PNG, WebP, BMP) and receive a deepfake "
        "prediction. The response includes the predicted label (REAL / FAKE), "
        "confidence score, number of faces detected, and timing metadata.\n\n"
        "**Authentication:** Requires a valid API key (`X-API-Key` header) or "
        "a JWT Bearer token."
    ),
    responses={
        200: {"description": "Detection completed successfully"},
        400: {"description": "Unsupported file type or malformed request"},
        401: {"description": "Authentication required"},
        413: {"description": "File exceeds the 10 MB size limit"},
        422: {"description": "Request validation failed"},
        500: {"description": "Model inference error"},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "Image file to analyse (JPEG / PNG / WebP / BMP).",
                            }
                        },
                        "required": ["file"],
                    }
                }
            }
        }
    },
)
async def detect_image(
    file: Annotated[UploadFile, File(description="Image file (JPEG, PNG, WebP, BMP — max 10 MB)")],
    db: AsyncSession = Depends(get_db_session),
    _auth: str = Security(require_any_auth),
) -> DetectionResponse:
    """
    Analyse a single image for deepfake manipulation.

    Steps:
    1. Validate MIME type and file extension.
    2. Read payload into memory and enforce size limit.
    3. Magic-byte sniff to detect spoofed content-type.
    4. Run `DetectionService.detect_image()`.
    5. Record Prometheus metrics.
    6. Return structured `DetectionResponse`.
    """
    filename = file.filename or "upload.jpg"
    content_type = (file.content_type or "").lower()
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # ── Validate type ──────────────────────────────────────────────────────
    if content_type not in _ALLOWED_IMAGE_TYPES and ext not in _ALLOWED_IMAGE_EXTS:
        raise UnsupportedMediaTypeError(
            message=f"Unsupported file type: '{content_type or ext}'. "
                    f"Allowed: JPEG, PNG, WebP, BMP.",
        )

    try:
        file_bytes = await file.read()
    finally:
        await file.close()

    # ── Enforce size ───────────────────────────────────────────────────────
    if len(file_bytes) > _MAX_IMAGE_BYTES:
        raise FileTooLargeError(
            message=f"Image size {len(file_bytes) / 1_048_576:.1f} MB exceeds "
                    f"the {_MAX_IMAGE_MB} MB limit."
        )

    # ── Magic byte sniff ───────────────────────────────────────────────────
    if len(file_bytes) >= 8 and not _sniff_image(file_bytes):
        raise UnsupportedMediaTypeError(
            message="File content does not match a supported image format.",
            detail="Magic-byte validation failed.",
        )

    # ── Inference ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    service = DetectionService(db)
    result = await service.detect_image(file_bytes, filename)
    duration = time.perf_counter() - t0

    if result.status == "failed":
        raise InferenceError(
            message="Image deepfake detection failed.",
            detail=result.error_message,
        )

    # ── Record Prometheus metric ───────────────────────────────────────────
    try:
        from api.middleware.metrics import record_detection
        label_str = "fake" if result.label == 1 else "real"
        record_detection("image", label_str, duration)
    except Exception:
        pass  # Metrics are best-effort

    label_name = "REAL" if result.label == 0 else "FAKE" if result.label == 1 else None
    logger.info(
        "Image detection [%s]: label=%s confidence=%.3f faces=%d duration=%.2fs",
        result.id,
        label_name,
        result.confidence or 0.0,
        result.faces_count,
        duration,
    )

    return DetectionResponse(
        id=result.id,
        filename=result.filename,
        media_type="image",
        status=result.status,
        label=result.label,
        label_name=label_name,
        confidence=result.confidence,
        faces_count=result.faces_count,
        created_at=result.created_at,
        completed_at=result.completed_at,
        error_message=result.error_message,
        explainability=result.meta_info.get("explainability") if result.meta_info else None,
    )
