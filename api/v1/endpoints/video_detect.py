"""
DeepGuard — api/v1/endpoints/video_detect.py

Dedicated video deepfake detection endpoint.

POST /api/v1/detect/video
  Stream-upload a video file and receive a deepfake prediction derived
  from sampled frame analysis.

Key features:
  - Streaming chunk-by-chunk write (avoids loading the entire video in RAM)
  - Magic-byte validation for MP4 / AVI / MOV / MKV containers
  - Configurable size limit (default 200 MB)
  - Prometheus metric recording
  - Temporary file cleanup on completion or error
"""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Security, UploadFile, status
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
router = APIRouter(tags=["Video Detection"])

# ── Constants ─────────────────────────────────────────────────────────────────

_ALLOWED_VIDEO_TYPES = frozenset({
    "video/mp4", "video/avi", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/webm",
})
_ALLOWED_VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})
_MAX_VIDEO_MB = 200
_MAX_VIDEO_BYTES = _MAX_VIDEO_MB * 1024 * 1024
_CHUNK_SIZE = 16 * 1024          # 16 KB chunks
_UPLOAD_DIR = Path("./uploads")

# Video container magic bytes
_VIDEO_MAGIC: list[tuple[bytes, str]] = [
    (b"\x00\x00\x00", "video/mp4"),           # MP4 / MOV (ftyp box)
    (b"RIFF", "video/avi"),                    # AVI
    (b"\x1a\x45\xdf\xa3", "video/x-matroska"),# MKV / WebM
]


def _sniff_video(header: bytes) -> bool:
    """Return True if the first bytes match a supported video container."""
    for magic, _ in _VIDEO_MAGIC:
        if header[:len(magic)] == magic:
            return True
    # MP4 'ftyp' box can appear at various byte offsets; accept generously
    if b"ftyp" in header[:32] or b"moov" in header[:32]:
        return True
    return False


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/detect/video",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect deepfake in a video",
    description=(
        "Upload a **video file** (MP4, AVI, MOV, MKV, WebM — max 200 MB) for "
        "deepfake analysis. Frames are sampled and individually analysed by the ViT "
        "model; the final prediction is the majority vote across sampled frames.\n\n"
        "**Authentication:** Requires a valid API key (`X-API-Key` header) or "
        "a JWT Bearer token.\n\n"
        "> ⚠️ Long videos (> 60 s) may take several seconds to process."
    ),
    responses={
        200: {"description": "Detection completed successfully"},
        400: {"description": "Unsupported file type"},
        401: {"description": "Authentication required"},
        413: {"description": f"File exceeds the {_MAX_VIDEO_MB} MB size limit"},
        422: {"description": "Request validation failed"},
        500: {"description": "Model inference error"},
    },
)
async def detect_video(
    file: Annotated[UploadFile, File(description=f"Video file (MP4/AVI/MOV/MKV — max {_MAX_VIDEO_MB} MB)")],
    db: AsyncSession = Depends(get_db_session),
    _auth: str = Security(require_any_auth),
) -> DetectionResponse:
    """
    Analyse a video file for deepfake manipulation.

    Steps:
    1. Validate MIME type and extension.
    2. Stream-write to a temporary file, enforcing size limit per chunk.
    3. Magic-byte sniff the first 32 bytes.
    4. Run `DetectionService.detect_video()`.
    5. Record Prometheus metrics.
    6. Clean up the temp file.
    """
    filename = file.filename or "upload.mp4"
    content_type = (file.content_type or "").lower()
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # ── Validate type ──────────────────────────────────────────────────────
    if content_type not in _ALLOWED_VIDEO_TYPES and ext not in _ALLOWED_VIDEO_EXTS:
        await file.close()
        raise UnsupportedMediaTypeError(
            message=f"Unsupported video type: '{content_type or ext}'. "
                    f"Allowed: MP4, AVI, MOV, MKV, WebM.",
        )

    # ── Stream to disk ─────────────────────────────────────────────────────
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = _UPLOAD_DIR / f"tmp_{uuid.uuid4().hex}_{filename}"
    file_size = 0
    first_chunk: bytes | None = None

    try:
        with open(temp_path, "wb") as buf:
            while chunk := await file.read(_CHUNK_SIZE):
                file_size += len(chunk)
                if file_size > _MAX_VIDEO_BYTES:
                    raise FileTooLargeError(
                        message=f"Video exceeds the {_MAX_VIDEO_MB} MB size limit "
                                f"(received > {file_size / 1_048_576:.0f} MB)."
                    )
                if first_chunk is None:
                    first_chunk = chunk
                buf.write(chunk)
    finally:
        await file.close()

    # ── Magic byte sniff ───────────────────────────────────────────────────
    if first_chunk and not _sniff_video(first_chunk[:32]):
        _cleanup(temp_path)
        raise UnsupportedMediaTypeError(
            message="File content does not match a supported video container.",
            detail="Magic-byte validation failed.",
        )

    # ── Inference ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        service = DetectionService(db)
        result = await service.detect_video(temp_path, filename)
    finally:
        _cleanup(temp_path)

    duration = time.perf_counter() - t0

    if result.status == "failed":
        raise InferenceError(
            message="Video deepfake detection failed.",
            detail=result.error_message,
        )

    # ── Prometheus ─────────────────────────────────────────────────────────
    try:
        from api.middleware.metrics import record_detection
        label_str = "fake" if result.label == 1 else "real"
        record_detection("video", label_str, duration)
    except Exception:
        pass

    label_name = "REAL" if result.label == 0 else "FAKE" if result.label == 1 else None
    logger.info(
        "Video detection [%s]: label=%s confidence=%.3f faces=%d duration=%.2fs",
        result.id,
        label_name,
        result.confidence or 0.0,
        result.faces_count,
        duration,
    )

    return DetectionResponse(
        id=result.id,
        filename=result.filename,
        media_type="video",
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cleanup(path: Path) -> None:
    try:
        if path.exists():
            os.remove(path)
    except Exception as exc:
        logger.warning("Could not delete temp file %s: %s", path, exc)
