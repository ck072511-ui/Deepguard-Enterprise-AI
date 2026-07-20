"""
DeepGuard — api/v1/endpoints/batch_detect.py

Batch image deepfake detection endpoint.

POST /api/v1/detect/batch
  Upload up to 32 images in a single request and receive an aggregated
  result list.  Each file is processed independently; partial failures
  are reported per-item without aborting the whole batch.

Design decisions:
  - Sequential processing (avoids GPU memory contention)
  - Per-item error capture so one bad file doesn't fail the whole batch
  - Returns a BatchDetectionResponse with summary statistics
  - Configurable batch-size ceiling (default 32 from api_config.yaml)
"""

import asyncio
import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Security, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.security import require_any_auth
from core.exceptions.api_exceptions import BatchTooLargeError, UnsupportedMediaTypeError
from database.session import get_db_session
from schemas.responses.detection import DetectionResponse
from services.detection.service import DetectionService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Batch Detection"])

_MAX_BATCH_SIZE = 32
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_ALLOWED_IMAGE_TYPES = frozenset({
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp",
})
_ALLOWED_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


# ── Response schema ───────────────────────────────────────────────────────────

class BatchItemResult(BaseModel):
    """Result for a single file in the batch."""
    index: int = Field(..., description="Zero-based position in the uploaded batch")
    filename: str = Field(..., description="Original filename")
    success: bool = Field(..., description="Whether detection succeeded")
    result: DetectionResponse | None = Field(None, description="Detection result (if successful)")
    error: str | None = Field(None, description="Error message (if failed)")


class BatchDetectionResponse(BaseModel):
    """Aggregated response for a batch detection request."""
    total: int = Field(..., description="Total files submitted")
    succeeded: int = Field(..., description="Files successfully processed")
    failed: int = Field(..., description="Files that failed processing")
    real_count: int = Field(..., description="Files classified as REAL")
    fake_count: int = Field(..., description="Files classified as FAKE")
    avg_confidence: float | None = Field(None, description="Average FAKE confidence across successful detections")
    duration_seconds: float = Field(..., description="Total wall-clock time in seconds")
    results: list[BatchItemResult] = Field(..., description="Per-file detection results")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/detect/batch",
    response_model=BatchDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch deepfake detection (images)",
    description=(
        "Upload **up to 32 images** in a single request. Each image is processed "
        "independently — a failure on one file does not abort the rest.\n\n"
        "The response includes per-file results and aggregated statistics "
        "(real count, fake count, average confidence, total duration).\n\n"
        "**Authentication:** Requires a valid API key (`X-API-Key` header) or "
        "a JWT Bearer token."
    ),
    responses={
        200: {"description": "Batch processed (check per-item `success` flag for partial failures)"},
        401: {"description": "Authentication required"},
        413: {"description": f"Batch exceeds the {_MAX_BATCH_SIZE}-file limit"},
        422: {"description": "Request validation failed"},
    },
)
async def detect_batch(
    files: Annotated[
        list[UploadFile],
        File(description=f"Up to {_MAX_BATCH_SIZE} image files (JPEG/PNG/WebP/BMP, 10 MB each)"),
    ],
    db: AsyncSession = Depends(get_db_session),
    _auth: str = Security(require_any_auth),
) -> BatchDetectionResponse:
    """
    Process multiple images in a single batch request.

    Validation:
      - Batch must not exceed _MAX_BATCH_SIZE files.
      - Each file is type-checked independently.

    Processing:
      - Sequential inference (GPU safety).
      - Per-item exceptions are captured and reported; the batch continues.
    """
    if len(files) > _MAX_BATCH_SIZE:
        raise BatchTooLargeError(
            message=f"Batch contains {len(files)} files; the maximum is {_MAX_BATCH_SIZE}."
        )

    t_batch_start = time.perf_counter()
    service = DetectionService(db)
    batch_results: list[BatchItemResult] = []
    confidences: list[float] = []
    real_count = fake_count = succeeded = failed = 0

    for idx, upload_file in enumerate(files):
        filename = upload_file.filename or f"file_{idx}.jpg"
        item = await _process_single(service, idx, filename, upload_file)
        batch_results.append(item)

        if item.success and item.result:
            succeeded += 1
            if item.result.label == 0:
                real_count += 1
            elif item.result.label == 1:
                fake_count += 1
            if item.result.confidence is not None:
                confidences.append(item.result.confidence)
        else:
            failed += 1

    duration = time.perf_counter() - t_batch_start
    avg_conf = sum(confidences) / len(confidences) if confidences else None

    logger.info(
        "Batch detection: total=%d succeeded=%d failed=%d real=%d fake=%d duration=%.2fs",
        len(files), succeeded, failed, real_count, fake_count, duration,
    )

    # Prometheus — record one aggregate metric
    try:
        from api.middleware.metrics import record_detection
        record_detection("batch_image", f"fake={fake_count}", duration)
    except Exception:
        pass

    return BatchDetectionResponse(
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        real_count=real_count,
        fake_count=fake_count,
        avg_confidence=avg_conf,
        duration_seconds=round(duration, 3),
        results=batch_results,
    )


async def _process_single(
    service: DetectionService,
    idx: int,
    filename: str,
    upload: UploadFile,
) -> BatchItemResult:
    """Process a single file within the batch, capturing any errors gracefully."""
    content_type = (upload.content_type or "").lower()
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        # Type check
        if content_type not in _ALLOWED_IMAGE_TYPES and ext not in _ALLOWED_IMAGE_EXTS:
            return BatchItemResult(
                index=idx,
                filename=filename,
                success=False,
                error=f"Unsupported type: '{content_type or ext}'",
            )

        file_bytes = await upload.read()

        # Size check
        if len(file_bytes) > _MAX_IMAGE_BYTES:
            return BatchItemResult(
                index=idx,
                filename=filename,
                success=False,
                error=f"File too large: {len(file_bytes) / 1_048_576:.1f} MB (max 10 MB)",
            )

        result = await service.detect_image(file_bytes, filename)

        if result.status == "failed":
            return BatchItemResult(
                index=idx,
                filename=filename,
                success=False,
                error=result.error_message or "Inference failed",
            )

        label_name = "REAL" if result.label == 0 else "FAKE" if result.label == 1 else None
        return BatchItemResult(
            index=idx,
            filename=filename,
            success=True,
            result=DetectionResponse(
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
            ),
        )

    except Exception as exc:
        logger.warning("Batch item %d (%s) failed: %s", idx, filename, exc)
        return BatchItemResult(
            index=idx,
            filename=filename,
            success=False,
            error=str(exc),
        )
    finally:
        await upload.close()
