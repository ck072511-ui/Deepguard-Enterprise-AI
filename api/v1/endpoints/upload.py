"""
DeepGuard — api/v1/endpoints/upload.py

Standalone file upload endpoint (pre-upload without immediate detection).

POST /api/v1/upload
  Upload a file for storage, receive a stable upload_id that can be
  referenced by downstream detection endpoints.

GET  /api/v1/upload/{upload_id}
  Retrieve metadata for a previously uploaded file.

DELETE /api/v1/upload/{upload_id}
  Delete a staged upload before detection.

Design:
  - Files are stored under ./uploads/<upload_id>_<filename>
  - Upload metadata tracked in-memory (extend to DB for persistence)
  - Supports both images and videos
  - Returns UploadInfo schema with id, size, mime type
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Security, UploadFile, status
from pydantic import BaseModel, Field

from api.auth.security import require_any_auth
from core.exceptions.api_exceptions import FileTooLargeError, UnsupportedMediaTypeError
from schemas.responses.common import UploadInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["Upload Management"])

_UPLOAD_DIR = Path("./uploads")
_MAX_BYTES = 512 * 1024 * 1024        # 512 MB absolute ceiling
_CHUNK_SIZE = 16 * 1024

_ALLOWED_TYPES = frozenset({
    # Images
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp",
    # Videos
    "video/mp4", "video/avi", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/webm",
})

_ALLOWED_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
})

# In-memory registry (replace with DB table for production)
_upload_registry: dict[str, UploadInfo] = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=UploadInfo,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file",
    description=(
        "Stage a file (image or video) for later deepfake detection. "
        "Returns a unique `upload_id` that can be used to reference the file.\n\n"
        "**Limits:** 10 MB for images, 200 MB for videos (512 MB absolute ceiling).\n\n"
        "**Authentication:** Requires a valid API key or JWT Bearer token."
    ),
    responses={
        201: {"description": "File uploaded and staged"},
        401: {"description": "Authentication required"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
    },
)
async def upload_file(
    file: Annotated[UploadFile, File(description="Image or video file to upload")],
    _auth: str = Security(require_any_auth),
) -> UploadInfo:
    """Stage a file upload without triggering detection."""
    filename = file.filename or "uploaded_file"
    content_type = (file.content_type or "").lower()
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if content_type not in _ALLOWED_TYPES and ext not in _ALLOWED_EXTS:
        await file.close()
        raise UnsupportedMediaTypeError(
            message=f"Unsupported file type: '{content_type or ext}'."
        )

    is_image = content_type.startswith("image") or ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    max_bytes = 10 * 1024 * 1024 if is_image else 200 * 1024 * 1024
    media_type = "image" if is_image else "video"

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = str(uuid.uuid4())
    dest_path = _UPLOAD_DIR / f"{upload_id}_{filename}"
    file_size = 0

    try:
        with open(dest_path, "wb") as buf:
            while chunk := await file.read(_CHUNK_SIZE):
                file_size += len(chunk)
                if file_size > max_bytes:
                    raise FileTooLargeError(
                        message=f"File exceeds the {'10' if is_image else '200'} MB limit."
                    )
                buf.write(chunk)
    except FileTooLargeError:
        _safe_delete(dest_path)
        raise
    finally:
        await file.close()

    info = UploadInfo(
        upload_id=upload_id,
        filename=filename,
        size_bytes=file_size,
        content_type=content_type or f"application/octet-stream",
        media_type=media_type,
        uploaded_at=datetime.now(timezone.utc),
    )
    _upload_registry[upload_id] = info
    logger.info("Upload staged: id=%s filename=%s size=%d bytes", upload_id, filename, file_size)
    return info


@router.get(
    "/{upload_id}",
    response_model=UploadInfo,
    summary="Get upload metadata",
    description="Retrieve metadata for a previously staged upload.",
    responses={404: {"description": "Upload not found"}},
)
async def get_upload(
    upload_id: str,
    _auth: str = Security(require_any_auth),
) -> UploadInfo:
    """Return metadata for a staged upload by its ID."""
    info = _upload_registry.get(upload_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload '{upload_id}' not found.",
        )
    return info


@router.delete(
    "/{upload_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete staged upload",
    description="Delete a staged upload file before detection.",
    responses={
        204: {"description": "Upload deleted"},
        404: {"description": "Upload not found"},
    },
)
async def delete_upload(
    upload_id: str,
    _auth: str = Security(require_any_auth),
) -> None:
    """Delete a staged upload and its on-disk file."""
    info = _upload_registry.get(upload_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload '{upload_id}' not found.",
        )
    # Delete file from disk
    for f in _UPLOAD_DIR.glob(f"{upload_id}_*"):
        _safe_delete(f)
    del _upload_registry[upload_id]
    logger.info("Upload deleted: id=%s", upload_id)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _safe_delete(path: Path) -> None:
    try:
        if path.exists():
            os.remove(path)
    except Exception as exc:
        logger.warning("Could not delete %s: %s", path, exc)
