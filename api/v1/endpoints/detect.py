"""
DeepGuard — api/v1/endpoints/detect.py

FastAPI endpoint for uploading media and executing deepfake detection.
"""

import os
import uuid
import shutil
import logging
import yaml

from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_db_session
from services.detection.service import DetectionService
from schemas.responses.detection import DetectionResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def load_upload_config() -> dict:
    """Load configuration defaults from configs/api_config.yaml."""
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "configs" / "api_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("upload", {})
    return {}


@router.post("/detect", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
async def detect_media(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session)
) -> DetectionResponse:
    """Upload an image or video to analyze it for deepfake manipulation."""
    config = load_upload_config()
    allowed_images = config.get(
        "allowed_image_types", ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    )
    allowed_videos = config.get(
        "allowed_video_types", ["video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"]
    )
    upload_dir_str = config.get("upload_dir", "./uploads")
    cleanup = config.get("cleanup_after_processing", True)

    content_type = file.content_type or ""
    filename = file.filename or "uploaded_file"

    service = DetectionService(db)

    # 1. Image detection path
    if content_type in allowed_images or filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        try:
            file_bytes = await file.read()
            # Perform check on size
            max_img_size = config.get("max_image_size_mb", 10) * 1024 * 1024
            if len(file_bytes) > max_img_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Image exceeds size limit of {config.get('max_image_size_mb', 10)} MB.",
                )

            result = await service.detect_image(file_bytes, filename)

            if result.status == "failed":
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Detection failed: {result.error_message}",
                )

            label_name = "REAL" if result.label == 0 else "FAKE" if result.label == 1 else None

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
        finally:
            await file.close()

    # 2. Video detection path
    elif content_type in allowed_videos or filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        # Setup upload path
        upload_path = Path(upload_dir_str)
        upload_path.mkdir(parents=True, exist_ok=True)
        temp_file_path = upload_path / f"temp_{uuid.uuid4()}_{filename}"

        try:
            # Write file to disk chunk-by-chunk to save RAM
            file_size = 0
            max_vid_size = config.get("max_video_size_mb", 200) * 1024 * 1024

            with open(temp_file_path, "wb") as buffer:
                while chunk := await file.read(8192):
                    file_size += len(chunk)
                    if file_size > max_vid_size:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Video exceeds size limit of {config.get('max_video_size_mb', 200)} MB.",
                        )
                    buffer.write(chunk)

            # Perform video detection
            result = await service.detect_video(temp_file_path, filename)

            if result.status == "failed":
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Detection failed: {result.error_message}",
                )

            label_name = "REAL" if result.label == 0 else "FAKE" if result.label == 1 else None

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

        finally:
            await file.close()
            # Clean up temp file
            if cleanup and temp_file_path.exists():
                try:
                    os.remove(temp_file_path)
                except Exception as ex:
                    logger.warning("Could not clean up temp file %s: %s", temp_file_path, str(ex))

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {content_type or 'unknown'}",
        )


@router.get("/detect", response_model=list[DetectionResponse], status_code=status.HTTP_200_OK)
async def get_history(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db_session)
) -> list[DetectionResponse]:
    """Retrieve paginated deepfake detection logs from database."""
    service = DetectionService(db)
    results = await service.detection_repo.get_all(limit=limit, offset=offset)
    return [
        DetectionResponse(
            id=result.id,
            filename=result.filename,
            media_type=result.media_type,
            status=result.status,
            label=result.label,
            label_name="REAL" if result.label == 0 else "FAKE" if result.label == 1 else None,
            confidence=result.confidence,
            faces_count=result.faces_count,
            created_at=result.created_at,
            completed_at=result.completed_at,
            error_message=result.error_message,
            explainability=result.meta_info.get("explainability") if result.meta_info else None,
        )
        for result in results
    ]

