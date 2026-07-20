"""
DeepGuard — api/v1/endpoints/history.py

Prediction history endpoints with full filtering, sorting, and pagination.

GET  /api/v1/history          — paginated list of detection records
GET  /api/v1/history/{id}     — single record by ID
DELETE /api/v1/history/{id}   — soft-delete / remove a record
GET  /api/v1/history/stats    — aggregated statistics (totals, accuracy, etc.)
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Security, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.security import require_any_auth
from database.models import DetectionResultDB
from database.session import get_db_session
from schemas.responses.common import PaginatedResponse
from schemas.responses.detection import DetectionResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/history", tags=["Prediction History"])


# ── Aggregated stats schema ────────────────────────────────────────────────────

from pydantic import BaseModel, Field


class DetectionStats(BaseModel):
    """Aggregated statistics over all detection records."""
    total: int = Field(..., description="Total detections stored")
    real: int = Field(..., description="REAL predictions")
    fake: int = Field(..., description="FAKE predictions")
    pending: int = Field(..., description="Still-processing records")
    failed: int = Field(..., description="Failed detections")
    images: int = Field(..., description="Image files processed")
    videos: int = Field(..., description="Video files processed")
    avg_confidence: float | None = Field(None, description="Average confidence across completed records")
    fake_rate: float | None = Field(None, description="Fraction classified as FAKE (0–1)")
    avg_latency: float | None = Field(None, description="Average latency in milliseconds")



# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_response(row: DetectionResultDB) -> DetectionResponse:
    label_name = "REAL" if row.label == 0 else "FAKE" if row.label == 1 else None
    return DetectionResponse(
        id=row.id,
        filename=row.filename,
        media_type=row.media_type,
        status=row.status,
        label=row.label,
        label_name=label_name,
        confidence=row.confidence,
        faces_count=row.faces_count,
        created_at=row.created_at,
        completed_at=row.completed_at,
        error_message=row.error_message,
        explainability=row.meta_info.get("explainability") if row.meta_info else None,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=PaginatedResponse[DetectionResponse],
    summary="List prediction history",
    description=(
        "Retrieve a **paginated, filtered, and sorted** list of deepfake detection "
        "records. Supports filtering by media type, label, and status."
    ),
)
async def list_history(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    media_type: str | None = Query(None, pattern=r"^(image|video)$", description="Filter by media type"),
    label: int | None = Query(None, ge=0, le=1, description="Filter: 0=REAL, 1=FAKE"),
    record_status: str | None = Query(None, alias="status", pattern=r"^(processing|completed|failed)$"),
    sort_by: str = Query("created_at", pattern=r"^(created_at|completed_at|confidence)$"),
    order: str = Query("desc", pattern=r"^(asc|desc)$"),
    db: AsyncSession = Depends(get_db_session),
    _auth: str = Security(require_any_auth),
) -> PaginatedResponse[DetectionResponse]:
    """Return paginated detection records with optional filters."""
    # Build query
    q = select(DetectionResultDB)
    if media_type:
        q = q.where(DetectionResultDB.media_type == media_type)
    if label is not None:
        q = q.where(DetectionResultDB.label == label)
    if record_status:
        q = q.where(DetectionResultDB.status == record_status)

    # Count total
    count_q = select(func.count()).select_from(q.subquery())
    total: int = (await db.execute(count_q)).scalar_one()

    # Sorting
    sort_col = getattr(DetectionResultDB, sort_by, DetectionResultDB.created_at)
    q = q.order_by(sort_col.desc() if order == "desc" else sort_col.asc())

    # Pagination
    offset = (page - 1) * page_size
    q = q.offset(offset).limit(page_size)

    rows = (await db.execute(q)).scalars().all()
    items = [_to_response(r) for r in rows]

    pages = max(1, -(-total // page_size))  # ceil division
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
        has_next=page < pages,
        has_prev=page > 1,
    )


@router.get(
    "/stats",
    response_model=DetectionStats,
    summary="Detection statistics",
    description="Return aggregated counts and metrics across all stored detection records.",
)
async def get_stats(
    db: AsyncSession = Depends(get_db_session),
    _auth: str = Security(require_any_auth),
) -> DetectionStats:
    """Compute aggregated statistics over the detection_results table."""
    q = select(DetectionResultDB)
    rows = (await db.execute(q)).scalars().all()

    total = len(rows)
    real = sum(1 for r in rows if r.label == 0)
    fake = sum(1 for r in rows if r.label == 1)
    pending = sum(1 for r in rows if r.status == "processing")
    failed = sum(1 for r in rows if r.status == "failed")
    images = sum(1 for r in rows if r.media_type == "image")
    videos = sum(1 for r in rows if r.media_type == "video")

    confidences = [r.confidence for r in rows if r.confidence is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None
    completed = real + fake
    fake_rate = fake / completed if completed > 0 else None

    durations = []
    for r in rows:
        if r.completed_at and r.created_at and r.status == "completed":
            c_at = r.completed_at
            cr_at = r.created_at
            if c_at.tzinfo != cr_at.tzinfo:
                c_at = c_at.replace(tzinfo=None)
                cr_at = cr_at.replace(tzinfo=None)
            durations.append((c_at - cr_at).total_seconds() * 1000)
    avg_latency = sum(durations) / len(durations) if durations else None

    return DetectionStats(
        total=total,
        real=real,
        fake=fake,
        pending=pending,
        failed=failed,
        images=images,
        videos=videos,
        avg_confidence=avg_confidence,
        fake_rate=fake_rate,
        avg_latency=avg_latency,
    )


@router.get(
    "/{detection_id}",
    response_model=DetectionResponse,
    summary="Get detection by ID",
    description="Retrieve a single detection record by its unique ID.",
    responses={404: {"description": "Record not found"}},
)
async def get_detection(
    detection_id: str,
    db: AsyncSession = Depends(get_db_session),
    _auth: str = Security(require_any_auth),
) -> DetectionResponse:
    """Fetch a single detection result by ID."""
    row = await db.get(DetectionResultDB, detection_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection record '{detection_id}' not found.",
        )
    return _to_response(row)


@router.delete(
    "/{detection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a detection record",
    description="Permanently delete a detection record from the database.",
    responses={
        204: {"description": "Record deleted"},
        404: {"description": "Record not found"},
    },
)
async def delete_detection(
    detection_id: str,
    db: AsyncSession = Depends(get_db_session),
    _auth: str = Security(require_any_auth),
) -> None:
    """Delete a detection record by ID."""
    row = await db.get(DetectionResultDB, detection_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection record '{detection_id}' not found.",
        )
    await db.delete(row)
    await db.commit()
    logger.info("Deleted detection record '%s'", detection_id)
