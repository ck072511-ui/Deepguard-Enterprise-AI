"""
DeepGuard — schemas/requests/detection.py

Pydantic request parameter models for detection endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class HistoryQueryParams(BaseModel):
    """Query parameters for GET /detect (history listing)."""

    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(20, ge=1, le=100, description="Number of results per page")
    media_type: str | None = Field(
        None,
        pattern=r"^(image|video)$",
        description="Filter by media type: 'image' or 'video'",
    )
    label: int | None = Field(
        None,
        ge=0,
        le=1,
        description="Filter by label: 0 = REAL, 1 = FAKE",
    )
    status: str | None = Field(
        None,
        pattern=r"^(processing|completed|failed)$",
        description="Filter by processing status",
    )
    sort_by: str = Field(
        "created_at",
        pattern=r"^(created_at|completed_at|confidence)$",
        description="Sort field",
    )
    order: str = Field(
        "desc",
        pattern=r"^(asc|desc)$",
        description="Sort direction",
    )

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class BatchQueryParams(BaseModel):
    """Query parameters shared across batch detection requests."""

    max_faces: int = Field(
        1,
        ge=1,
        le=10,
        description="Maximum faces to analyse per image (higher = slower)",
    )
    return_embeddings: bool = Field(
        False,
        description="Include face embedding vectors in the response",
    )
