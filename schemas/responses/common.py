"""
DeepGuard — schemas/responses/common.py

Shared response envelope schemas used across the DeepGuard API.
"""

from __future__ import annotations
from datetime import datetime
from typing import Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated list wrapper."""

    items: list[T] = Field(..., description="Page of results")
    total: int = Field(..., description="Total number of matching records")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Number of results per page")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether a next page exists")
    has_prev: bool = Field(..., description="Whether a previous page exists")


class ErrorDetail(BaseModel):
    """Structured error detail inside an error envelope."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    detail: str | None = Field(None, description="Additional error context")
    request_id: str | None = Field(None, description="Request tracing ID")


class ErrorResponse(BaseModel):
    """Top-level error envelope returned on all non-2xx responses."""
    error: ErrorDetail


class UploadInfo(BaseModel):
    """Metadata returned for a newly uploaded file."""
    upload_id: str = Field(..., description="Unique upload identifier")
    filename: str = Field(..., description="Original file name")
    size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type of the uploaded file")
    media_type: str = Field(..., description="'image' or 'video'")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
