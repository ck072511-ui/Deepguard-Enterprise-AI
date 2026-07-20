"""
DeepGuard — database/models.py

SQLAlchemy 2.0 ORM model definitions for DeepGuard domain entities.
"""

from datetime import datetime
from typing import Any
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base class for SQLAlchemy models."""
    pass


class DetectionResultDB(Base):
    """Database representation of an image/video deepfake detection run."""

    __tablename__ = "detection_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(50), default="image")
    status: Mapped[str] = mapped_column(String(50), default="processing")
    label: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 0 = REAL, 1 = FAKE
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    faces_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    meta_info: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class ModelVersionDB(Base):
    """Database representation of registered and tracked ViT models."""

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    registry_path: Mapped[str] = mapped_column(String(500), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
