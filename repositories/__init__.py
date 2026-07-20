"""
DeepGuard — Repositories layer.

Implements the Repository Pattern for data persistence.
All database queries are isolated here, away from business logic.

Packages:
    repositories.interfaces — Abstract base repository interfaces
    repositories.sqlite     — SQLite/SQLAlchemy concrete implementations
"""

from repositories.interfaces.detection import IDetectionRepository
from repositories.interfaces.model import IModelRepository
from repositories.sqlite.detection import SQLiteDetectionRepository
from repositories.sqlite.model import SQLiteModelRepository

__all__ = [
    "IDetectionRepository",
    "IModelRepository",
    "SQLiteDetectionRepository",
    "SQLiteModelRepository",
]
