"""
DeepGuard — repositories/interfaces/detection.py

Abstract interface defining data access operations for detection results.
"""

from abc import ABC, abstractmethod
from database.models import DetectionResultDB


class IDetectionRepository(ABC):
    """Interface for database access of DetectionResultDB objects."""

    @abstractmethod
    async def add(self, result: DetectionResultDB) -> DetectionResultDB:
        """Persist a new detection result record.

        Args:
            result: The DetectionResultDB instance to save.

        Returns:
            The saved DetectionResultDB instance.
        """
        pass

    @abstractmethod
    async def get_by_id(self, result_id: str) -> DetectionResultDB | None:
        """Retrieve a detection result by its unique identifier.

        Args:
            result_id: String UUID.

        Returns:
            DetectionResultDB instance or None if not found.
        """
        pass

    @abstractmethod
    async def get_all(self, limit: int = 100, offset: int = 0) -> list[DetectionResultDB]:
        """Retrieve a paginated list of detection results.

        Args:
            limit: Maximum records to return.
            offset: Number of records to skip.

        Returns:
            List of DetectionResultDB objects.
        """
        pass
