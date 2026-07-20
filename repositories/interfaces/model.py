"""
DeepGuard — repositories/interfaces/model.py

Abstract interface defining data access operations for registered model versions.
"""

from abc import ABC, abstractmethod
from database.models import ModelVersionDB


class IModelRepository(ABC):
    """Interface for database access of ModelVersionDB objects."""

    @abstractmethod
    async def add(self, model: ModelVersionDB) -> ModelVersionDB:
        """Register a new model version.

        Args:
            model: The ModelVersionDB instance to save.

        Returns:
            The saved ModelVersionDB instance.
        """
        pass

    @abstractmethod
    async def get_by_id(self, model_id: str) -> ModelVersionDB | None:
        """Retrieve a model version by its unique identifier.

        Args:
            model_id: String UUID.

        Returns:
            ModelVersionDB instance or None if not found.
        """
        pass

    @abstractmethod
    async def get_active(self) -> ModelVersionDB | None:
        """Retrieve the currently active model version used for inference.

        Returns:
            Active ModelVersionDB instance or None if no model is set active.
        """
        pass

    @abstractmethod
    async def set_active(self, model_id: str) -> ModelVersionDB | None:
        """Mark a model version as active and deactivate all other model versions.

        Args:
            model_id: The ID of the model version to activate.

        Returns:
            The activated ModelVersionDB instance or None if not found.
        """
        pass

    @abstractmethod
    async def get_all(self) -> list[ModelVersionDB]:
        """Retrieve all registered model versions.

        Returns:
            List of all ModelVersionDB objects.
        """
        pass
