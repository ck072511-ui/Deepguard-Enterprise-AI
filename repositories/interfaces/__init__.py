"""
DeepGuard — repositories/interfaces package.

Exposes interfaces for repository services.
"""

from repositories.interfaces.detection import IDetectionRepository
from repositories.interfaces.model import IModelRepository

__all__ = ["IDetectionRepository", "IModelRepository"]

