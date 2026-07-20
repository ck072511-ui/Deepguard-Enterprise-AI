"""
DeepGuard — schemas/responses/models.py

Pydantic model for serializing model registry entries.
"""

from datetime import datetime
from pydantic import BaseModel


class ModelVersionResponse(BaseModel):
    """Schema representing the serialized model version configuration."""

    id: str
    name: str
    version: str
    registry_path: str
    active: bool
    created_at: datetime
