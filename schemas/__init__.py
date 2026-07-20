"""
DeepGuard — Schemas package.

Pydantic v2 schemas for API request validation and response serialization.

Packages:
    schemas.requests  — Inbound request validation models
    schemas.responses — Outbound response serialization models
"""

from schemas.responses.detection import DetectionResponse
from schemas.responses.models import ModelVersionResponse
from schemas.responses.health import HealthCheckResponse

__all__ = ["DetectionResponse", "ModelVersionResponse", "HealthCheckResponse"]
