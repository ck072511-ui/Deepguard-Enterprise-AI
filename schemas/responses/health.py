"""
DeepGuard — schemas/responses/health.py

Pydantic model for API health status check.
"""

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Schema representing the status indicators for database and model systems."""

    status: str = Field("healthy", description="Overall health status")
    database: str = Field("connected", description="Database connection health")
    model: str = Field("loaded", description="Active model inference capability status")
    mlflow: str = Field("connected", description="MLflow experiment tracker connectivity")
