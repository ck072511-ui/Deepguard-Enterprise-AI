"""
DeepGuard — schemas/responses/detection.py

Pydantic model for serializing deepfake detection response.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ExplainabilityInfo(BaseModel):
    """Schema representing Explainable AI metrics and visualization assets."""

    real_probability: float = Field(..., description="Real class probability")
    fake_probability: float = Field(..., description="Fake class probability")
    confidence_score: float = Field(..., description="Classification confidence score")
    explanation: str = Field(..., description="Natural language prediction explanation")
    heatmap_b64: Optional[str] = Field(None, description="Base64 encoded Jpeg image of the blended heatmap overlay")
    attention_b64: Optional[str] = Field(None, description="Base64 encoded Jpeg image of the raw attention map")
    gradcam_b64: Optional[str] = Field(None, description="Base64 encoded Jpeg image of the GradCAM activation map")


class DetectionResponse(BaseModel):
    """Schema representing the serialized deepfake detection task output."""

    id: str = Field(..., description="Unique task identifier")
    filename: str = Field(..., description="Name of the processed file")
    media_type: str = Field(..., description="Type of media (image/video)")
    status: str = Field(..., description="Processing status (processing, completed, failed)")
    label: int | None = Field(None, description="Predicted label (0 = REAL, 1 = FAKE)")
    label_name: str | None = Field(None, description="Predicted label name (REAL, FAKE)")
    confidence: float | None = Field(None, description="Prediction probability of the FAKE class")
    faces_count: int = Field(..., description="Number of faces detected in the media")
    created_at: datetime = Field(..., description="Timestamp of when the request was received")
    completed_at: datetime | None = Field(None, description="Timestamp of when prediction completed")
    error_message: str | None = Field(None, description="Error details if status is failed")
    explainability: Optional[ExplainabilityInfo] = Field(None, description="Explainable AI results")

