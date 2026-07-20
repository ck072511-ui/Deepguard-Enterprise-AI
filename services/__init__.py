"""
DeepGuard — Services layer.

Houses all business-logic service classes that orchestrate use cases,
coordinate repositories, and call ML inference pipelines.

Services:
    services.detection        — Deepfake detection inference service
    services.training         — Model training orchestration service
    services.model_registry   — MLflow model registry service
"""

from services.detection.service import DetectionService

__all__ = ["DetectionService"]
