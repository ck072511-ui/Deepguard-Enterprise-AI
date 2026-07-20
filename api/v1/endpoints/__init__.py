"""
DeepGuard — api/v1/endpoints/__init__.py

Exports all v1 endpoint routers.
"""

from api.v1.endpoints.health import router as health_router
from api.v1.endpoints.models import router as models_router
from api.v1.endpoints.detect import router as detect_router
from api.v1.endpoints.auth import router as auth_router
from api.v1.endpoints.image_detect import router as image_detect_router
from api.v1.endpoints.video_detect import router as video_detect_router
from api.v1.endpoints.batch_detect import router as batch_detect_router
from api.v1.endpoints.history import router as history_router
from api.v1.endpoints.upload import router as upload_router
from api.v1.endpoints.model_info import router as model_info_router

__all__ = [
    "health_router",
    "models_router",
    "detect_router",
    "auth_router",
    "image_detect_router",
    "video_detect_router",
    "batch_detect_router",
    "history_router",
    "upload_router",
    "model_info_router",
]
