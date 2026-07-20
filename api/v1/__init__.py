"""
DeepGuard — api/v1/__init__.py

Version 1 of the REST API. All routes are prefixed with /api/v1.
"""

from fastapi import APIRouter

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

v1_router = APIRouter()

# ── Core ──────────────────────────────────────────────────────────────────────
v1_router.include_router(health_router, tags=["Health"])

# ── Authentication ────────────────────────────────────────────────────────────
v1_router.include_router(auth_router)

# ── Detection ─────────────────────────────────────────────────────────────────
v1_router.include_router(detect_router, tags=["Detection (Legacy)"])
v1_router.include_router(image_detect_router)
v1_router.include_router(video_detect_router)
v1_router.include_router(batch_detect_router)

# ── History ───────────────────────────────────────────────────────────────────
v1_router.include_router(history_router)

# ── Upload ────────────────────────────────────────────────────────────────────
v1_router.include_router(upload_router)

# ── Models ────────────────────────────────────────────────────────────────────
v1_router.include_router(models_router, tags=["Model Registry"])
v1_router.include_router(model_info_router)

__all__ = ["v1_router"]
