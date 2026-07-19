"""
DeepGuard — Deepfake Detection System using Vision Transformers.

This is the root API package exposing the FastAPI application router.

Package Structure:
    api/v1/endpoints/  — Route handlers for each resource
    api/middleware/    — Authentication, logging, rate-limiting middleware
    api/dependencies/  — FastAPI dependency injection providers
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("deepguard")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__: list[str] = ["__version__"]
