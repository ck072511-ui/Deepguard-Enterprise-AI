"""
DeepGuard — Backend application package.

Entry point for the FastAPI application. This package wires together
all layers: API routes, services, repositories, and database sessions.
"""

from backend.main import create_application

__all__ = ["create_application"]

