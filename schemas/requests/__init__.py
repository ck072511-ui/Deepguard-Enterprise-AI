"""
DeepGuard — schemas/requests/__init__.py

Exports all inbound request schemas.
"""

from schemas.requests.detection import HistoryQueryParams, BatchQueryParams

__all__ = ["HistoryQueryParams", "BatchQueryParams"]
