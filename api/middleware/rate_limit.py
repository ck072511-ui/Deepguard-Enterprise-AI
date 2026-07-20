"""
DeepGuard — api/middleware/rate_limit.py

Sliding window rate limiting middleware for FastAPI.
"""

import time
import logging
from collections import defaultdict
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware enforcing sliding window rate limiting based on client IP."""

    def __init__(self, app, requests_per_minute: int = 60, enabled: bool = True) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.history: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.enabled:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Evict timestamps older than 60 seconds (1 minute window)
        cutoff = now - 60.0
        self.history[client_ip] = [t for t in self.history[client_ip] if t > cutoff]

        if len(self.history[client_ip]) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded for client %s", client_ip)
            return Response(
                content="Rate limit exceeded. Try again in a minute.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="text/plain",
            )

        self.history[client_ip].append(now)
        return await call_next(request)
