"""
DeepGuard — api/middleware/logging.py

Structured request/response logging middleware.

Logs every request with:
  - Unique X-Request-ID (generated if absent)
  - Method, path, query string
  - Client IP
  - Response status code
  - Wall-clock duration in milliseconds

Log format (JSON-compatible) example:
  REQUEST  GET /api/v1/health [req_id=abc-123 ip=127.0.0.1]
  RESPONSE 200 in 4.2ms       [req_id=abc-123]
"""

import time
import uuid
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("deepguard.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that attaches a request ID and logs every HTTP request."""

    SKIP_PATHS = frozenset({"/metrics", "/favicon.ico"})

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 1. Attach / generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # 2. Mutate scope state so downstream handlers can read it
        request.state.request_id = request_id

        skip = request.url.path in self.SKIP_PATHS
        start = time.perf_counter()

        if not skip:
            client_ip = (request.client.host if request.client else "unknown")
            qs = f"?{request.url.query}" if request.url.query else ""
            logger.info(
                "REQUEST  %s %s%s [req_id=%s ip=%s]",
                request.method,
                request.url.path,
                qs,
                request_id,
                client_ip,
            )

        try:
            response: Response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            if not skip:
                logger.error(
                    "RESPONSE 500 in %.1fms [req_id=%s] (unhandled exception)",
                    duration_ms,
                    request_id,
                )
            raise

        duration_ms = (time.perf_counter() - start) * 1000

        # 3. Inject X-Request-ID into response headers
        response.headers["X-Request-ID"] = request_id

        if not skip:
            log_fn = logger.warning if response.status_code >= 400 else logger.info
            log_fn(
                "RESPONSE %d in %.1fms [req_id=%s]",
                response.status_code,
                duration_ms,
                request_id,
            )

        return response
