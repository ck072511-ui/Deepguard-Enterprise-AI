"""
DeepGuard — api/exception_handlers.py

Global FastAPI exception handlers.
Maps both domain exceptions and standard HTTP errors to a consistent
JSON envelope:

    {
        "error": {
            "code":       "UNSUPPORTED_MEDIA_TYPE",
            "message":    "The uploaded file type is not supported.",
            "detail":     "...",          // optional
            "request_id": "abc-123"       // from X-Request-ID header
        }
    }
"""

import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError as PydanticValidationError

from core.exceptions.api_exceptions import DeepGuardBaseException

logger = logging.getLogger(__name__)


# ── Helper ────────────────────────────────────────────────────────────────────

def _error_envelope(
    request: Request,
    code: str,
    message: str,
    detail: str | None = None,
    status_code: int = 500,
) -> JSONResponse:
    request_id = request.headers.get("X-Request-ID", "")
    body: dict = {"error": {"code": code, "message": message}}
    if detail:
        body["error"]["detail"] = detail
    if request_id:
        body["error"]["request_id"] = request_id
    return JSONResponse(status_code=status_code, content=body)


# ── Handlers ──────────────────────────────────────────────────────────────────

async def deepguard_exception_handler(
    request: Request, exc: DeepGuardBaseException
) -> JSONResponse:
    """Handle all domain-specific DeepGuard exceptions."""
    logger.warning(
        "[%s] %s — %s",
        request.headers.get("X-Request-ID", "-"),
        exc.error_code,
        exc.message,
    )
    return _error_envelope(
        request,
        code=exc.error_code,
        message=exc.message,
        detail=exc.detail,
        status_code=exc.status_code,
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle Starlette/FastAPI HTTP exceptions with consistent envelope."""
    logger.info(
        "[%s] HTTP %d — %s",
        request.headers.get("X-Request-ID", "-"),
        exc.status_code,
        exc.detail,
    )
    http_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        413: "PAYLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "UNPROCESSABLE_ENTITY",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_SERVER_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    code = http_code_map.get(exc.status_code, "HTTP_ERROR")
    return _error_envelope(
        request,
        code=code,
        message=str(exc.detail),
        status_code=exc.status_code,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic request validation errors with field-level details."""
    errors = exc.errors()
    logger.info(
        "[%s] Validation error — %d field(s) failed",
        request.headers.get("X-Request-ID", "-"),
        len(errors),
    )
    formatted = [
        {
            "field": " → ".join(str(loc) for loc in err["loc"]),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in errors
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_FAILED",
                "message": "Request validation failed.",
                "fields": formatted,
                "request_id": request.headers.get("X-Request-ID", ""),
            }
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled server errors — never leaks tracebacks to clients."""
    logger.exception(
        "[%s] Unhandled exception at %s",
        request.headers.get("X-Request-ID", "-"),
        request.url.path,
    )
    return _error_envelope(
        request,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred. Please try again later.",
        status_code=500,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI application."""
    app.add_exception_handler(DeepGuardBaseException, deepguard_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
