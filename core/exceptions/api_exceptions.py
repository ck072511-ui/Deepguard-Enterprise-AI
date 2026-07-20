"""
DeepGuard — core/exceptions/api_exceptions.py

Domain-specific API exception hierarchy.
All exceptions map to HTTP status codes via the global exception handlers
registered in backend/main.py.
"""

from __future__ import annotations


class DeepGuardBaseException(Exception):
    """Root exception for all application-level errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, detail: str | None = None) -> None:
        self.message = message or self.__class__.message
        self.detail = detail
        super().__init__(self.message)


# ── 400 Bad Request ───────────────────────────────────────────────────────────

class BadRequestError(DeepGuardBaseException):
    status_code = 400
    error_code = "BAD_REQUEST"
    message = "The request is malformed or contains invalid parameters."


class UnsupportedMediaTypeError(DeepGuardBaseException):
    status_code = 415
    error_code = "UNSUPPORTED_MEDIA_TYPE"
    message = "The uploaded file type is not supported."


class FileTooLargeError(DeepGuardBaseException):
    status_code = 413
    error_code = "FILE_TOO_LARGE"
    message = "The uploaded file exceeds the maximum allowed size."


class ValidationFailedError(DeepGuardBaseException):
    status_code = 422
    error_code = "VALIDATION_FAILED"
    message = "Request validation failed."


# ── 401 / 403 Auth ────────────────────────────────────────────────────────────

class UnauthorizedError(DeepGuardBaseException):
    status_code = 401
    error_code = "UNAUTHORIZED"
    message = "Authentication credentials are missing or invalid."


class ForbiddenError(DeepGuardBaseException):
    status_code = 403
    error_code = "FORBIDDEN"
    message = "You do not have permission to perform this action."


# ── 404 Not Found ─────────────────────────────────────────────────────────────

class NotFoundError(DeepGuardBaseException):
    status_code = 404
    error_code = "NOT_FOUND"
    message = "The requested resource was not found."


# ── 429 Rate Limit ────────────────────────────────────────────────────────────

class RateLimitExceededError(DeepGuardBaseException):
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
    message = "Too many requests. Please slow down."


# ── 5xx Server Errors ─────────────────────────────────────────────────────────

class InferenceError(DeepGuardBaseException):
    status_code = 500
    error_code = "INFERENCE_ERROR"
    message = "Model inference failed."


class ModelNotLoadedError(DeepGuardBaseException):
    status_code = 503
    error_code = "MODEL_NOT_LOADED"
    message = "The detection model is not loaded or unavailable."


class StorageError(DeepGuardBaseException):
    status_code = 500
    error_code = "STORAGE_ERROR"
    message = "File storage operation failed."


class BatchTooLargeError(DeepGuardBaseException):
    status_code = 413
    error_code = "BATCH_TOO_LARGE"
    message = "Batch size exceeds the maximum allowed limit."
