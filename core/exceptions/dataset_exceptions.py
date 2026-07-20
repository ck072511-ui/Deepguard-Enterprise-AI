"""
DeepGuard — core/exceptions/dataset_exceptions.py

Domain-level exception hierarchy for all dataset operations.
All custom dataset exceptions inherit from DatasetError so callers
can catch at any granularity level.

Exception Tree:
    DeepGuardError
    └── DatasetError
        ├── DatasetNotFoundError
        ├── DatasetCorruptedError
        ├── DatasetValidationError
        │   ├── DatasetStructureError
        │   └── DatasetIntegrityError
        ├── DatasetDownloadError
        ├── FaceExtractionError
        ├── DatasetSplitError
        └── DatasetVersionError
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DeepGuardError(Exception):
    """Root exception for all DeepGuard errors.

    Args:
        message: Human-readable error description.
        context: Optional dict of additional debugging context.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        """Return formatted string with context if available."""
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}(message={self.message!r}, context={self.context!r})"


class DatasetError(DeepGuardError):
    """Base exception for all dataset-related errors."""


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset root directory or required file does not exist.

    Args:
        path: The path that was not found.
        dataset_name: Name of the dataset being accessed.
    """

    def __init__(self, path: Path | str, dataset_name: str = "") -> None:
        self.path = Path(path)
        self.dataset_name = dataset_name
        msg = f"Dataset not found at '{self.path}'"
        if dataset_name:
            msg = f"[{dataset_name}] {msg}"
        super().__init__(
            msg,
            context={"path": str(self.path), "dataset_name": dataset_name},
        )


class DatasetCorruptedError(DatasetError):
    """Raised when dataset files fail integrity verification (checksum mismatch).

    Args:
        path: The corrupted file path.
        expected_hash: Expected checksum value.
        actual_hash: Computed checksum value.
    """

    def __init__(
        self,
        path: Path | str,
        expected_hash: str = "",
        actual_hash: str = "",
    ) -> None:
        self.path = Path(path)
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Dataset file corrupted: '{self.path}'",
            context={
                "path": str(self.path),
                "expected_hash": expected_hash,
                "actual_hash": actual_hash,
            },
        )


class DatasetValidationError(DatasetError):
    """Raised when dataset fails structural or semantic validation.

    Args:
        message: Description of what validation failed.
        dataset_name: Name of the dataset being validated.
        errors: List of individual validation error messages.
    """

    def __init__(
        self,
        message: str,
        dataset_name: str = "",
        errors: list[str] | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.errors: list[str] = errors or []
        super().__init__(
            message,
            context={
                "dataset_name": dataset_name,
                "error_count": len(self.errors),
                "errors": self.errors[:5],  # first 5 for context
            },
        )


class DatasetStructureError(DatasetValidationError):
    """Raised when the dataset directory structure is invalid or missing required paths."""

    def __init__(
        self,
        dataset_name: str,
        missing_paths: list[str],
    ) -> None:
        self.missing_paths = missing_paths
        super().__init__(
            f"[{dataset_name}] Invalid directory structure. Missing: {missing_paths}",
            dataset_name=dataset_name,
            errors=missing_paths,
        )


class DatasetIntegrityError(DatasetValidationError):
    """Raised when one or more dataset files fail checksum verification."""

    def __init__(
        self,
        dataset_name: str,
        failed_files: list[str],
    ) -> None:
        self.failed_files = failed_files
        super().__init__(
            f"[{dataset_name}] {len(failed_files)} file(s) failed integrity check.",
            dataset_name=dataset_name,
            errors=failed_files,
        )


class DatasetDownloadError(DatasetError):
    """Raised when a dataset download fails.

    Args:
        dataset_name: Name of the dataset being downloaded.
        reason: Human-readable reason for the failure.
        url: The URL that failed (if applicable).
    """

    def __init__(
        self,
        dataset_name: str,
        reason: str,
        url: str = "",
    ) -> None:
        self.dataset_name = dataset_name
        self.reason = reason
        self.url = url
        super().__init__(
            f"[{dataset_name}] Download failed: {reason}",
            context={"dataset_name": dataset_name, "reason": reason, "url": url},
        )


class FaceExtractionError(DatasetError):
    """Raised when face detection or extraction fails on a media file.

    Args:
        source_path: Path to the image or video frame.
        reason: Why extraction failed.
        frame_index: Frame number (for video sources).
    """

    def __init__(
        self,
        source_path: Path | str,
        reason: str,
        frame_index: int = -1,
    ) -> None:
        self.source_path = Path(source_path)
        self.reason = reason
        self.frame_index = frame_index
        ctx: dict[str, Any] = {
            "source_path": str(self.source_path),
            "reason": reason,
        }
        if frame_index >= 0:
            ctx["frame_index"] = frame_index
        super().__init__(
            f"Face extraction failed on '{self.source_path.name}': {reason}",
            context=ctx,
        )


class DatasetSplitError(DatasetError):
    """Raised when train/val/test split cannot be created.

    Args:
        reason: Description of the split failure.
        total_samples: Total sample count at time of failure.
    """

    def __init__(self, reason: str, total_samples: int = 0) -> None:
        self.reason = reason
        super().__init__(
            f"Dataset split failed: {reason}",
            context={"reason": reason, "total_samples": total_samples},
        )


class DatasetVersionError(DatasetError):
    """Raised when dataset versioning operations fail.

    Args:
        reason: Description of the versioning failure.
        version: The version string involved (if known).
    """

    def __init__(self, reason: str, version: str = "") -> None:
        self.reason = reason
        super().__init__(
            f"Dataset versioning error: {reason}",
            context={"reason": reason, "version": version},
        )


class UnsupportedDatasetError(DatasetError):
    """Raised when an unsupported dataset name is requested.

    Args:
        dataset_name: The unsupported dataset name.
        supported: List of supported dataset names.
    """

    def __init__(self, dataset_name: str, supported: list[str]) -> None:
        self.dataset_name = dataset_name
        self.supported = supported
        super().__init__(
            f"Unsupported dataset '{dataset_name}'. Supported: {supported}",
            context={"dataset_name": dataset_name, "supported": supported},
        )


class EmptyDatasetError(DatasetError):
    """Raised when a dataset contains no usable samples after filtering.

    Args:
        dataset_name: Name of the empty dataset.
        filters_applied: Description of filters that were applied.
    """

    def __init__(self, dataset_name: str, filters_applied: str = "") -> None:
        self.dataset_name = dataset_name
        super().__init__(
            f"[{dataset_name}] No samples found after applying filters.",
            context={"dataset_name": dataset_name, "filters": filters_applied},
        )
