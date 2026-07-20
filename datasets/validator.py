"""
DeepGuard — datasets/validator.py

Dataset structure and integrity validator implementing IDatasetValidator.

Checks:
  - Required directory existence
  - File extension compliance
  - Minimum file size
  - Optional checksum verification (SHA-256)
  - Class balance warnings
  - Generates a structured validation report
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core.domain.entities.dataset_entity import DatasetName, Label, SampleEntity
from core.exceptions.dataset_exceptions import (
    DatasetIntegrityError,
    DatasetStructureError,
    DatasetValidationError,
)
from core.interfaces.dataset_interface import IDatasetValidator

logger = logging.getLogger(__name__)

# Minimum acceptable file size in bytes
_MIN_FILE_SIZE_BYTES = 1024

_ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
}

# Required directories per dataset type
_REQUIRED_STRUCTURE: dict[DatasetName, list[str]] = {
    DatasetName.CELEB_DF: ["Celeb-real", "Celeb-synthesis"],
    DatasetName.FF_PLUS_PLUS: [
        "original_sequences",
        "manipulated_sequences",
    ],
    DatasetName.DFDC: [],  # Dynamic — just needs at least one part dir
    DatasetName.CUSTOM: [],  # Flexible — checked at load time
}


class DatasetValidator(IDatasetValidator):
    """Validates dataset structure, file integrity, and class balance.

    Args:
        min_file_size_bytes: Minimum file size to consider valid.
        check_readable:      If True, attempt to open each file.
        allowed_extensions:  Set of allowed file extensions.
    """

    def __init__(
        self,
        min_file_size_bytes: int = _MIN_FILE_SIZE_BYTES,
        check_readable: bool = False,
        allowed_extensions: set[str] | None = None,
    ) -> None:
        self._min_size = min_file_size_bytes
        self._check_readable = check_readable
        self._allowed_extensions = allowed_extensions or _ALLOWED_EXTENSIONS

    # ------------------------------------------------------------------
    # IDatasetValidator interface
    # ------------------------------------------------------------------

    def validate_structure(
        self,
        root: Path,
        dataset_name: DatasetName,
    ) -> list[str]:
        """Validate the required directory structure for a dataset type.

        Args:
            root:         Root directory to validate.
            dataset_name: Expected dataset type.

        Returns:
            List of error strings. Empty = valid.
        """
        errors: list[str] = []

        if not root.exists():
            return [f"Root directory does not exist: '{root}'"]

        if not root.is_dir():
            return [f"Root path is not a directory: '{root}'"]

        required = _REQUIRED_STRUCTURE.get(dataset_name, [])
        for required_dir in required:
            full_path = root / required_dir
            if not full_path.exists():
                errors.append(f"Required directory missing: '{required_dir}'")
            elif not full_path.is_dir():
                errors.append(f"Path exists but is not a directory: '{required_dir}'")

        # Dataset-specific checks
        if dataset_name == DatasetName.DFDC:
            part_dirs = [d for d in root.iterdir() if d.is_dir() and "part" in d.name.lower()]
            if not part_dirs:
                errors.append(
                    "No part directories found. Expected 'dfdc_train_part_*' subdirectories."
                )

        if dataset_name == DatasetName.CUSTOM:
            if not any(root.iterdir()):
                errors.append("Custom dataset root is empty.")

        if errors:
            logger.warning(
                "[Validator] %s structure errors: %s", dataset_name, errors
            )
        else:
            logger.info("[Validator] Structure OK for %s.", dataset_name)

        return errors

    def validate_integrity(
        self,
        samples: list[SampleEntity],
        checksum_file: Path | None = None,
    ) -> list[str]:
        """Verify file integrity for all samples.

        Checks: existence, minimum size, extension, optional checksum.

        Args:
            samples:       List of samples to validate.
            checksum_file: Path to a JSON file mapping filename → sha256 hash.

        Returns:
            List of error strings. Empty = all files valid.
        """
        errors: list[str] = []
        checksums: dict[str, str] = {}

        if checksum_file and checksum_file.exists():
            with checksum_file.open("r", encoding="utf-8") as f:
                checksums = json.load(f)

        for sample in samples:
            path = sample.path

            # Existence check
            if not path.exists():
                errors.append(f"MISSING: '{path}'")
                continue

            # Extension check
            if path.suffix.lower() not in self._allowed_extensions:
                errors.append(f"INVALID_EXT: '{path.name}' ({path.suffix})")
                continue

            # Size check
            size = path.stat().st_size
            if size < self._min_size:
                errors.append(
                    f"TOO_SMALL: '{path.name}' ({size} bytes < {self._min_size})"
                )
                continue

            # Readability check (optional, slow)
            if self._check_readable:
                err = self._check_file_readable(path)
                if err:
                    errors.append(err)
                    continue

            # Checksum verification (if manifest provided)
            if checksums:
                expected = checksums.get(path.name)
                if expected:
                    actual = self._sha256(path)
                    if actual != expected:
                        errors.append(
                            f"CHECKSUM_FAIL: '{path.name}' "
                            f"expected={expected[:8]}... got={actual[:8]}..."
                        )

        if errors:
            logger.warning("[Validator] %d integrity errors.", len(errors))
        else:
            logger.info("[Validator] All %d files passed integrity check.", len(samples))

        return errors

    def generate_report(
        self,
        root: Path,
        dataset_name: DatasetName,
        samples: list[SampleEntity],
    ) -> dict[str, Any]:
        """Generate a comprehensive validation report.

        Args:
            root:         Dataset root directory.
            dataset_name: Dataset identifier.
            samples:      All loaded dataset samples.

        Returns:
            Structured report dictionary with structure and integrity sections.
        """
        structure_errors = self.validate_structure(root, dataset_name)
        integrity_errors = self.validate_integrity(samples)

        n_real = sum(1 for s in samples if s.label == Label.REAL)
        n_fake = sum(1 for s in samples if s.label == Label.FAKE)
        total = len(samples)
        balance = round(n_real / max(total, 1), 4)

        balance_warnings: list[str] = []
        if total > 0 and not (0.3 <= balance <= 0.7):
            balance_warnings.append(
                f"Class imbalance detected: real={n_real} fake={n_fake} "
                f"(balance={balance:.2%}). Consider oversampling or class weights."
            )

        missing_files = sum(1 for s in samples if not s.path.exists())

        report: dict[str, Any] = {
            "dataset_name": str(dataset_name),
            "root": str(root),
            "total_samples": total,
            "real_count": n_real,
            "fake_count": n_fake,
            "class_balance": balance,
            "missing_files": missing_files,
            "structure": {
                "valid": len(structure_errors) == 0,
                "error_count": len(structure_errors),
                "errors": structure_errors,
            },
            "integrity": {
                "valid": len(integrity_errors) == 0,
                "error_count": len(integrity_errors),
                "errors": integrity_errors[:20],  # Cap at 20 for readability
                "total_errors": len(integrity_errors),
            },
            "class_balance": {
                "valid": len(balance_warnings) == 0,
                "balance_ratio": balance,
                "warnings": balance_warnings,
            },
            "overall_valid": (
                len(structure_errors) == 0
                and len(integrity_errors) == 0
                and missing_files == 0
            ),
        }

        logger.info(
            "[Validator] Report complete | valid=%s structure_errors=%d integrity_errors=%d",
            report["overall_valid"],
            len(structure_errors),
            len(integrity_errors),
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute SHA-256 checksum of a file.

        Args:
            path: File to hash.

        Returns:
            Lowercase hex string of the SHA-256 digest.
        """
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _check_file_readable(path: Path) -> str | None:
        """Attempt to open and read the first 4 bytes of a file.

        Args:
            path: File to check.

        Returns:
            Error string if unreadable, None otherwise.
        """
        try:
            with path.open("rb") as f:
                header = f.read(4)
            if not header:
                return f"EMPTY_FILE: '{path.name}'"
        except OSError as exc:
            return f"UNREADABLE: '{path.name}' ({exc})"
        return None
