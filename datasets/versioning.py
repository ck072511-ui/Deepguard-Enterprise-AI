"""
DeepGuard — datasets/versioning.py

Dataset version tracking implementing IDataVersioner.

Creates JSON manifest files that record:
  - Complete sample inventory (paths + labels + metadata)
  - Dataset-level metadata (name, version, created_at)
  - SHA-256 checksums for each file
  - Git commit hash (if in a repo)

Manifests enable exact dataset reproduction and audit trails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.domain.entities.dataset_entity import (
    CompressionLevel,
    DatasetMetadataEntity,
    DatasetName,
    Label,
    ManipulationType,
    MediaType,
    SampleEntity,
    SplitName,
)
from core.exceptions.dataset_exceptions import DatasetVersionError
from core.interfaces.dataset_interface import IDataVersioner

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = "1.0"


class DatasetVersioner(IDataVersioner):
    """SHA-256-based dataset version tracker with JSON manifests.

    Manifests are named: ``{dataset_name}_{timestamp}.manifest.json``

    Args:
        compute_file_checksums: If True, compute SHA-256 for every file
                                (slow but thorough). Defaults to False.
    """

    def __init__(self, compute_file_checksums: bool = False) -> None:
        self._compute_checksums = compute_file_checksums

    # ------------------------------------------------------------------
    # IDataVersioner interface
    # ------------------------------------------------------------------

    def save_version(
        self,
        samples: list[SampleEntity],
        metadata: DatasetMetadataEntity,
        output_dir: Path,
    ) -> Path:
        """Persist a versioned manifest of the current dataset state.

        Args:
            samples:    All dataset samples to record.
            metadata:   Dataset-level metadata entity.
            output_dir: Directory where the manifest is saved.

        Returns:
            Path to the saved JSON manifest file.

        Raises:
            DatasetVersionError: If the manifest cannot be written.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        uid_suffix = str(uuid.uuid4())[:8]
        manifest_name = f"{metadata.name}_{timestamp}_{uid_suffix}.manifest.json"
        manifest_path = output_dir / manifest_name

        manifest = self._build_manifest(samples, metadata)
        manifest_checksum = self._checksum_dict(manifest)
        manifest["manifest_checksum"] = manifest_checksum

        try:
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str, ensure_ascii=False)
        except OSError as exc:
            raise DatasetVersionError(
                f"Failed to write manifest: {exc}",
                version=manifest.get("version", ""),
            ) from exc

        logger.info(
            "[Versioner] Manifest saved | path=%s samples=%d checksum=%s...",
            manifest_path.name,
            len(samples),
            manifest_checksum[:8],
        )
        return manifest_path

    def load_version(
        self,
        manifest_path: Path,
    ) -> tuple[list[SampleEntity], DatasetMetadataEntity]:
        """Load a previously versioned dataset from its manifest.

        Args:
            manifest_path: Path to the JSON manifest file.

        Returns:
            Tuple of (list of SampleEntity, DatasetMetadataEntity).

        Raises:
            DatasetVersionError: If the manifest is invalid or checksum fails.
        """
        if not manifest_path.exists():
            raise DatasetVersionError(
                f"Manifest file not found: '{manifest_path}'",
                version=manifest_path.stem,
            )

        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest: dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise DatasetVersionError(
                f"Failed to read manifest '{manifest_path}': {exc}",
                version=manifest_path.stem,
            ) from exc

        # Verify checksum
        stored_checksum = manifest.pop("manifest_checksum", None)
        if stored_checksum:
            actual_checksum = self._checksum_dict(manifest)
            if actual_checksum != stored_checksum:
                raise DatasetVersionError(
                    f"Manifest checksum mismatch for '{manifest_path.name}'. "
                    "The file may be corrupted.",
                    version=manifest_path.stem,
                )

        samples = self._deserialise_samples(manifest.get("samples", []))
        meta = self._deserialise_metadata(manifest.get("metadata", {}))

        logger.info(
            "[Versioner] Manifest loaded | samples=%d dataset=%s",
            len(samples),
            meta.name,
        )
        return samples, meta

    def list_versions(self, output_dir: Path) -> list[dict[str, Any]]:
        """List all manifest files in a directory, newest first.

        Args:
            output_dir: Directory to scan for manifest files.

        Returns:
            List of summary dicts with keys: path, name, created_at,
            total_samples, dataset_name.
        """
        if not output_dir.exists():
            return []

        manifests: list[dict[str, Any]] = []
        for path in sorted(output_dir.glob("*.manifest.json"), reverse=True):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)
                meta = data.get("metadata", {})
                manifests.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "created_at": meta.get("created_at", ""),
                        "total_samples": data.get("total_samples", 0),
                        "dataset_name": meta.get("name", "unknown"),
                        "version": meta.get("version", ""),
                        "manifest_version": data.get("manifest_version", ""),
                    }
                )
            except Exception as exc:
                logger.warning("[Versioner] Could not read '%s': %s", path.name, exc)

        return manifests

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_manifest(
        self,
        samples: list[SampleEntity],
        metadata: DatasetMetadataEntity,
    ) -> dict[str, Any]:
        """Construct the full manifest dictionary.

        Args:
            samples:  All dataset samples.
            metadata: Dataset metadata entity.

        Returns:
            JSON-serialisable manifest dictionary.
        """
        serialised_samples = []
        for s in samples:
            entry = s.to_dict()
            if self._compute_checksums and s.path.exists():
                entry["sha256"] = self._sha256_file(s.path)
            serialised_samples.append(entry)

        return {
            "manifest_version": _MANIFEST_VERSION,
            "manifest_id": str(uuid.uuid4()),
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "git_commit": self._get_git_commit(),
            "total_samples": len(samples),
            "metadata": metadata.to_dict(),
            "samples": serialised_samples,
        }

    @staticmethod
    def _deserialise_samples(raw: list[dict[str, Any]]) -> list[SampleEntity]:
        """Reconstruct SampleEntity objects from manifest data.

        Args:
            raw: List of sample dictionaries from manifest.

        Returns:
            List of SampleEntity objects.
        """
        samples: list[SampleEntity] = []
        for entry in raw:
            try:
                sample = SampleEntity(
                    sample_id=entry.get("sample_id", str(uuid.uuid4())),
                    path=Path(entry["path"]),
                    label=Label(int(entry["label"])),
                    dataset_name=DatasetName(entry["dataset_name"]),
                    media_type=MediaType(entry.get("media_type", "image")),
                    manipulation=ManipulationType(
                        entry.get("manipulation", ManipulationType.UNKNOWN)
                    ),
                    compression=CompressionLevel(
                        entry.get("compression", CompressionLevel.UNKNOWN)
                    ),
                    subject_id=entry.get("subject_id", ""),
                    video_id=entry.get("video_id", ""),
                    frame_index=int(entry.get("frame_index", -1)),
                    split=SplitName(entry.get("split", SplitName.TRAIN)),
                    metadata=entry.get("metadata", {}),
                )
                samples.append(sample)
            except (KeyError, ValueError) as exc:
                logger.warning("[Versioner] Skipping corrupt sample entry: %s", exc)

        return samples

    @staticmethod
    def _deserialise_metadata(raw: dict[str, Any]) -> DatasetMetadataEntity:
        """Reconstruct DatasetMetadataEntity from manifest data.

        Args:
            raw: Metadata dictionary from manifest.

        Returns:
            DatasetMetadataEntity instance.
        """
        return DatasetMetadataEntity(
            dataset_id=raw.get("dataset_id", str(uuid.uuid4())),
            name=DatasetName(raw.get("name", DatasetName.CUSTOM)),
            version=raw.get("version", "1.0"),
            root_path=Path(raw.get("root_path", ".")),
            total_samples=int(raw.get("total_samples", 0)),
            real_count=int(raw.get("real_count", 0)),
            fake_count=int(raw.get("fake_count", 0)),
            created_at=datetime.fromisoformat(
                raw.get("created_at", datetime.now(tz=timezone.utc).isoformat())
            ),
            checksum=raw.get("checksum", ""),
            description=raw.get("description", ""),
            tags=raw.get("tags", {}),
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Compute the SHA-256 checksum of a file.

        Args:
            path: File to hash.

        Returns:
            Lowercase hex string digest.
        """
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _checksum_dict(data: dict[str, Any]) -> str:
        """Compute a SHA-256 checksum of a JSON-serialisable dictionary.

        Args:
            data: Dictionary to hash.

        Returns:
            Lowercase hex string of the SHA-256 digest.
        """
        serialised = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialised).hexdigest()

    @staticmethod
    def _get_git_commit() -> str:
        """Retrieve the current Git commit hash.

        Returns:
            Short commit hash string, or 'unknown' if not in a Git repo.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return "unknown"
