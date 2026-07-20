"""
DeepGuard — datasets/loaders/custom_loader.py

PyTorch Dataset loader for user-provided custom datasets.

Expected directory structure (two supported layouts):

Layout A — simple real/fake directories (images or videos):
    <root>/
    ├── real/
    │   ├── img001.jpg
    │   └── video001.mp4
    └── fake/
        ├── img001.jpg
        └── video001.mp4

Layout B — class-named subdirectories (sklearn-style):
    <root>/
    ├── 0/   or  real/
    └── 1/   or  fake/

Layout C — flat directory with a CSV manifest:
    <root>/
    ├── manifest.csv   ← columns: path,label  (label: 0/1 or real/fake)
    └── ...media files...

The loader auto-detects the layout on construction.
"""

from __future__ import annotations

import csv
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.domain.entities.dataset_entity import (
    DatasetMetadataEntity,
    DatasetName,
    Label,
    ManipulationType,
    MediaType,
    SampleEntity,
    SplitName,
)
from core.exceptions.dataset_exceptions import (
    DatasetNotFoundError,
    DatasetStructureError,
    DatasetValidationError,
)
from core.interfaces.dataset_interface import IDatasetLoader
from datasets.loaders.base_loader import BaseDeepfakeDataset

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_ALL_EXTENSIONS = _IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS

# Canonical directory name aliases for real/fake classes
_REAL_ALIASES = {"real", "0", "original", "genuine"}
_FAKE_ALIASES = {"fake", "1", "manipulated", "deepfake", "synthetic"}


class CustomDataset(BaseDeepfakeDataset):
    """PyTorch Dataset for any user-organised custom dataset.

    Automatically detects the directory layout (A, B, or C) and loads
    all matching media files with their corresponding labels.

    Args:
        root:           Dataset root directory.
        split:          'train' | 'val' | 'test'.
        real_dir:       Name of the real-samples subdirectory (Layout A/B).
        fake_dir:       Name of the fake-samples subdirectory (Layout A/B).
        manifest_file:  CSV manifest filename (Layout C). None = auto-detect.
        val_fraction:   Fraction of data for validation split.
        test_fraction:  Fraction of data for test split.
        transform:      Albumentations Compose or None.
        face_extractor: IFaceExtractor or None.
        use_cache:      Enable disk caching.
        cache_dir:      Cache directory.
        image_size:     Target square output size.
        max_samples:    Maximum samples (None = all).
        seed:           Random seed for reproducible splits.
    """

    def __init__(
        self,
        root: Path | str,
        split: SplitName = SplitName.TRAIN,
        real_dir: str = "real",
        fake_dir: str = "fake",
        manifest_file: str | None = None,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        transform: Any | None = None,
        face_extractor: Any | None = None,
        *,
        use_cache: bool = False,
        cache_dir: Path | str | None = None,
        image_size: int = 224,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self._real_dir = real_dir
        self._fake_dir = fake_dir
        self._manifest_file = manifest_file
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction
        self._seed = seed

        super().__init__(
            root=root,
            split=split,
            transform=transform,
            face_extractor=face_extractor,
            use_cache=use_cache,
            cache_dir=cache_dir,
            image_size=image_size,
            max_samples=max_samples,
        )

    @property
    def dataset_name(self) -> str:
        """Return dataset identifier."""
        return DatasetName.CUSTOM

    def _discover_samples(self) -> list[SampleEntity]:
        """Auto-detect layout and load samples for the requested split.

        Returns:
            List of SampleEntity for this split.

        Raises:
            DatasetStructureError: If no usable layout is detected.
        """
        layout = self._detect_layout()
        logger.debug("[Custom] Detected layout: '%s'", layout)

        if layout == "manifest":
            all_samples = self._load_from_manifest()
        elif layout in ("A", "B"):
            all_samples = self._load_from_directories()
        else:
            raise DatasetStructureError(
                DatasetName.CUSTOM,
                [
                    "Could not detect a valid dataset layout. "
                    "Expected: real/ + fake/ dirs, or manifest.csv."
                ],
            )

        return self._split_samples(all_samples)

    def _detect_layout(self) -> str:
        """Determine which directory layout the dataset uses.

        Returns:
            'manifest' | 'A' | 'B' | 'unknown'.
        """
        # Layout C: CSV manifest
        manifest_name = self._manifest_file or "manifest.csv"
        if (self._root / manifest_name).exists():
            return "manifest"

        # Layout A/B: real and fake directories
        real_path = self._find_class_dir(_REAL_ALIASES, self._real_dir)
        fake_path = self._find_class_dir(_FAKE_ALIASES, self._fake_dir)
        if real_path or fake_path:
            return "A"

        return "unknown"

    def _find_class_dir(self, aliases: set[str], preferred: str) -> Path | None:
        """Find a class directory by checking preferred name then aliases.

        Args:
            aliases:   Set of acceptable directory name aliases.
            preferred: Preferred directory name to check first.

        Returns:
            Found Path or None.
        """
        preferred_path = self._root / preferred
        if preferred_path.exists():
            return preferred_path
        for item in self._root.iterdir():
            if item.is_dir() and item.name.lower() in aliases:
                return item
        return None

    def _load_from_directories(self) -> list[SampleEntity]:
        """Load all samples from real/ and fake/ subdirectories.

        Returns:
            Combined list of real and fake SampleEntity objects.
        """
        samples: list[SampleEntity] = []

        for label, aliases, preferred in [
            (Label.REAL, _REAL_ALIASES, self._real_dir),
            (Label.FAKE, _FAKE_ALIASES, self._fake_dir),
        ]:
            class_dir = self._find_class_dir(aliases, preferred)
            if class_dir is None:
                logger.warning("[Custom] No directory found for label '%s'.", label.name)
                continue

            count = 0
            for media_file in sorted(class_dir.rglob("*")):
                if media_file.is_dir():
                    continue
                if media_file.suffix.lower() not in _ALL_EXTENSIONS:
                    continue

                media_type = (
                    MediaType.IMAGE
                    if media_file.suffix.lower() in _IMAGE_EXTENSIONS
                    else MediaType.VIDEO
                )
                manipulation = (
                    ManipulationType.NONE if label == Label.REAL else ManipulationType.UNKNOWN
                )

                samples.append(
                    SampleEntity.create(
                        path=media_file,
                        label=label,
                        dataset_name=DatasetName.CUSTOM,
                        media_type=media_type,
                        manipulation=manipulation,
                        video_id=media_file.stem,
                        subject_id=media_file.parent.name,
                    )
                )
                count += 1

            logger.debug("[Custom] Loaded %d %s samples from '%s'.", count, label.name, class_dir)

        return samples

    def _load_from_manifest(self) -> list[SampleEntity]:
        """Load samples from a CSV manifest file.

        CSV format (with or without header):
            path,label
            relative/path/to/file.jpg,0
            relative/path/to/file.jpg,fake

        Args: None (uses self._root and self._manifest_file).

        Returns:
            List of SampleEntity parsed from the manifest.

        Raises:
            DatasetValidationError: If the manifest cannot be parsed.
        """
        manifest_name = self._manifest_file or "manifest.csv"
        manifest_path = self._root / manifest_name

        samples: list[SampleEntity] = []
        errors: list[str] = []

        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise DatasetValidationError(
                    f"Empty manifest file: '{manifest_path}'",
                    dataset_name=DatasetName.CUSTOM,
                )
            if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise DatasetValidationError(
                    f"Manifest must have 'path' and 'label' columns. "
                    f"Found: {list(reader.fieldnames)}",
                    dataset_name=DatasetName.CUSTOM,
                )

            for i, row in enumerate(reader):
                raw_path = row.get("path", "").strip()
                raw_label = row.get("label", "").strip()

                if not raw_path or not raw_label:
                    errors.append(f"Row {i + 2}: missing path or label.")
                    continue

                full_path = (self._root / raw_path).resolve()

                try:
                    label = Label.from_string(raw_label)
                except ValueError as exc:
                    errors.append(f"Row {i + 2}: {exc}")
                    continue

                media_type = (
                    MediaType.IMAGE
                    if full_path.suffix.lower() in _IMAGE_EXTENSIONS
                    else MediaType.VIDEO
                )
                samples.append(
                    SampleEntity.create(
                        path=full_path,
                        label=label,
                        dataset_name=DatasetName.CUSTOM,
                        media_type=media_type,
                        manipulation=(
                            ManipulationType.NONE
                            if label == Label.REAL
                            else ManipulationType.UNKNOWN
                        ),
                        video_id=full_path.stem,
                    )
                )

        if errors:
            logger.warning(
                "[Custom] %d manifest parsing errors:\n  %s",
                len(errors),
                "\n  ".join(errors[:5]),
            )

        logger.debug("[Custom] Loaded %d samples from manifest.", len(samples))
        return samples

    def _split_samples(self, samples: list[SampleEntity]) -> list[SampleEntity]:
        """Deterministically split samples by hash into train/val/test.

        Args:
            samples: All discovered samples.

        Returns:
            Subset for the current split.
        """
        import hashlib

        n_test_thresh = int(self._test_fraction * 1000)
        n_val_thresh = int((self._test_fraction + self._val_fraction) * 1000)

        result: list[SampleEntity] = []
        for sample in samples:
            h = int(hashlib.md5(str(sample.path).encode()).hexdigest(), 16)
            bucket = h % 1000

            if bucket < n_test_thresh:
                if self._split == SplitName.TEST:
                    result.append(sample)
            elif bucket < n_val_thresh:
                if self._split == SplitName.VAL:
                    result.append(sample)
            else:
                if self._split == SplitName.TRAIN:
                    result.append(sample)

        return result


class CustomLoader(IDatasetLoader):
    """IDatasetLoader adapter for Custom datasets.

    Args:
        root:       Custom dataset root directory.
        real_dir:   Name of the real-samples directory.
        fake_dir:   Name of the fake-samples directory.
    """

    def __init__(
        self,
        root: Path | str,
        real_dir: str = "real",
        fake_dir: str = "fake",
    ) -> None:
        self._root = Path(root)
        self._real_dir = real_dir
        self._fake_dir = fake_dir
        if not self._root.exists():
            raise DatasetNotFoundError(self._root, DatasetName.CUSTOM)

    @property
    def dataset_name(self) -> DatasetName:
        """Return dataset identifier."""
        return DatasetName.CUSTOM

    @property
    def root_path(self) -> Path:
        """Return the root directory."""
        return self._root

    def load(self) -> list[SampleEntity]:
        """Load all samples from all splits combined.

        Returns:
            Complete list of SampleEntity.
        """
        all_samples: list[SampleEntity] = []
        for split in SplitName:
            try:
                ds = CustomDataset(
                    root=self._root,
                    split=split,
                    real_dir=self._real_dir,
                    fake_dir=self._fake_dir,
                )
                all_samples.extend(ds.samples)
            except Exception as exc:
                logger.warning("[Custom] Could not load split '%s': %s", split, exc)
        return all_samples

    def get_sample_count(self) -> dict[str, int]:
        """Count real and fake samples by directory scan.

        Returns:
            Dictionary with 'real', 'fake', 'total' keys.
        """

        def _count_dir(path: Path) -> int:
            if not path.exists():
                return 0
            return sum(1 for f in path.rglob("*") if f.suffix.lower() in _ALL_EXTENSIONS)

        real = _count_dir(self._root / self._real_dir)
        fake = _count_dir(self._root / self._fake_dir)
        return {"real": real, "fake": fake, "total": real + fake}

    def get_metadata(self) -> DatasetMetadataEntity:
        """Return dataset-level metadata.

        Returns:
            DatasetMetadataEntity instance.
        """
        counts = self.get_sample_count()
        return DatasetMetadataEntity(
            dataset_id=str(uuid.uuid4()),
            name=DatasetName.CUSTOM,
            version="1.0",
            root_path=self._root,
            total_samples=counts["total"],
            real_count=counts["real"],
            fake_count=counts["fake"],
            created_at=datetime.now(tz=timezone.utc),
            description="User-provided custom deepfake dataset.",
        )
