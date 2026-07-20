"""
DeepGuard — datasets/loaders/dfdc_loader.py

PyTorch Dataset loader for the Deepfake Detection Challenge (DFDC) dataset.

Dataset structure expected:
    <root>/
    ├── dfdc_train_part_0/
    │   ├── metadata.json    ← {filename: {label: "REAL"|"FAKE"}}
    │   ├── aaqaifqrwn.mp4
    │   └── ...
    ├── dfdc_train_part_1/
    │   └── ...
    └── ...  (up to part_49)

Reference: Dolhansky et al., "The Deepfake Detection Challenge Dataset",
arXiv 2006.07397.
"""

from __future__ import annotations

import json
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
)
from core.interfaces.dataset_interface import IDatasetLoader
from datasets.loaders.base_loader import BaseDeepfakeDataset

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
_METADATA_FILENAME = "metadata.json"


class DFDCDataset(BaseDeepfakeDataset):
    """PyTorch Dataset for the DFDC dataset.

    Reads metadata.json files across all part directories to discover
    samples and their labels. Supports filtering by part number range
    for memory-efficient loading of the large dataset.

    Args:
        root:             DFDC root directory containing part folders.
        split:            'train' | 'val' | 'test'.
        part_range:       Optional (start, end) part numbers to load.
                          None loads all available parts.
        val_fraction:     Fraction of training parts used as validation.
        test_fraction:    Fraction of parts used as test.
        transform:        Albumentations Compose or None.
        face_extractor:   IFaceExtractor or None.
        use_cache:        Disk caching for preprocessed frames.
        cache_dir:        Cache directory.
        image_size:       Target output size.
        max_samples:      Maximum samples to load (None = all).
        seed:             Seed for deterministic part-based split.
    """

    def __init__(
        self,
        root: Path | str,
        split: SplitName = SplitName.TRAIN,
        part_range: tuple[int, int] | None = None,
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
        self._part_range = part_range
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
        return DatasetName.DFDC

    def _discover_samples(self) -> list[SampleEntity]:
        """Discover all DFDC samples by reading metadata.json files.

        Partitions at the part-directory level for clean splits:
        no video from the same part appears in both train and test.

        Returns:
            List of SampleEntity for the requested split.

        Raises:
            DatasetStructureError: If no part directories are found.
        """
        all_parts = self._find_part_directories()

        if not all_parts:
            raise DatasetStructureError(
                DatasetName.DFDC,
                ["No 'dfdc_train_part_*' directories found in root."],
            )

        split_parts = self._partition_parts(all_parts)
        logger.debug(
            "[DFDC] split=%s assigned_parts=%d/%d",
            self._split,
            len(split_parts),
            len(all_parts),
        )

        samples: list[SampleEntity] = []
        for part_dir in split_parts:
            samples.extend(self._load_part(part_dir))

        return samples

    def _find_part_directories(self) -> list[Path]:
        """Discover all DFDC part directories in root.

        Returns:
            Sorted list of existing part directory paths.
        """
        parts: list[Path] = []
        for item in sorted(self._root.iterdir()):
            if not item.is_dir():
                continue
            name = item.name.lower()
            if "part" in name or "dfdc" in name:
                if self._part_range is None:
                    parts.append(item)
                else:
                    # Extract part number from directory name
                    try:
                        num_str = "".join(filter(str.isdigit, item.name.split("_")[-1]))
                        part_num = int(num_str)
                        if self._part_range[0] <= part_num <= self._part_range[1]:
                            parts.append(item)
                    except (ValueError, IndexError):
                        parts.append(item)  # Include if we can't parse the number

        return parts

    def _partition_parts(self, all_parts: list[Path]) -> list[Path]:
        """Assign part directories to train/val/test splits.

        Uses deterministic index-based assignment so the same seed
        always produces the same split.

        Args:
            all_parts: All available part directories sorted by name.

        Returns:
            List of part directories for the requested split.
        """
        import random

        rng = random.Random(self._seed)
        shuffled = list(all_parts)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_test = max(1, int(n * self._test_fraction))
        n_val = max(1, int(n * self._val_fraction))
        n_train = n - n_val - n_test

        if self._split == SplitName.TEST:
            return shuffled[:n_test]
        if self._split == SplitName.VAL:
            return shuffled[n_test : n_test + n_val]
        # TRAIN
        return shuffled[n_test + n_val :]

    def _load_part(self, part_dir: Path) -> list[SampleEntity]:
        """Load all samples from a single DFDC part directory.

        Args:
            part_dir: Path to a part directory containing metadata.json.

        Returns:
            List of SampleEntity from this part.
        """
        metadata_path = part_dir / _METADATA_FILENAME
        if not metadata_path.exists():
            logger.warning("[DFDC] metadata.json not found in '%s'; skipping.", part_dir)
            return []

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata: dict[str, dict[str, Any]] = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("[DFDC] Failed to read metadata in '%s': %s", part_dir, exc)
            return []

        samples: list[SampleEntity] = []
        for filename, info in metadata.items():
            video_path = part_dir / filename
            if not video_path.suffix.lower() in _VIDEO_EXTENSIONS:
                continue

            label_str = str(info.get("label", "FAKE")).upper()
            label = Label.REAL if label_str == "REAL" else Label.FAKE
            original = info.get("original", "")

            samples.append(
                SampleEntity.create(
                    path=video_path,
                    label=label,
                    dataset_name=DatasetName.DFDC,
                    media_type=MediaType.VIDEO,
                    manipulation=(
                        ManipulationType.NONE
                        if label == Label.REAL
                        else ManipulationType.UNKNOWN
                    ),
                    video_id=video_path.stem,
                    subject_id=original or video_path.stem,
                    metadata={"part": part_dir.name, "original": original},
                )
            )

        logger.debug(
            "[DFDC] Part '%s': %d samples.", part_dir.name, len(samples)
        )
        return samples


class DFDCLoader(IDatasetLoader):
    """IDatasetLoader adapter for the DFDC dataset.

    Args:
        root: DFDC root directory path.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise DatasetNotFoundError(self._root, DatasetName.DFDC)

    @property
    def dataset_name(self) -> DatasetName:
        """Return dataset identifier."""
        return DatasetName.DFDC

    @property
    def root_path(self) -> Path:
        """Return the root directory path."""
        return self._root

    def load(self) -> list[SampleEntity]:
        """Load all samples from all splits.

        Returns:
            Combined SampleEntity list.
        """
        all_samples: list[SampleEntity] = []
        for split in SplitName:
            try:
                ds = DFDCDataset(root=self._root, split=split)
                all_samples.extend(ds.samples)
            except Exception as exc:
                logger.warning("[DFDC] Could not load split '%s': %s", split, exc)
        return all_samples

    def get_sample_count(self) -> dict[str, int]:
        """Return approximate counts by scanning metadata files.

        Returns:
            Dictionary with 'real', 'fake', 'total' keys.
        """
        real, fake = 0, 0
        for meta_file in self._root.rglob(_METADATA_FILENAME):
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    data: dict[str, dict[str, Any]] = json.load(f)
                for info in data.values():
                    lbl = str(info.get("label", "FAKE")).upper()
                    if lbl == "REAL":
                        real += 1
                    else:
                        fake += 1
            except Exception:
                continue
        return {"real": real, "fake": fake, "total": real + fake}

    def get_metadata(self) -> DatasetMetadataEntity:
        """Return dataset-level metadata.

        Returns:
            DatasetMetadataEntity instance.
        """
        counts = self.get_sample_count()
        return DatasetMetadataEntity(
            dataset_id=str(uuid.uuid4()),
            name=DatasetName.DFDC,
            version="1.0",
            root_path=self._root,
            total_samples=counts["total"],
            real_count=counts["real"],
            fake_count=counts["fake"],
            created_at=datetime.now(tz=timezone.utc),
            description="DFDC: Deepfake Detection Challenge dataset (~100k videos).",
        )
