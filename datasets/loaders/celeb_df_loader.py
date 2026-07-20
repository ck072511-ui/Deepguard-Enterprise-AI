"""
DeepGuard — datasets/loaders/celeb_df_loader.py

PyTorch Dataset loader for the CelebDF-v2 deepfake detection dataset.

Dataset structure expected:
    <root>/
    ├── Celeb-real/          # Real celebrity videos
    ├── YouTube-real/        # Real YouTube videos
    ├── Celeb-synthesis/     # Fake (GAN-synthesized) videos
    └── List_of_testing_videos.txt  # Official test split list

Reference: Yuezun Li et al., "Celeb-DF: A Large-scale Challenging Dataset
for DeepFake Forensics", CVPR 2020.
"""

from __future__ import annotations

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

# Official directory names in CelebDF-v2
_REAL_DIRS = ["Celeb-real", "YouTube-real"]
_FAKE_DIRS = ["Celeb-synthesis"]
_TEST_LIST_FILE = "List_of_testing_videos.txt"

# Supported video extensions
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class CelebDFDataset(BaseDeepfakeDataset):
    """PyTorch Dataset for CelebDF-v2.

    Loads video-level samples (or frame-level if frames are pre-extracted).
    Respects the official test split list when split='test'.

    Args:
        root:             Path to CelebDF-v2 root directory.
        split:            'train' | 'val' | 'test'.
        transform:        Albumentations Compose or None.
        face_extractor:   IFaceExtractor or None.
        use_cache:        Enable disk caching of preprocessed frames.
        cache_dir:        Cache directory path.
        image_size:       Target square output size.
        max_samples:      Limit sample count (None = all).
        train_val_ratio:  Fraction of non-test data used for training.
        seed:             Random seed for train/val split.
    """

    def __init__(
        self,
        root: Path | str,
        split: SplitName = SplitName.TRAIN,
        transform: Any | None = None,
        face_extractor: Any | None = None,
        *,
        use_cache: bool = False,
        cache_dir: Path | str | None = None,
        image_size: int = 224,
        max_samples: int | None = None,
        train_val_ratio: float = 0.9,
        seed: int = 42,
    ) -> None:
        self._train_val_ratio = train_val_ratio
        self._seed = seed
        self._test_video_ids: set[str] = set()

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

    # ------------------------------------------------------------------
    # BaseDeepfakeDataset contract
    # ------------------------------------------------------------------

    @property
    def dataset_name(self) -> str:
        """Return dataset identifier."""
        return DatasetName.CELEB_DF

    def _discover_samples(self) -> list[SampleEntity]:
        """Discover all video samples from CelebDF-v2 directory structure.

        Loads the official test list to partition samples correctly.
        Non-test samples are split into train/val by ratio.

        Returns:
            Ordered list of SampleEntity for the requested split.

        Raises:
            DatasetStructureError: If required directories are missing.
        """
        self._validate_structure()
        self._load_test_list()

        all_real = self._collect_videos(_REAL_DIRS, Label.REAL)
        all_fake = self._collect_videos(_FAKE_DIRS, Label.FAKE)
        all_samples = all_real + all_fake

        if self._split == SplitName.TEST:
            selected = [s for s in all_samples if s.video_id in self._test_video_ids]
        else:
            non_test = [s for s in all_samples if s.video_id not in self._test_video_ids]
            selected = self._split_train_val(non_test)

        logger.debug(
            "[CelebDF] split=%s selected=%d / total=%d",
            self._split,
            len(selected),
            len(all_samples),
        )
        return selected

    # ------------------------------------------------------------------
    # IDatasetLoader convenience methods (used by CelebDFLoader adapter)
    # ------------------------------------------------------------------

    def _validate_structure(self) -> None:
        """Validate that all required CelebDF directories exist.

        Raises:
            DatasetStructureError: If any required directory is missing.
        """
        missing: list[str] = []
        for dir_name in _REAL_DIRS + _FAKE_DIRS:
            if not (self._root / dir_name).exists():
                missing.append(dir_name)
        if missing:
            raise DatasetStructureError(DatasetName.CELEB_DF, missing)

    def _load_test_list(self) -> None:
        """Parse the official test list file into a set of video IDs.

        Silently skips if the file does not exist (test split not available).
        """
        test_list_path = self._root / _TEST_LIST_FILE
        if not test_list_path.exists():
            logger.warning(
                "[CelebDF] '%s' not found; test split will be empty.", _TEST_LIST_FILE
            )
            return

        with test_list_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Format: "1 Celeb-real/id0_0002.mp4"
                    parts = line.split()
                    if parts:
                        video_path = parts[-1]
                        video_id = Path(video_path).stem
                        self._test_video_ids.add(video_id)

        logger.debug("[CelebDF] Loaded %d test video IDs.", len(self._test_video_ids))

    def _collect_videos(
        self,
        dir_names: list[str],
        label: Label,
    ) -> list[SampleEntity]:
        """Collect all video samples from a list of directories.

        Args:
            dir_names: List of subdirectory names to scan.
            label:     Ground-truth label to assign.

        Returns:
            List of SampleEntity objects.
        """
        samples: list[SampleEntity] = []
        manipulation = ManipulationType.NONE if label == Label.REAL else ManipulationType.UNKNOWN

        for dir_name in dir_names:
            directory = self._root / dir_name
            if not directory.exists():
                logger.warning("[CelebDF] Directory not found: '%s'", directory)
                continue

            for video_file in sorted(directory.iterdir()):
                if video_file.suffix.lower() not in _VIDEO_EXTENSIONS:
                    continue
                samples.append(
                    SampleEntity.create(
                        path=video_file,
                        label=label,
                        dataset_name=DatasetName.CELEB_DF,
                        media_type=MediaType.VIDEO,
                        manipulation=manipulation,
                        subject_id=video_file.stem.split("_")[0],
                        video_id=video_file.stem,
                    )
                )

        return samples

    def _split_train_val(self, samples: list[SampleEntity]) -> list[SampleEntity]:
        """Split non-test samples into train or val partition.

        Uses a deterministic hash-based split to avoid data leakage.

        Args:
            samples: All non-test samples.

        Returns:
            Subset of samples for the current split (train or val).
        """
        import hashlib

        threshold = int(self._train_val_ratio * 1000)
        result: list[SampleEntity] = []

        for sample in samples:
            h = int(hashlib.md5(sample.video_id.encode()).hexdigest(), 16)
            bucket = h % 1000
            is_train = bucket < threshold

            if self._split == SplitName.TRAIN and is_train:
                result.append(sample)
            elif self._split == SplitName.VAL and not is_train:
                result.append(sample)

        return result


class CelebDFLoader(IDatasetLoader):
    """IDatasetLoader adapter for CelebDF-v2.

    Provides the standard loader interface (load, get_sample_count,
    get_metadata) on top of CelebDFDataset.

    Args:
        root: CelebDF-v2 root directory path.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise DatasetNotFoundError(self._root, DatasetName.CELEB_DF)

    @property
    def dataset_name(self) -> DatasetName:
        """Return the dataset identifier."""
        return DatasetName.CELEB_DF

    @property
    def root_path(self) -> Path:
        """Return the root directory path."""
        return self._root

    def load(self) -> list[SampleEntity]:
        """Discover and return all samples from all splits combined.

        Returns:
            Complete list of SampleEntity objects.
        """
        all_samples: list[SampleEntity] = []
        for split in SplitName:
            try:
                ds = CelebDFDataset(root=self._root, split=split)
                all_samples.extend(ds.samples)
            except Exception as exc:
                logger.warning("[CelebDF] Could not load split '%s': %s", split, exc)
        return all_samples

    def get_sample_count(self) -> dict[str, int]:
        """Return sample counts without loading all samples.

        Returns:
            Dictionary with 'real', 'fake', 'total' keys.
        """
        real = sum(
            1
            for d in _REAL_DIRS
            for f in (self._root / d).glob("*")
            if f.suffix.lower() in _VIDEO_EXTENSIONS
            if (self._root / d).exists()
        )
        fake = sum(
            1
            for d in _FAKE_DIRS
            for f in (self._root / d).glob("*")
            if f.suffix.lower() in _VIDEO_EXTENSIONS
            if (self._root / d).exists()
        )
        return {"real": real, "fake": fake, "total": real + fake}

    def get_metadata(self) -> DatasetMetadataEntity:
        """Return dataset-level metadata.

        Returns:
            DatasetMetadataEntity instance.
        """
        counts = self.get_sample_count()
        return DatasetMetadataEntity(
            dataset_id=str(uuid.uuid4()),
            name=DatasetName.CELEB_DF,
            version="2.0",
            root_path=self._root,
            total_samples=counts["total"],
            real_count=counts["real"],
            fake_count=counts["fake"],
            created_at=datetime.now(tz=timezone.utc),
            description="CelebDF-v2: Celebrity deepfake video dataset.",
        )
