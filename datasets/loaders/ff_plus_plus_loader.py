"""
DeepGuard — datasets/loaders/ff_plus_plus_loader.py

PyTorch Dataset loader for FaceForensics++ (FF++) deepfake dataset.

Dataset structure expected:
    <root>/
    ├── original_sequences/
    │   ├── actors/raw/videos/          (or c23/videos/, c40/videos/)
    │   └── youtube/raw/videos/
    ├── manipulated_sequences/
    │   ├── Deepfakes/raw/videos/
    │   ├── Face2Face/raw/videos/
    │   ├── FaceSwap/raw/videos/
    │   ├── NeuralTextures/raw/videos/
    │   └── FaceShifter/raw/videos/
    └── splits/
        ├── train.json
        ├── val.json
        └── test.json

Reference: Rössler et al., "FaceForensics++: Learning to Detect Manipulated
Facial Images", ICCV 2019.
"""

from __future__ import annotations

import json
import logging
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
from core.exceptions.dataset_exceptions import (
    DatasetNotFoundError,
    DatasetStructureError,
)
from core.interfaces.dataset_interface import IDatasetLoader
from datasets.loaders.base_loader import BaseDeepfakeDataset

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Mapping from directory name to ManipulationType
_MANIPULATION_MAP: dict[str, ManipulationType] = {
    "Deepfakes": ManipulationType.DEEPFAKES,
    "Face2Face": ManipulationType.FACE2FACE,
    "FaceSwap": ManipulationType.FACESWAP,
    "NeuralTextures": ManipulationType.NEURAL_TEXTURES,
    "FaceShifter": ManipulationType.FACESHIFTER,
}

_COMPRESSION_MAP: dict[str, CompressionLevel] = {
    "raw": CompressionLevel.RAW,
    "c0": CompressionLevel.RAW,
    "c23": CompressionLevel.LIGHT,
    "c40": CompressionLevel.HEAVY,
}

_ALL_MANIPULATIONS = list(_MANIPULATION_MAP.keys())


class FFPlusPlusDataset(BaseDeepfakeDataset):
    """PyTorch Dataset for FaceForensics++ (FF++).

    Supports all manipulation types and compression levels.
    Respects official JSON splits when available.

    Args:
        root:                Path to FF++ root directory.
        split:               'train' | 'val' | 'test'.
        compression:         Compression level: 'c0' | 'c23' | 'c40'.
        manipulation_types:  List of manipulation method names to include.
                             None = include all.
        transform:           Albumentations Compose or None.
        face_extractor:      IFaceExtractor or None.
        use_cache:           Disk caching for preprocessed frames.
        cache_dir:           Cache directory.
        image_size:          Target square output size.
        max_samples:         Sample limit (None = all).
    """

    def __init__(
        self,
        root: Path | str,
        split: SplitName = SplitName.TRAIN,
        compression: str = "c23",
        manipulation_types: list[str] | None = None,
        transform: Any | None = None,
        face_extractor: Any | None = None,
        *,
        use_cache: bool = False,
        cache_dir: Path | str | None = None,
        image_size: int = 224,
        max_samples: int | None = None,
    ) -> None:
        self._compression = compression
        self._manipulation_types = manipulation_types or _ALL_MANIPULATIONS
        self._official_split_ids: set[str] | None = None

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
        return DatasetName.FF_PLUS_PLUS

    def _discover_samples(self) -> list[SampleEntity]:
        """Discover all samples for the requested split.

        Loads official JSON splits if available; falls back to
        a ratio-based split otherwise.

        Returns:
            List of SampleEntity for this split and compression level.

        Raises:
            DatasetStructureError: If no sequences can be found.
        """
        self._load_official_split()
        samples: list[SampleEntity] = []
        samples.extend(self._collect_real_sequences())
        samples.extend(self._collect_fake_sequences())

        if not samples:
            raise DatasetStructureError(
                DatasetName.FF_PLUS_PLUS,
                [f"No sequences found for compression='{self._compression}'"],
            )

        logger.debug(
            "[FF++] split=%s compression=%s types=%s samples=%d",
            self._split,
            self._compression,
            self._manipulation_types,
            len(samples),
        )
        return samples

    def _load_official_split(self) -> None:
        """Load the official train/val/test video ID list from JSON.

        Silently skips if splits directory does not exist.
        """
        splits_dir = self._root / "splits"
        split_file = splits_dir / f"{self._split}.json"

        if not split_file.exists():
            logger.debug("[FF++] Official split file not found: '%s'.", split_file)
            return

        with split_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Format: [["000", "001"], ["002", "003"], ...]
        ids: set[str] = set()
        for pair in data:
            for vid_id in pair:
                ids.add(str(vid_id).zfill(3))

        self._official_split_ids = ids
        logger.debug("[FF++] Loaded %d official split IDs.", len(ids))

    def _collect_real_sequences(self) -> list[SampleEntity]:
        """Collect real (original) video sequences.

        Returns:
            List of real SampleEntity objects.
        """
        samples: list[SampleEntity] = []
        for source in ["actors", "youtube"]:
            seq_dir = self._root / "original_sequences" / source / self._compression / "videos"
            if not seq_dir.exists():
                # Try alternate path without subdirectory
                seq_dir = self._root / "original_sequences" / source / "raw" / "videos"
            if not seq_dir.exists():
                logger.debug("[FF++] Real sequence dir not found: '%s'", seq_dir)
                continue

            for video in sorted(seq_dir.iterdir()):
                if video.suffix.lower() not in _VIDEO_EXTENSIONS:
                    continue
                if not self._is_in_split(video.stem):
                    continue
                samples.append(
                    SampleEntity.create(
                        path=video,
                        label=Label.REAL,
                        dataset_name=DatasetName.FF_PLUS_PLUS,
                        media_type=MediaType.VIDEO,
                        manipulation=ManipulationType.NONE,
                        compression=_COMPRESSION_MAP.get(
                            self._compression, CompressionLevel.UNKNOWN
                        ),
                        video_id=video.stem,
                        subject_id=video.stem,
                    )
                )
        return samples

    def _collect_fake_sequences(self) -> list[SampleEntity]:
        """Collect fake (manipulated) video sequences.

        Returns:
            List of fake SampleEntity objects.
        """
        samples: list[SampleEntity] = []
        manip_root = self._root / "manipulated_sequences"

        for manip_name in self._manipulation_types:
            manip_dir = manip_root / manip_name / self._compression / "videos"
            if not manip_dir.exists():
                manip_dir = manip_root / manip_name / "raw" / "videos"
            if not manip_dir.exists():
                logger.debug("[FF++] Manip dir not found: '%s'", manip_dir)
                continue

            manipulation = _MANIPULATION_MAP.get(manip_name, ManipulationType.UNKNOWN)

            for video in sorted(manip_dir.iterdir()):
                if video.suffix.lower() not in _VIDEO_EXTENSIONS:
                    continue
                # FF++ video IDs are like "000_003.mp4" → base ID is "000"
                base_id = video.stem.split("_")[0]
                if not self._is_in_split(base_id):
                    continue
                samples.append(
                    SampleEntity.create(
                        path=video,
                        label=Label.FAKE,
                        dataset_name=DatasetName.FF_PLUS_PLUS,
                        media_type=MediaType.VIDEO,
                        manipulation=manipulation,
                        compression=_COMPRESSION_MAP.get(
                            self._compression, CompressionLevel.UNKNOWN
                        ),
                        video_id=video.stem,
                        subject_id=base_id,
                    )
                )
        return samples

    def _is_in_split(self, video_id: str) -> bool:
        """Check if a video ID belongs to the current split.

        Args:
            video_id: Video stem ID string.

        Returns:
            True if the video should be included in this split.
        """
        if self._official_split_ids is None:
            return True  # No official split — include all
        return video_id in self._official_split_ids


class FFPlusPlusLoader(IDatasetLoader):
    """IDatasetLoader adapter for FaceForensics++.

    Args:
        root:        FF++ root directory path.
        compression: Compression level to load ('c23' default).
    """

    def __init__(self, root: Path | str, compression: str = "c23") -> None:
        self._root = Path(root)
        self._compression = compression
        if not self._root.exists():
            raise DatasetNotFoundError(self._root, DatasetName.FF_PLUS_PLUS)

    @property
    def dataset_name(self) -> DatasetName:
        """Return dataset identifier."""
        return DatasetName.FF_PLUS_PLUS

    @property
    def root_path(self) -> Path:
        """Return the root directory path."""
        return self._root

    def load(self) -> list[SampleEntity]:
        """Return all samples across all splits.

        Returns:
            Combined list of SampleEntity objects.
        """
        all_samples: list[SampleEntity] = []
        for split in SplitName:
            try:
                ds = FFPlusPlusDataset(
                    root=self._root, split=split, compression=self._compression
                )
                all_samples.extend(ds.samples)
            except Exception as exc:
                logger.warning("[FF++] Could not load split '%s': %s", split, exc)
        return all_samples

    def get_sample_count(self) -> dict[str, int]:
        """Return approximate sample counts via directory scan.

        Returns:
            Dictionary with 'real', 'fake', 'total' keys.
        """
        real = sum(
            1
            for source in ["actors", "youtube"]
            for d in [
                self._root / "original_sequences" / source / self._compression / "videos",
                self._root / "original_sequences" / source / "raw" / "videos",
            ]
            if d.exists()
            for f in d.iterdir()
            if f.suffix.lower() in _VIDEO_EXTENSIONS
        )
        fake = sum(
            1
            for manip in _ALL_MANIPULATIONS
            for d in [
                self._root / "manipulated_sequences" / manip / self._compression / "videos",
                self._root / "manipulated_sequences" / manip / "raw" / "videos",
            ]
            if d.exists()
            for f in d.iterdir()
            if f.suffix.lower() in _VIDEO_EXTENSIONS
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
            name=DatasetName.FF_PLUS_PLUS,
            version="1.0",
            root_path=self._root,
            total_samples=counts["total"],
            real_count=counts["real"],
            fake_count=counts["fake"],
            created_at=datetime.now(tz=timezone.utc),
            description=(
                f"FaceForensics++ | compression={self._compression} | "
                f"manipulations={_ALL_MANIPULATIONS}"
            ),
            tags={"compression": self._compression},
        )
