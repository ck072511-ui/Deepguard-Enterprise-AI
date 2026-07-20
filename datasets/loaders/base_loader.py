"""
DeepGuard — datasets/loaders/base_loader.py

Abstract base class for all dataset loaders, providing:
  - PyTorch Dataset interface (subclass of torch.utils.data.Dataset)
  - Shared sample loading, caching, and transform application logic
  - Error handling with structured logging
  - Transparent disk caching of preprocessed face crops

All concrete loaders (CelebDF, FF++, DFDC, Custom) extend BaseDeepfakeDataset.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.domain.entities.dataset_entity import Label, SampleEntity, SplitName
from core.exceptions.dataset_exceptions import DatasetNotFoundError, EmptyDatasetError

logger = logging.getLogger(__name__)


class BaseDeepfakeDataset(Dataset):
    """Abstract PyTorch Dataset providing shared infrastructure for all loaders.

    Responsibilities of this class:
        - Validate the root path on construction.
        - Load and cache preprocessed images.
        - Apply Albumentations transforms if provided.
        - Enforce interface through abstract methods.

    Responsibilities of subclasses:
        - Implement ``_discover_samples()`` to return the raw file list.
        - Implement ``dataset_name`` property.

    Args:
        root:            Path to the dataset root directory.
        split:           Dataset partition to load (train/val/test).
        transform:       Albumentations Compose transform or None.
        face_extractor:  IFaceExtractor or None (skip face extraction).
        use_cache:       Cache preprocessed crops to disk for speed.
        cache_dir:       Directory for disk cache (default: root/.cache).
        image_size:      Target output image size in pixels.
        max_samples:     Limit number of samples (None = all). For debugging.
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
    ) -> None:
        super().__init__()
        self._root = Path(root)
        self._split = split
        self._transform = transform
        self._face_extractor = face_extractor
        self._use_cache = use_cache
        self._image_size = image_size

        if not self._root.exists():
            raise DatasetNotFoundError(self._root, str(self.dataset_name))

        self._cache_dir: Path = (
            Path(cache_dir) if cache_dir else self._root / ".cache" / str(split)
        )
        if self._use_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Discover samples — implemented by each concrete subclass
        self._samples: list[SampleEntity] = self._discover_samples()

        if not self._samples:
            raise EmptyDatasetError(str(self.dataset_name), f"split={split}")

        if max_samples is not None:
            self._samples = self._samples[:max_samples]

        logger.info(
            "[%s] %s split loaded | samples=%d real=%d fake=%d",
            self.dataset_name,
            split,
            len(self._samples),
            self._count_label(Label.REAL),
            self._count_label(Label.FAKE),
        )

    # ------------------------------------------------------------------
    # Abstract interface — implemented by each concrete loader
    # ------------------------------------------------------------------

    @abstractmethod
    def _discover_samples(self) -> list[SampleEntity]:
        """Discover all samples for this dataset and split.

        Returns:
            Ordered list of SampleEntity objects.
        """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the dataset identifier string."""

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of samples in this split."""
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load, preprocess, and transform a single sample.

        Args:
            index: Integer index into the sample list.

        Returns:
            Tuple of (image_tensor, label_int).
            image_tensor: Float32 tensor of shape (C, H, W).
            label_int:    0 = real, 1 = fake.

        Raises:
            IndexError: If index is out of bounds.
            RuntimeError: If the image cannot be loaded after retries.
        """
        if index < 0 or index >= len(self._samples):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self._samples)}."
            )

        sample = self._samples[index]

        # Try disk cache first
        image: np.ndarray | None = None
        cache_key = self._make_cache_key(sample)
        if self._use_cache:
            image = self._load_from_cache(cache_key)

        if image is None:
            image = self._load_image(sample)
            if self._use_cache and image is not None:
                self._save_to_cache(cache_key, image)

        if image is None:
            logger.error("Failed to load sample %d (path=%s).", index, sample.path)
            raise RuntimeError(
                f"Could not load image for sample {sample.sample_id} "
                f"at path '{sample.path}'."
            )

        # Apply Albumentations transform
        if self._transform is not None:
            augmented = self._transform(image=image)
            tensor: torch.Tensor = augmented["image"]
        else:
            tensor = torch.from_numpy(
                np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
            )

        return tensor, int(sample.label)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def samples(self) -> list[SampleEntity]:
        """Return the list of all SampleEntity objects in this split."""
        return self._samples

    @property
    def split(self) -> SplitName:
        """Return the split name this dataset represents."""
        return self._split

    @property
    def class_counts(self) -> dict[str, int]:
        """Return sample counts per class.

        Returns:
            Dictionary with keys 'real', 'fake', 'total'.
        """
        return {
            "real": self._count_label(Label.REAL),
            "fake": self._count_label(Label.FAKE),
            "total": len(self._samples),
        }

    @property
    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced training.

        Returns:
            Float32 tensor of shape (2,) = [weight_real, weight_fake].
        """
        counts = self.class_counts
        total = counts["total"]
        n_real = max(counts["real"], 1)
        n_fake = max(counts["fake"], 1)
        w_real = total / (2.0 * n_real)
        w_fake = total / (2.0 * n_fake)
        return torch.tensor([w_real, w_fake], dtype=torch.float32)

    def get_labels(self) -> list[int]:
        """Return all labels as a flat integer list (for samplers).

        Returns:
            List of 0/1 integers.
        """
        return [int(s.label) for s in self._samples]

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self, sample: SampleEntity) -> np.ndarray | None:
        """Load an image from a SampleEntity path.

        For video paths, delegates to video_preprocessor via subclass override.
        For image paths, loads directly with OpenCV.

        Args:
            sample: SampleEntity whose path to load.

        Returns:
            RGB uint8 numpy array, or None if loading fails.
        """
        path = sample.path
        if not path.exists():
            logger.warning("File not found: '%s'", path)
            return None

        suffix = path.suffix.lower()
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        if suffix in video_extensions:
            return self._load_from_video(sample)

        # Image loading via OpenCV
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("OpenCV failed to read '%s'.", path)
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Face extraction
        if self._face_extractor is not None:
            faces = self._face_extractor.extract_faces(rgb)
            if faces:
                return faces[0]

        return cv2.resize(rgb, (self._image_size, self._image_size))

    def _load_from_video(self, sample: SampleEntity) -> np.ndarray | None:
        """Load a single representative frame from a video sample.

        Uses frame_index if set, otherwise takes the middle frame.

        Args:
            sample: SampleEntity with a video path.

        Returns:
            RGB uint8 numpy array, or None on failure.
        """
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            logger.warning("OpenCV cannot open video '%s'.", sample.path)
            return None

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_frame = sample.frame_index if sample.frame_index >= 0 else total // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret or frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._face_extractor is not None:
            faces = self._face_extractor.extract_faces(rgb)
            if faces:
                return faces[0]

        return cv2.resize(rgb, (self._image_size, self._image_size))

    # ------------------------------------------------------------------
    # Disk caching
    # ------------------------------------------------------------------

    def _make_cache_key(self, sample: SampleEntity) -> str:
        """Compute a unique cache key for a sample.

        Args:
            sample: The sample to key.

        Returns:
            SHA-256 hex string (first 16 characters).
        """
        raw = f"{sample.path}|{sample.frame_index}|{self._image_size}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _load_from_cache(self, key: str) -> np.ndarray | None:
        """Attempt to load a preprocessed image from disk cache.

        Args:
            key: Cache key string.

        Returns:
            Cached numpy array, or None if not in cache.
        """
        cache_file = self._cache_dir / f"{key}.pkl"
        if not cache_file.exists():
            return None
        try:
            with cache_file.open("rb") as f:
                return pickle.load(f)  # noqa: S301
        except Exception as exc:
            logger.debug("Cache read failed for key=%s: %s", key, exc)
            return None

    def _save_to_cache(self, key: str, image: np.ndarray) -> None:
        """Persist a preprocessed image to disk cache.

        Args:
            key:   Cache key string.
            image: Preprocessed RGB numpy array to cache.
        """
        cache_file = self._cache_dir / f"{key}.pkl"
        try:
            with cache_file.open("wb") as f:
                pickle.dump(image, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.debug("Cache write failed for key=%s: %s", key, exc)

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _count_label(self, label: Label) -> int:
        """Count samples with a specific label.

        Args:
            label: Label to count.

        Returns:
            Integer count.
        """
        return sum(1 for s in self._samples if s.label == label)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"split={self._split}, "
            f"samples={len(self._samples)}, "
            f"root={self._root})"
        )
