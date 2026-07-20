"""
DeepGuard — datasets/factory.py

Dataset factory — the single entry point for creating PyTorch DataLoaders
for any supported dataset and split combination.

Follows the Factory Pattern to decouple consumer code from concrete
loader implementations.

Usage:
    >>> from datasets.factory import DatasetFactory
    >>> factory = DatasetFactory(config)
    >>> train_loader = factory.create_dataloader("ff++", "train")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.domain.entities.dataset_entity import DatasetName, SplitName
from core.exceptions.dataset_exceptions import UnsupportedDatasetError
from datasets.augmentations.train_transforms import build_train_transforms
from datasets.augmentations.val_transforms import build_val_transforms
from datasets.loaders.base_loader import BaseDeepfakeDataset
from datasets.loaders.celeb_df_loader import CelebDFDataset
from datasets.loaders.custom_loader import CustomDataset
from datasets.loaders.dfdc_loader import DFDCDataset
from datasets.loaders.ff_plus_plus_loader import FFPlusPlusDataset
from datasets.preprocessors.face_extractor import FaceExtractor, NullFaceExtractor

logger = logging.getLogger(__name__)


class DatasetFactory:
    """Factory for creating datasets and DataLoaders from configuration.

    Args:
        config: Dataset configuration dictionary (from dataset_config.yaml).
                Must have 'datasets', 'preprocessing', and 'augmentation' keys.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._preprocessing = config.get("preprocessing", {})
        self._aug_config = config.get("augmentation", {})

    def create_dataset(
        self,
        dataset_name: str,
        split: str,
        *,
        use_face_extraction: bool = True,
        use_cache: bool = False,
        max_samples: int | None = None,
        aug_severity: str = "medium",
        **kwargs: Any,
    ) -> BaseDeepfakeDataset:
        """Create a PyTorch Dataset for the given dataset and split.

        Args:
            dataset_name:        One of 'celeb-df', 'ff++', 'dfdc', 'custom'.
            split:               'train' | 'val' | 'test'.
            use_face_extraction: If True, apply face detection preprocessing.
            use_cache:           Enable disk caching of preprocessed faces.
            max_samples:         Limit sample count (None = all).
            aug_severity:        Augmentation strength: 'light'|'medium'|'heavy'.
            **kwargs:            Extra keyword arguments forwarded to the loader.

        Returns:
            Configured BaseDeepfakeDataset subclass instance.

        Raises:
            UnsupportedDatasetError: If dataset_name is not supported.
            DatasetNotFoundError:    If the dataset root does not exist.
        """
        try:
            name = DatasetName(dataset_name)
            split_enum = SplitName(split)
        except ValueError:
            supported = [str(d) for d in DatasetName]
            raise UnsupportedDatasetError(dataset_name, supported)

        # Build transform
        image_size = self._aug_config.get("image_size", 224)
        mean = tuple(self._aug_config.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]))
        std = tuple(self._aug_config.get("normalize", {}).get("std", [0.229, 0.224, 0.225]))

        if split_enum == SplitName.TRAIN:
            transform = build_train_transforms(
                image_size=image_size, mean=mean, std=std, severity=aug_severity  # type: ignore
            )
        else:
            transform = build_val_transforms(image_size=image_size, mean=mean, std=std)  # type: ignore

        # Build face extractor
        face_extractor = None
        if use_face_extraction:
            face_cfg = self._preprocessing.get("face_detection", {})
            backend = face_cfg.get("backend", "auto")
            min_conf = face_cfg.get("min_confidence", 0.9)
            keep_largest = face_cfg.get("keep_largest_only", False)
            fallback = face_cfg.get("fallback_to_full_frame", True)
            face_extractor = FaceExtractor(
                backend=backend,
                keep_largest_only=keep_largest,
                fallback_to_full=fallback,
                image_size=image_size,
            )

        # Resolve dataset root
        root = self._resolve_root(name)

        # Build dataset
        common_kwargs = dict(
            root=root,
            split=split_enum,
            transform=transform,
            face_extractor=face_extractor,
            use_cache=use_cache,
            image_size=image_size,
            max_samples=max_samples,
        )
        common_kwargs.update(kwargs)

        dataset = self._build_dataset(name, common_kwargs)
        logger.info(
            "[Factory] Created %s | split=%s samples=%d",
            type(dataset).__name__,
            split,
            len(dataset),
        )
        return dataset

    def create_dataloader(
        self,
        dataset_name: str,
        split: str,
        *,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_weighted_sampler: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        **kwargs: Any,
    ) -> DataLoader:
        """Create a configured PyTorch DataLoader.

        Args:
            dataset_name:         Dataset name.
            split:                'train' | 'val' | 'test'.
            batch_size:           Samples per batch.
            num_workers:          DataLoader worker processes.
            pin_memory:           Pin memory for faster GPU transfer.
            use_weighted_sampler: Balance batches via WeightedRandomSampler.
            prefetch_factor:      Prefetch batches per worker.
            persistent_workers:   Keep workers alive between epochs.
            **kwargs:             Extra arguments forwarded to create_dataset.

        Returns:
            Configured PyTorch DataLoader.
        """
        dataset = self.create_dataset(dataset_name, split, **kwargs)
        split_enum = SplitName(split)
        is_train = split_enum == SplitName.TRAIN

        sampler = None
        shuffle = is_train

        if use_weighted_sampler and is_train:
            sampler = self._build_weighted_sampler(dataset)
            shuffle = False  # Sampler and shuffle are mutually exclusive

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=is_train,  # Drop last incomplete batch during training
        )

        logger.info(
            "[Factory] Created DataLoader | split=%s batches=%d batch_size=%d",
            split,
            len(loader),
            batch_size,
        )
        return loader

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_root(self, name: DatasetName) -> Path:
        """Resolve the root directory path for a dataset from config.

        Args:
            name: Dataset identifier.

        Returns:
            Resolved absolute Path to the dataset root.
        """
        datasets_cfg = self._config.get("datasets", {})
        dataset_cfg = datasets_cfg.get(str(name), {})
        root_str = dataset_cfg.get("root", f"./datasets/raw/{name}")
        return Path(root_str).resolve()

    @staticmethod
    def _build_dataset(
        name: DatasetName,
        kwargs: dict[str, Any],
    ) -> BaseDeepfakeDataset:
        """Instantiate the concrete dataset class for a given name.

        Args:
            name:   Dataset identifier.
            kwargs: Constructor keyword arguments.

        Returns:
            Concrete BaseDeepfakeDataset subclass instance.
        """
        class_map: dict[DatasetName, type[BaseDeepfakeDataset]] = {
            DatasetName.CELEB_DF: CelebDFDataset,
            DatasetName.FF_PLUS_PLUS: FFPlusPlusDataset,
            DatasetName.DFDC: DFDCDataset,
            DatasetName.CUSTOM: CustomDataset,
        }
        dataset_class = class_map[name]
        return dataset_class(**kwargs)

    @staticmethod
    def _build_weighted_sampler(dataset: BaseDeepfakeDataset) -> WeightedRandomSampler:
        """Build a WeightedRandomSampler to handle class imbalance.

        Assigns each sample a weight inversely proportional to its class frequency.

        Args:
            dataset: Dataset instance with get_labels() method.

        Returns:
            Configured WeightedRandomSampler.
        """
        labels = dataset.get_labels()
        class_counts = [labels.count(0), labels.count(1)]  # [real, fake]
        total = len(labels)

        weights_per_class = [
            total / max(c, 1) for c in class_counts
        ]
        sample_weights = torch.tensor(
            [weights_per_class[lbl] for lbl in labels], dtype=torch.float32
        )

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
