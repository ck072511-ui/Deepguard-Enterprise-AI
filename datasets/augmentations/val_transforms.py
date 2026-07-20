"""
DeepGuard — datasets/augmentations/val_transforms.py

Albumentations-based deterministic transforms for validation and test sets.
No stochastic augmentation — only resizing and normalization are applied
to ensure reproducible evaluation results.

Also provides a Test-Time Augmentation (TTA) pipeline that applies
a fixed set of flip/crop variants for ensemble-style inference.
"""

from __future__ import annotations

import logging
from typing import Callable

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def build_val_transforms(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    *,
    center_crop: bool = True,
) -> A.Compose:
    """Build a deterministic validation / test augmentation pipeline.

    Applies only geometric alignment (resize / center-crop) and ImageNet
    normalization. No random transforms.

    Args:
        image_size:  Target square image size in pixels.
        mean:        Channel-wise normalization mean.
        std:         Channel-wise normalization std.
        center_crop: If True, resize-then-center-crop; otherwise just resize.

    Returns:
        Compiled Albumentations ``Compose`` transform.
    """
    logger.debug(
        "Building val/test transforms | size=%d center_crop=%s", image_size, center_crop
    )

    spatial: list[A.BasicTransform]
    if center_crop:
        # Resize slightly larger then take center crop — standard ImageNet eval
        resize_size = int(image_size * 256 / 224)
        spatial = [
            A.Resize(height=resize_size, width=resize_size, p=1.0),
            A.CenterCrop(height=image_size, width=image_size, p=1.0),
        ]
    else:
        spatial = [
            A.Resize(height=image_size, width=image_size, p=1.0),
        ]

    pipeline = A.Compose(
        [
            *spatial,
            A.Normalize(mean=list(mean), std=list(std), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    logger.debug("Val/test transforms compiled.")
    return pipeline


def build_tta_transforms(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> list[A.Compose]:
    """Build a list of Test-Time Augmentation (TTA) pipelines.

    Returns 5 deterministic views of the same image:
        1. Center crop (identity)
        2. Horizontal flip + center crop
        3. Top-left crop
        4. Top-right crop
        5. Bottom-center crop

    This ensemble reduces stochastic variance during inference.

    Args:
        image_size: Target square image size.
        mean:       Normalization mean.
        std:        Normalization std.

    Returns:
        List of 5 compiled Albumentations Compose pipelines.
    """
    logger.debug("Building TTA transforms | size=%d variants=5", image_size)

    resize_size = int(image_size * 256 / 224)

    def _base(extra: list[A.BasicTransform]) -> A.Compose:
        return A.Compose(
            [
                A.Resize(height=resize_size, width=resize_size),
                *extra,
                A.Normalize(mean=list(mean), std=list(std), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

    return [
        # 1. Standard center crop
        _base([A.CenterCrop(height=image_size, width=image_size)]),
        # 2. Horizontal flip + center crop
        _base([
            A.HorizontalFlip(p=1.0),
            A.CenterCrop(height=image_size, width=image_size),
        ]),
        # 3. Top-left
        _base([A.Crop(x_min=0, y_min=0, x_max=image_size, y_max=image_size)]),
        # 4. Top-right
        _base([
            A.Crop(
                x_min=resize_size - image_size,
                y_min=0,
                x_max=resize_size,
                y_max=image_size,
            )
        ]),
        # 5. Bottom-center
        _base([
            A.Crop(
                x_min=(resize_size - image_size) // 2,
                y_min=resize_size - image_size,
                x_max=(resize_size - image_size) // 2 + image_size,
                y_max=resize_size,
            )
        ]),
    ]


def apply_tta(
    image: np.ndarray,
    tta_transforms: list[A.Compose],
) -> list[np.ndarray]:
    """Apply all TTA transforms to a single image and return tensor list.

    Args:
        image:           Input RGB numpy array (H, W, C) uint8.
        tta_transforms:  List of TTA Compose pipelines from build_tta_transforms.

    Returns:
        List of PyTorch tensors (C, H, W) float32, one per TTA variant.
    """
    results = []
    for transform in tta_transforms:
        augmented = transform(image=image)
        results.append(augmented["image"])
    return results


def get_normalization_params() -> dict[str, tuple[float, float, float]]:
    """Return the default ImageNet normalization parameters.

    Returns:
        Dictionary with 'mean' and 'std' tuples.
    """
    return {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }


def denormalize(
    tensor: "torch.Tensor",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> "torch.Tensor":
    """Reverse ImageNet normalization for visualization.

    Applies: pixel = tensor * std + mean, then clamps to [0, 1].

    Args:
        tensor: Normalized float32 tensor of shape (C, H, W) or (B, C, H, W).
        mean:   Per-channel means used during normalization.
        std:    Per-channel stds used during normalization.

    Returns:
        Denormalized tensor with values in [0.0, 1.0].
    """
    import torch

    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 4:
        # Batch: (B, C, H, W)
        mean_t = mean_t.view(1, 3, 1, 1)
        std_t = std_t.view(1, 3, 1, 1)
    elif tensor.ndim == 3:
        # Single: (C, H, W)
        mean_t = mean_t.view(3, 1, 1)
        std_t = std_t.view(3, 1, 1)

    return (tensor * std_t + mean_t).clamp(0.0, 1.0)
