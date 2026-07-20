"""
DeepGuard — datasets/augmentations/train_transforms.py

Albumentations-based augmentation pipeline for training.
Applies aggressive, realistic perturbations that simulate real-world
imaging conditions, compression artifacts, and deepfake residuals.

Design Principles:
- All transforms are reproducible via seed control.
- Compression-quality transforms mimic JPEG artifacts common in deepfakes.
- Face-region-aware: does not distort face geometry beyond plausibility.
"""

from __future__ import annotations

import logging
from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def build_train_transforms(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    *,
    use_tta: bool = False,
    severity: str = "medium",
) -> A.Compose:
    """Build the Albumentations augmentation pipeline for training.

    Augmentation severity levels:
        - ``light``:  Minimal, safe augmentations for small datasets.
        - ``medium``: Standard balanced augmentation (default).
        - ``heavy``:  Aggressive augmentation for larger datasets.

    Args:
        image_size: Target square image size in pixels.
        mean:       Channel-wise normalization mean (ImageNet default).
        std:        Channel-wise normalization std (ImageNet default).
        use_tta:    If True, returns a TTA-compatible pipeline (unused here,
                    kept for interface consistency with val_transforms).
        severity:   Augmentation strength: 'light' | 'medium' | 'heavy'.

    Returns:
        Compiled Albumentations ``Compose`` transform ready for ``__call__``.

    Raises:
        ValueError: If ``severity`` is not one of the accepted values.
    """
    accepted_severities = {"light", "medium", "heavy"}
    if severity not in accepted_severities:
        raise ValueError(
            f"Invalid severity '{severity}'. Must be one of {accepted_severities}."
        )

    logger.debug(
        "Building train transforms | size=%d severity=%s tta=%s",
        image_size,
        severity,
        use_tta,
    )

    # ── Spatial Transforms ────────────────────────────────
    spatial_transforms = _build_spatial(image_size, severity)

    # ── Colour / Photometric Transforms ──────────────────
    colour_transforms = _build_colour(severity)

    # ── Noise & Degradation ───────────────────────────────
    degradation_transforms = _build_degradation(severity)

    # ── Regularisation (Cutout / CoarseDropout) ───────────
    regularisation_transforms = _build_regularisation(severity)

    pipeline = A.Compose(
        [
            *spatial_transforms,
            *colour_transforms,
            *degradation_transforms,
            *regularisation_transforms,
            A.Normalize(mean=list(mean), std=list(std), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    logger.debug("Train transforms compiled with %d stages.", len(pipeline.transforms))
    return pipeline


# ---------------------------------------------------------------------------
# Private helpers — each builds a self-contained transform list
# ---------------------------------------------------------------------------


def _build_spatial(image_size: int, severity: str) -> list[A.BasicTransform]:
    """Build spatial / geometric augmentation transforms.

    Args:
        image_size: Target output resolution.
        severity:   Augmentation strength level.

    Returns:
        List of spatial Albumentations transforms.
    """
    scale_map = {
        "light": (0.9, 1.0),
        "medium": (0.8, 1.0),
        "heavy": (0.7, 1.0),
    }
    rotate_map = {"light": 5, "medium": 15, "heavy": 25}
    flip_p_map = {"light": 0.3, "medium": 0.5, "heavy": 0.5}

    return [
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=scale_map[severity],
            ratio=(0.75, 1.33),
            p=1.0,
        ),
        A.HorizontalFlip(p=flip_p_map[severity]),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=rotate_map[severity],
            border_mode=0,
            p=0.3 if severity != "light" else 0.1,
        ),
        A.Perspective(scale=(0.02, 0.05), p=0.1 if severity == "heavy" else 0.0),
    ]


def _build_colour(severity: str) -> list[A.BasicTransform]:
    """Build photometric / colour augmentation transforms.

    Args:
        severity: Augmentation strength level.

    Returns:
        List of colour Albumentations transforms.
    """
    jitter_p_map = {"light": 0.3, "medium": 0.5, "heavy": 0.7}
    jitter_brightness = {"light": 0.1, "medium": 0.2, "heavy": 0.4}
    jitter_contrast = {"light": 0.1, "medium": 0.2, "heavy": 0.4}

    return [
        A.OneOf(
            [
                A.ColorJitter(
                    brightness=jitter_brightness[severity],
                    contrast=jitter_contrast[severity],
                    saturation=0.2,
                    hue=0.1,
                    p=1.0,
                ),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0,
                ),
            ],
            p=jitter_p_map[severity],
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3 if severity != "light" else 0.0,
        ),
        A.ToGray(p=0.03),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
    ]


def _build_degradation(severity: str) -> list[A.BasicTransform]:
    """Build image degradation transforms that simulate real-world artifacts.

    These are crucial for deepfake detection, as manipulation pipelines
    often leave specific compression and blurring signatures.

    Args:
        severity: Augmentation strength level.

    Returns:
        List of degradation Albumentations transforms.
    """
    compression_p = {"light": 0.2, "medium": 0.4, "heavy": 0.6}
    blur_p = {"light": 0.1, "medium": 0.3, "heavy": 0.5}

    return [
        # JPEG compression simulation — critical for detecting compression
        # artefacts that deepfake generators often introduce
        A.OneOf(
            [
                A.ImageCompression(
                    quality_range=(40, 95),
                    p=1.0,
                ),
                A.Downscale(
                    scale_range=(0.5, 0.9),
                    p=1.0,
                ),
            ],
            p=compression_p[severity],
        ),
        # Blur — simulates motion blur and out-of-focus faces in videos
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ],
            p=blur_p[severity],
        ),
        # Noise — simulates sensor noise and video encoding noise
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.04, 0.22), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ],
            p=0.2 if severity != "light" else 0.05,
        ),
    ]


def _build_regularisation(severity: str) -> list[A.BasicTransform]:
    """Build regularisation / occlusion transforms.

    Args:
        severity: Augmentation strength level.

    Returns:
        List of regularisation Albumentations transforms.
    """
    if severity == "light":
        return []

    max_holes = {"medium": 4, "heavy": 8}
    max_height = {"medium": 24, "heavy": 40}

    return [
        A.CoarseDropout(
            num_holes_range=(1, max_holes[severity]),
            hole_height_range=(8, max_height[severity]),
            hole_width_range=(8, max_height[severity]),
            fill=0,
            p=0.25,
        ),
    ]


def get_transform_params(transform: A.Compose) -> dict[str, Any]:
    """Extract a human-readable summary of all transforms in a pipeline.

    Args:
        transform: Compiled Albumentations Compose pipeline.

    Returns:
        Dictionary with transform names and their parameters.
    """
    params: dict[str, Any] = {}
    for t in transform.transforms:
        name = type(t).__name__
        params[name] = {
            "p": getattr(t, "p", None),
        }
    return params
