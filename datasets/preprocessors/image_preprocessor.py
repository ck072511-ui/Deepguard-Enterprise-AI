"""
DeepGuard — datasets/preprocessors/image_preprocessor.py

OpenCV-based image preprocessing pipeline:
  - Colour space conversion (BGR ↔ RGB)
  - Resizing with quality-preserving interpolation
  - Per-channel normalization
  - Histogram equalization (CLAHE)
  - Image quality assessment

Implements IPreprocessor interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from core.interfaces.dataset_interface import IPreprocessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizationParams:
    """Immutable normalization parameter set.

    Attributes:
        mean:  Per-channel mean values for subtraction.
        std:   Per-channel standard deviation for scaling.
        scale: Pixel value scale factor (default 255.0 → [0, 1]).
    """

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    scale: float = 255.0

    @classmethod
    def imagenet(cls) -> "NormalizationParams":
        """Return standard ImageNet normalization parameters."""
        return cls(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    @classmethod
    def zero_one(cls) -> "NormalizationParams":
        """Return simple [0, 1] scaling with no mean subtraction."""
        return cls(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))


class ImagePreprocessor(IPreprocessor):
    """OpenCV-based image preprocessing for deepfake detection.

    Applies a configurable pipeline of:
        1. BGR → RGB conversion (if input is from OpenCV)
        2. Resize to target square resolution
        3. Optional CLAHE histogram equalization (luminance channel)
        4. Optional per-channel normalization

    Args:
        target_size:    Output image height and width (square).
        input_is_bgr:   Set True if loading with OpenCV (auto-converts to RGB).
        apply_clahe:    Apply CLAHE histogram equalization before normalization.
        normalize:      Apply ImageNet-style channel normalization.
        norm_params:    Normalization parameters (defaults to ImageNet).
        interpolation:  OpenCV resize interpolation flag.
    """

    def __init__(
        self,
        target_size: int = 224,
        *,
        input_is_bgr: bool = True,
        apply_clahe: bool = False,
        normalize: bool = False,
        norm_params: NormalizationParams | None = None,
        interpolation: int = cv2.INTER_CUBIC,
    ) -> None:
        self._target_size = target_size
        self._input_is_bgr = input_is_bgr
        self._apply_clahe = apply_clahe
        self._normalize = normalize
        self._norm_params = norm_params or NormalizationParams.imagenet()
        self._interpolation = interpolation
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        logger.debug(
            "ImagePreprocessor | size=%d bgr=%s clahe=%s normalize=%s",
            target_size,
            input_is_bgr,
            apply_clahe,
            normalize,
        )

    # ------------------------------------------------------------------
    # IPreprocessor interface
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply the full preprocessing pipeline to a single image.

        Args:
            image: Input numpy array (H, W, C) uint8, BGR or RGB.

        Returns:
            Preprocessed float32 or uint8 numpy array (H, W, C) RGB.

        Raises:
            ValueError: If the input image is None or has wrong shape.
        """
        self._validate_image(image)

        # Step 1: colour space
        img = self._convert_colour(image)

        # Step 2: resize
        img = self._resize(img)

        # Step 3: CLAHE (luminance only)
        if self._apply_clahe:
            img = self._apply_clahe_transform(img)

        # Step 4: normalization → float32
        if self._normalize:
            img = self._apply_normalization(img)

        return img

    def preprocess_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Apply preprocessing to a list of images.

        Args:
            images: List of BGR or RGB numpy arrays.

        Returns:
            List of preprocessed arrays.
        """
        return [self.preprocess(img) for img in images]

    # ------------------------------------------------------------------
    # Specific operations exposed as public API
    # ------------------------------------------------------------------

    def load_from_path(self, path: str | "Path") -> np.ndarray:  # noqa: F821
        """Load an image from disk using OpenCV and apply preprocessing.

        Args:
            path: Absolute path to an image file.

        Returns:
            Preprocessed RGB numpy array.

        Raises:
            ValueError: If the file cannot be read by OpenCV.
        """
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"OpenCV could not read image at '{path}'.")
        return self.preprocess(img)  # Input is BGR from OpenCV

    def resize(self, image: np.ndarray, size: int | None = None) -> np.ndarray:
        """Resize image to a square target resolution.

        Args:
            image: Input numpy array.
            size:  Target size; uses instance default if None.

        Returns:
            Resized numpy array.
        """
        target = size or self._target_size
        return cv2.resize(image, (target, target), interpolation=self._interpolation)

    def normalize_array(
        self,
        image: np.ndarray,
        params: NormalizationParams | None = None,
    ) -> np.ndarray:
        """Normalize a uint8 image array to float32.

        Args:
            image:  Input uint8 RGB numpy array.
            params: Normalization parameters (uses instance default if None).

        Returns:
            Float32 numpy array normalized per-channel.
        """
        p = params or self._norm_params
        img_f = image.astype(np.float32) / p.scale
        mean = np.array(p.mean, dtype=np.float32)
        std = np.array(p.std, dtype=np.float32)
        return (img_f - mean) / std

    def compute_image_statistics(self, image: np.ndarray) -> dict[str, Any]:
        """Compute per-channel statistics for a single image.

        Args:
            image: RGB uint8 numpy array (H, W, 3).

        Returns:
            Dictionary with keys: mean, std, min, max (per channel),
            plus overall sharpness (Laplacian variance).
        """
        self._validate_image(image)
        stats: dict[str, Any] = {}
        channel_names = ["R", "G", "B"]
        for i, ch in enumerate(channel_names):
            channel = image[:, :, i].astype(np.float32)
            stats[f"{ch}_mean"] = float(np.mean(channel))
            stats[f"{ch}_std"] = float(np.std(channel))
            stats[f"{ch}_min"] = int(np.min(channel))
            stats[f"{ch}_max"] = int(np.max(channel))

        # Sharpness via Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        stats["sharpness"] = round(float(laplacian_var), 4)
        stats["height"] = image.shape[0]
        stats["width"] = image.shape[1]
        stats["aspect_ratio"] = round(image.shape[1] / max(image.shape[0], 1), 4)
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Raise ValueError for invalid image inputs.

        Args:
            image: Input to validate.

        Raises:
            ValueError: If image is None, not a numpy array, or has wrong dims.
        """
        if image is None:
            raise ValueError("Input image is None.")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected np.ndarray, got {type(image).__name__}.")
        if image.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D.")
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Image must have 1, 3, or 4 channels, got {image.shape[2]}."
            )

    def _convert_colour(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB if the input flag is set.

        Args:
            image: BGR or RGB numpy array.

        Returns:
            RGB numpy array.
        """
        if self._input_is_bgr and image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return image

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to target_size × target_size.

        Args:
            image: Input numpy array.

        Returns:
            Resized array.
        """
        h, w = image.shape[:2]
        if h == self._target_size and w == self._target_size:
            return image
        return cv2.resize(
            image,
            (self._target_size, self._target_size),
            interpolation=self._interpolation,
        )

    def _apply_clahe_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel of the image in LAB colour space.

        Preserves colour while enhancing local contrast.

        Args:
            image: RGB uint8 numpy array.

        Returns:
            CLAHE-enhanced RGB uint8 numpy array.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_eq = self._clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    def _apply_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply per-channel normalization to a uint8 image.

        Args:
            image: RGB uint8 numpy array.

        Returns:
            Float32 normalized numpy array.
        """
        return self.normalize_array(image, self._norm_params)
