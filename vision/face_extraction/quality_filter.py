"""
DeepGuard — vision/face_extraction/quality_filter.py

Image quality filtering for extracted face crops.

Filters run fast (no DNN) on each extracted face crop and produce a
``QualityResult`` that decides whether the crop is usable for training
or inference.

Checks performed:
  1. Minimum face size   — reject crops smaller than N pixels on shortest side
  2. Aspect ratio        — reject extreme portrait/landscape crops
  3. Laplacian sharpness — reject blurry faces (motion blur, out-of-focus)
  4. Brightness         — reject over/under-exposed faces
  5. Histogram spread   — reject near-uniform (solid colour) images
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from vision.face_extraction.base import QualityResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable thresholds for the quality filter.

    Attributes:
        min_face_size:       Minimum shortest-side length in pixels.
        max_aspect_ratio:    Maximum width/height ratio (rejects very wide/tall crops).
        min_sharpness:       Minimum Laplacian variance (higher = sharper).
        min_brightness:      Minimum mean pixel brightness [0, 255].
        max_brightness:      Maximum mean pixel brightness [0, 255].
        min_histogram_spread: Minimum grey-level std to reject uniform patches.
    """

    min_face_size: int = 32
    max_aspect_ratio: float = 2.5
    min_sharpness: float = 20.0
    min_brightness: float = 10.0
    max_brightness: float = 245.0
    min_histogram_spread: float = 8.0

    @classmethod
    def strict(cls) -> "QualityThresholds":
        """Return thresholds suitable for high-quality training data.

        Returns:
            QualityThresholds with tight bounds.
        """
        return cls(
            min_face_size=64,
            max_aspect_ratio=1.8,
            min_sharpness=50.0,
            min_brightness=20.0,
            max_brightness=235.0,
            min_histogram_spread=15.0,
        )

    @classmethod
    def relaxed(cls) -> "QualityThresholds":
        """Return thresholds suitable for inference (tolerates more noise).

        Returns:
            QualityThresholds with loose bounds.
        """
        return cls(
            min_face_size=24,
            max_aspect_ratio=3.5,
            min_sharpness=5.0,
            min_brightness=5.0,
            max_brightness=250.0,
            min_histogram_spread=3.0,
        )


class QualityFilter:
    """Evaluates image quality of extracted face crops.

    All checks are performed on the crop image (not the original frame),
    so the filter should be applied after face cropping.

    Args:
        thresholds: Quality thresholds to use. Defaults to standard thresholds.
    """

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self._thresholds = thresholds or QualityThresholds()

    @property
    def thresholds(self) -> QualityThresholds:
        """Return the active quality thresholds."""
        return self._thresholds

    def evaluate(self, face_crop: np.ndarray) -> QualityResult:
        """Evaluate quality of a face crop image.

        Runs all checks in sequence; stops at the first failure and
        returns an informative reason string.

        Args:
            face_crop: RGB uint8 numpy array of the extracted face region.

        Returns:
            QualityResult with passed=True/False and computed metrics.

        Raises:
            ValueError: If face_crop is None or wrong dtype.
        """
        if face_crop is None or not isinstance(face_crop, np.ndarray):
            raise ValueError("face_crop must be a numpy array")
        if face_crop.size == 0:
            return QualityResult(
                passed=False, sharpness=0.0, face_size=0, aspect_ratio=0.0,
                reason="EMPTY_IMAGE",
            )

        h, w = face_crop.shape[:2]
        face_size = min(h, w)
        aspect_ratio = w / max(h, 1)

        # Check 1: minimum face size
        if face_size < self._thresholds.min_face_size:
            return QualityResult(
                passed=False,
                sharpness=0.0,
                face_size=face_size,
                aspect_ratio=aspect_ratio,
                reason=f"TOO_SMALL:{face_size}px<{self._thresholds.min_face_size}px",
            )

        # Check 2: aspect ratio
        if aspect_ratio > self._thresholds.max_aspect_ratio:
            return QualityResult(
                passed=False,
                sharpness=0.0,
                face_size=face_size,
                aspect_ratio=aspect_ratio,
                reason=f"BAD_ASPECT_RATIO:{aspect_ratio:.2f}>{self._thresholds.max_aspect_ratio}",
            )

        # Convert to grey for intensity checks
        grey = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)

        # Check 3: brightness
        brightness = float(grey.mean())
        if brightness < self._thresholds.min_brightness:
            return QualityResult(
                passed=False,
                sharpness=0.0,
                face_size=face_size,
                aspect_ratio=aspect_ratio,
                reason=f"UNDEREXPOSED:{brightness:.1f}<{self._thresholds.min_brightness}",
            )
        if brightness > self._thresholds.max_brightness:
            return QualityResult(
                passed=False,
                sharpness=0.0,
                face_size=face_size,
                aspect_ratio=aspect_ratio,
                reason=f"OVEREXPOSED:{brightness:.1f}>{self._thresholds.max_brightness}",
            )

        # Check 4: histogram spread (uniform patches)
        spread = float(grey.std())
        if spread < self._thresholds.min_histogram_spread:
            return QualityResult(
                passed=False,
                sharpness=0.0,
                face_size=face_size,
                aspect_ratio=aspect_ratio,
                reason=f"LOW_CONTRAST:{spread:.1f}<{self._thresholds.min_histogram_spread}",
            )

        # Check 5: sharpness (Laplacian variance)
        sharpness = self._compute_sharpness(grey)
        if sharpness < self._thresholds.min_sharpness:
            return QualityResult(
                passed=False,
                sharpness=sharpness,
                face_size=face_size,
                aspect_ratio=aspect_ratio,
                reason=f"BLURRY:{sharpness:.1f}<{self._thresholds.min_sharpness}",
            )

        return QualityResult(
            passed=True,
            sharpness=sharpness,
            face_size=face_size,
            aspect_ratio=aspect_ratio,
            reason="OK",
        )

    def filter_batch(
        self, face_crops: list[np.ndarray]
    ) -> list[tuple[np.ndarray, QualityResult]]:
        """Evaluate a batch of face crops and return (crop, result) pairs.

        Args:
            face_crops: List of RGB uint8 face crop arrays.

        Returns:
            List of (crop, QualityResult) for all inputs (including failures).
        """
        return [(crop, self.evaluate(crop)) for crop in face_crops]

    def filter_passed(
        self, face_crops: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Return only the face crops that pass all quality checks.

        Args:
            face_crops: List of RGB uint8 face crop arrays.

        Returns:
            Filtered list containing only passing crops.
        """
        return [crop for crop in face_crops if self.evaluate(crop).passed]

    def compute_quality_score(self, face_crop: np.ndarray) -> float:
        """Compute a continuous quality score in [0.0, 1.0].

        Higher is better. Combines sharpness, brightness distance from ideal,
        and size into a single scalar.

        Args:
            face_crop: RGB uint8 face crop array.

        Returns:
            Quality score in [0.0, 1.0].
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0

        h, w = face_crop.shape[:2]
        grey = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)

        # Sharpness component (normalised to [0, 1] with soft cap at 200)
        sharpness = self._compute_sharpness(grey)
        sharpness_score = min(sharpness / 200.0, 1.0)

        # Brightness component (distance from ideal 128)
        brightness = float(grey.mean())
        brightness_score = 1.0 - abs(brightness - 128.0) / 128.0
        brightness_score = max(0.0, brightness_score)

        # Size component (normalised to [0, 1] with soft cap at 224)
        size_score = min(min(h, w) / 224.0, 1.0)

        # Weighted combination
        return 0.5 * sharpness_score + 0.3 * brightness_score + 0.2 * size_score

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sharpness(grey: np.ndarray) -> float:
        """Compute Laplacian variance as a sharpness measure.

        Args:
            grey: Greyscale uint8 image.

        Returns:
            Laplacian variance (higher = sharper).
        """
        return float(cv2.Laplacian(grey, cv2.CV_64F).var())
