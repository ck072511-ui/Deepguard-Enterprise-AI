"""
DeepGuard — vision/face_extraction/base.py

Core domain types for the face extraction module:
  - FaceDetection: immutable result of a single face detection
  - DetectionConfig: configuration value object for all detectors
  - IDetector: abstract base class all detector backends implement

These types are infrastructure-agnostic and carry no framework imports.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceDetection:
    """Immutable result of detecting a single face in an image.

    All coordinates are in pixel space (absolute, not normalised).

    Attributes:
        bbox_xyxy:    Bounding box as (x1, y1, x2, y2) integers.
        confidence:   Detection confidence in [0.0, 1.0].
        landmarks_5pt: Optional 5-point landmarks array shaped (5, 2).
                       Order: left-eye, right-eye, nose, left-mouth, right-mouth.
        face_id:      Monotonic index within the current image (0-based).
        backend:      Name of the detector that produced this result.
    """

    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    landmarks_5pt: np.ndarray | None = field(default=None, compare=False, hash=False)
    face_id: int = 0
    backend: str = ""

    @property
    def x1(self) -> int:
        """Left coordinate."""
        return self.bbox_xyxy[0]

    @property
    def y1(self) -> int:
        """Top coordinate."""
        return self.bbox_xyxy[1]

    @property
    def x2(self) -> int:
        """Right coordinate."""
        return self.bbox_xyxy[2]

    @property
    def y2(self) -> int:
        """Bottom coordinate."""
        return self.bbox_xyxy[3]

    @property
    def width(self) -> int:
        """Bounding box width in pixels."""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        """Bounding box height in pixels."""
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        """Bounding box area in pixels²."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height ratio."""
        return self.width / max(self.height, 1)

    @property
    def has_landmarks(self) -> bool:
        """Return True if 5-point landmarks are available."""
        return (
            self.landmarks_5pt is not None
            and isinstance(self.landmarks_5pt, np.ndarray)
            and self.landmarks_5pt.shape == (5, 2)
        )

    def clamp_to_image(self, image_w: int, image_h: int) -> "FaceDetection":
        """Return a new FaceDetection with bbox clamped to image bounds.

        Args:
            image_w: Image width in pixels.
            image_h: Image height in pixels.

        Returns:
            New FaceDetection with clamped coordinates.
        """
        x1 = max(0, min(self.x1, image_w - 1))
        y1 = max(0, min(self.y1, image_h - 1))
        x2 = max(0, min(self.x2, image_w))
        y2 = max(0, min(self.y2, image_h))
        return FaceDetection(
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=self.confidence,
            landmarks_5pt=self.landmarks_5pt,
            face_id=self.face_id,
            backend=self.backend,
        )

    def with_margin(self, margin: float, image_w: int, image_h: int) -> "FaceDetection":
        """Return expanded bbox with a fractional margin, clamped to image.

        Args:
            margin:  Fractional expansion relative to bbox dimensions (e.g. 0.3).
            image_w: Image width for clamping.
            image_h: Image height for clamping.

        Returns:
            New FaceDetection with expanded, clamped coordinates.
        """
        dw = int(self.width * margin)
        dh = int(self.height * margin)
        x1 = max(0, self.x1 - dw)
        y1 = max(0, self.y1 - dh)
        x2 = min(image_w, self.x2 + dw)
        y2 = min(image_h, self.y2 + dh)
        return FaceDetection(
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=self.confidence,
            landmarks_5pt=self.landmarks_5pt,
            face_id=self.face_id,
            backend=self.backend,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dict with bbox, confidence, landmark array, face_id, backend.
        """
        lm = None
        if self.landmarks_5pt is not None:
            lm = self.landmarks_5pt.tolist()
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": round(float(self.confidence), 6),
            "landmarks_5pt": lm,
            "face_id": self.face_id,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class QualityResult:
    """Result of the image quality filter for one face crop.

    Attributes:
        passed:       True if the face passes all quality thresholds.
        sharpness:    Laplacian variance (higher = sharper).
        face_size:    Minimum of width/height in pixels.
        aspect_ratio: Width/height ratio.
        reason:       Human-readable failure reason, or 'OK'.
    """

    passed: bool
    sharpness: float
    face_size: int
    aspect_ratio: float
    reason: str = "OK"


@dataclass(frozen=True)
class DetectionConfig:
    """Immutable configuration for all detector backends.

    Attributes:
        min_face_size:   Minimum face size in pixels (used to filter tiny faces).
        min_confidence:  Minimum detection confidence [0, 1].
        max_faces:       Maximum faces to return per image (0 = unlimited).
        margin:          Fractional margin to add around each detected face.
        keep_largest:    If True, return only the highest-area face.
    """

    min_face_size: int = 40
    min_confidence: float = 0.85
    max_faces: int = 0
    margin: float = 0.2
    keep_largest: bool = False

    def fingerprint(self) -> str:
        """Return a short hex string uniquely identifying this config.

        Used as part of cache keys.

        Returns:
            8-character hex fingerprint.
        """
        raw = (
            f"{self.min_face_size}|{self.min_confidence}|"
            f"{self.max_faces}|{self.margin}|{self.keep_largest}"
        ).encode()
        return hashlib.blake2b(raw, digest_size=4).hexdigest()

    @classmethod
    def default(cls) -> "DetectionConfig":
        """Return default configuration.

        Returns:
            DetectionConfig with sensible production defaults.
        """
        return cls()

    @classmethod
    def fast(cls) -> "DetectionConfig":
        """Return config tuned for speed over precision.

        Returns:
            DetectionConfig with relaxed thresholds.
        """
        return cls(min_face_size=80, min_confidence=0.75, max_faces=1, keep_largest=True)

    @classmethod
    def high_quality(cls) -> "DetectionConfig":
        """Return config tuned for maximum precision.

        Returns:
            DetectionConfig with tight thresholds.
        """
        return cls(min_face_size=40, min_confidence=0.95, max_faces=0, margin=0.3)


# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------


class IDetector(ABC):
    """Abstract interface all face detector backends must implement.

    Subclasses wrap a specific detection library and translate its output
    into a list of ``FaceDetection`` objects.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the detector backend identifier string."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the compute device ('cpu', 'cuda:0', etc.)."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if all required libraries are installed."""

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[FaceDetection]:
        """Detect faces in a single RGB image.

        Args:
            image:  RGB uint8 numpy array of shape (H, W, 3).
            config: Optional override for detection parameters.

        Returns:
            List of FaceDetection objects, sorted by confidence descending.
        """

    @abstractmethod
    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        config: DetectionConfig | None = None,
    ) -> list[list[FaceDetection]]:
        """Detect faces in a batch of RGB images.

        Args:
            images: Sequence of RGB uint8 arrays.
            config: Optional override for detection parameters.

        Returns:
            List of detection lists, one per input image.
        """

    def _validate_image(self, image: np.ndarray) -> None:
        """Validate that the input is a non-empty RGB uint8 array.

        Args:
            image: Input to validate.

        Raises:
            ValueError: If the image fails validation.
        """
        if image is None:
            raise ValueError("image must not be None")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"image must be a numpy array, got {type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must be (H, W, 3) RGB array, got shape {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"image must be uint8, got {image.dtype}")
        if image.size == 0:
            raise ValueError("image must not be empty")

    def _apply_config_filters(
        self,
        detections: list[FaceDetection],
        config: DetectionConfig,
        image_w: int,
        image_h: int,
    ) -> list[FaceDetection]:
        """Apply min_confidence, min_face_size, max_faces, keep_largest filters.

        Args:
            detections: Raw detections from the backend.
            config:     Detection configuration.
            image_w:    Image width for margin clamping.
            image_h:    Image height for margin clamping.

        Returns:
            Filtered, re-indexed detections.
        """
        # Confidence filter
        filtered = [d for d in detections if d.confidence >= config.min_confidence]

        # Size filter
        filtered = [d for d in filtered if min(d.width, d.height) >= config.min_face_size]

        # Apply margin expansion
        if config.margin > 0:
            filtered = [d.with_margin(config.margin, image_w, image_h) for d in filtered]

        # Sort by confidence descending
        filtered.sort(key=lambda d: d.confidence, reverse=True)

        # Keep largest only
        if config.keep_largest and filtered:
            largest = max(filtered, key=lambda d: d.area)
            filtered = [largest]

        # Max faces limit
        if config.max_faces > 0:
            filtered = filtered[: config.max_faces]

        # Re-index face_id
        filtered = [
            FaceDetection(
                bbox_xyxy=d.bbox_xyxy,
                confidence=d.confidence,
                landmarks_5pt=d.landmarks_5pt,
                face_id=i,
                backend=d.backend,
            )
            for i, d in enumerate(filtered)
        ]
        return filtered
