"""
DeepGuard — vision/face_extraction/detectors/mediapipe_detector.py

MediaPipe face detection backend.

MediaPipe's Face Detection model provides lightweight, mobile-optimised
detection with 6 keypoints. We map the first 5 to our canonical landmark
format (left-eye, right-eye, nose, left-mouth, right-mouth).

Key characteristics:
  - CPU-only (MediaPipe does not expose GPU in Python API)
  - Very fast on small/medium images
  - Relative coordinates — must be scaled to pixel space
  - Short-range (< 2m) and full-range models available
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from vision.face_extraction.base import DetectionConfig, FaceDetection, IDetector

logger = logging.getLogger(__name__)

_MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp  # type: ignore[import]

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None  # type: ignore[assignment]

# MediaPipe keypoint indices → canonical landmark order
# MediaPipe keypoints: 0=right_eye, 1=left_eye, 2=nose, 3=mouth, 4=right_ear, 5=left_ear
# Our order:           left_eye, right_eye, nose, left_mouth, right_mouth
_MP_TO_CANONICAL = [1, 0, 2, 3, 3]  # approximate — mouth corners not split


class MediaPipeDetector(IDetector):
    """Face detector backed by MediaPipe Face Detection.

    Args:
        device:        MediaPipe is CPU-only; this parameter is accepted but ignored.
        config:        Default detection configuration.
        model_selection: 0 = short-range (within 2m), 1 = full-range.
    """

    def __init__(
        self,
        device: str = "cpu",
        config: DetectionConfig | None = None,
        model_selection: int = 1,
    ) -> None:
        self._config = config or DetectionConfig.default()
        self._model_selection = model_selection
        self._detector = None

        if _MEDIAPIPE_AVAILABLE:
            self._detector = (
                mp.solutions.face_detection.FaceDetection(  # type: ignore[union-attr]
                    model_selection=model_selection,
                    min_detection_confidence=0.5,
                )
            )
            logger.debug(
                "MediaPipeDetector initialised (model_selection=%d).", model_selection
            )
        else:
            logger.warning("mediapipe not installed; MediaPipeDetector is unavailable.")

    # ------------------------------------------------------------------
    # IDetector interface
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Return 'mediapipe'."""
        return "mediapipe"

    @property
    def device(self) -> str:
        """Return 'cpu' (MediaPipe Python API is CPU-only)."""
        return "cpu"

    @property
    def is_available(self) -> bool:
        """Return True if mediapipe is installed and model loaded."""
        return _MEDIAPIPE_AVAILABLE and self._detector is not None

    def detect(
        self,
        image: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[FaceDetection]:
        """Detect faces in a single RGB image using MediaPipe.

        Args:
            image:  RGB uint8 array (H, W, 3).
            config: Optional config override.

        Returns:
            List of FaceDetection sorted by confidence descending.

        Raises:
            ValueError: If image is invalid.
        """
        self._validate_image(image)
        if not self.is_available:
            return []

        cfg = config or self._config
        h, w = image.shape[:2]

        try:
            result = self._detector.process(image)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("MediaPipeDetector.detect failed: %s", exc)
            return []

        if not result.detections:
            return []

        raw: list[FaceDetection] = []
        for j, detection in enumerate(result.detections):
            box = detection.location_data.relative_bounding_box
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))
            conf = float(detection.score[0]) if detection.score else 0.5

            # Extract keypoints → (5, 2) landmark array
            lm_arr = self._extract_landmarks(detection, w, h)

            raw.append(
                FaceDetection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=conf,
                    landmarks_5pt=lm_arr,
                    face_id=j,
                    backend=self.backend_name,
                )
            )

        return self._apply_config_filters(raw, cfg, w, h)

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        config: DetectionConfig | None = None,
    ) -> list[list[FaceDetection]]:
        """Detect faces in a batch (sequential; MediaPipe has no batch API).

        Args:
            images: Sequence of RGB uint8 arrays.
            config: Optional config override.

        Returns:
            List of detection lists, one per image.
        """
        cfg = config or self._config
        return [self.detect(img, cfg) for img in images]

    def __del__(self) -> None:
        """Release MediaPipe resources on garbage collection."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_landmarks(detection: object, w: int, h: int) -> np.ndarray | None:
        """Extract 5 canonical landmarks from a MediaPipe detection.

        MediaPipe provides 6 keypoints; we take the first 5 and reorder
        them into left-eye, right-eye, nose, left-mouth, right-mouth.

        Args:
            detection: MediaPipe detection proto.
            w:         Image width for scaling.
            h:         Image height for scaling.

        Returns:
            (5, 2) float32 array or None if keypoints are missing.
        """
        try:
            kps = detection.location_data.relative_keypoints  # type: ignore[attr-defined]
            if not kps or len(kps) < 5:
                return None

            # MediaPipe ordering: right_eye(0), left_eye(1), nose(2), mouth(3)
            # We map to: left_eye, right_eye, nose, left_mouth, right_mouth
            canonical_idx = _MP_TO_CANONICAL
            pts = np.array(
                [[kps[i].x * w, kps[i].y * h] for i in canonical_idx],
                dtype=np.float32,
            )
            return pts  # (5, 2)
        except Exception:
            return None
