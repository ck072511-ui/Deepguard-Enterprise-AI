"""
DeepGuard — vision/face_extraction/detectors/mtcnn_detector.py

MTCNN face detector backend wrapping ``facenet_pytorch.MTCNN``.

MTCNN (Multi-task Cascaded Convolutional Networks) produces:
  - High-accuracy bounding boxes
  - 5-point facial landmarks (left eye, right eye, nose, left/right mouth)
  - Per-box confidence scores

Supports native batched inference and GPU acceleration.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from vision.face_extraction.base import DetectionConfig, FaceDetection, IDetector

logger = logging.getLogger(__name__)

_MTCNN_AVAILABLE = False
try:
    from facenet_pytorch import MTCNN as _MTCNN  # type: ignore[import]

    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN = None  # type: ignore[assignment,misc]


# Standard MTCNN landmark indices:
# 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth
_LANDMARK_ORDER = [0, 1, 2, 3, 4]


class MTCNNDetector(IDetector):
    """Face detector backed by facenet-pytorch MTCNN.

    Args:
        device:     Compute device: 'cpu', 'cuda', 'cuda:N', or 'auto'.
        config:     Default detection configuration.
        min_face_size: Minimum face size passed to MTCNN constructor.
        thresholds: MTCNN P-net / R-net / O-net confidence thresholds.
    """

    def __init__(
        self,
        device: str = "auto",
        config: DetectionConfig | None = None,
        min_face_size: int = 20,
        thresholds: tuple[float, float, float] = (0.6, 0.7, 0.7),
    ) -> None:
        self._config = config or DetectionConfig.default()
        self._min_face_size = min_face_size
        self._thresholds = list(thresholds)
        self._device_str = self._resolve_device(device)
        self._model: Any = None

        if _MTCNN_AVAILABLE:
            self._model = _MTCNN(
                min_face_size=min_face_size,
                thresholds=self._thresholds,
                keep_all=True,
                device=self._device_str,
                post_process=False,
                select_largest=False,
            )
            logger.debug("MTCNNDetector initialised on device='%s'.", self._device_str)
        else:
            logger.warning("facenet-pytorch not installed; MTCNNDetector is unavailable.")

    # ------------------------------------------------------------------
    # IDetector interface
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Return 'mtcnn'."""
        return "mtcnn"

    @property
    def device(self) -> str:
        """Return active compute device string."""
        return self._device_str

    @property
    def is_available(self) -> bool:
        """Return True if facenet-pytorch is installed."""
        return _MTCNN_AVAILABLE

    def detect(
        self,
        image: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[FaceDetection]:
        """Detect faces in a single RGB image using MTCNN.

        Args:
            image:  RGB uint8 numpy array (H, W, 3).
            config: Optional config override.

        Returns:
            List of FaceDetection sorted by confidence descending.

        Raises:
            ValueError:          If image is invalid.
            RuntimeError:        If MTCNN is not available.
        """
        self._validate_image(image)
        if not self._is_ready():
            return []

        cfg = config or self._config
        results = self.detect_batch([image], cfg)
        return results[0]

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        config: DetectionConfig | None = None,
    ) -> list[list[FaceDetection]]:
        """Detect faces in a batch of RGB images.

        Leverages MTCNN's native batch support for GPU efficiency.

        Args:
            images: Sequence of RGB uint8 numpy arrays.
            config: Optional config override.

        Returns:
            List of detection lists, one per input image.
        """
        if not self._is_ready():
            return [[] for _ in images]

        cfg = config or self._config
        import torch
        from PIL import Image as PILImage

        pil_images = [PILImage.fromarray(img) for img in images]
        try:
            boxes_batch, probs_batch, landmarks_batch = self._model.detect(
                pil_images, landmarks=True
            )
        except Exception as exc:
            logger.error("MTCNNDetector batch detection failed: %s", exc)
            return [[] for _ in images]

        all_results: list[list[FaceDetection]] = []
        for i, (img, boxes, probs, landmarks) in enumerate(
            zip(images, boxes_batch or [], probs_batch or [], landmarks_batch or [])
        ):
            h, w = img.shape[:2]
            detections = self._parse_single(boxes, probs, landmarks, w, h, cfg)
            all_results.append(detections)

        # Pad for any None entries from MTCNN when no face found
        while len(all_results) < len(images):
            all_results.append([])

        return all_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_single(
        self,
        boxes: Any,
        probs: Any,
        landmarks: Any,
        image_w: int,
        image_h: int,
        cfg: DetectionConfig,
    ) -> list[FaceDetection]:
        """Parse MTCNN output for a single image into FaceDetection objects.

        Args:
            boxes:    (N, 4) float array or None.
            probs:    (N,) float array or None.
            landmarks: (N, 5, 2) float array or None.
            image_w:  Image width.
            image_h:  Image height.
            cfg:      Detection configuration.

        Returns:
            Filtered list of FaceDetection objects.
        """
        if boxes is None or probs is None:
            return []

        raw: list[FaceDetection] = []
        for j, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            lm_arr: np.ndarray | None = None
            if landmarks is not None and len(landmarks) > j:
                lm = landmarks[j]  # shape (5, 2)
                lm_arr = np.array(lm, dtype=np.float32)
            raw.append(
                FaceDetection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(prob),
                    landmarks_5pt=lm_arr,
                    face_id=j,
                    backend=self.backend_name,
                )
            )

        return self._apply_config_filters(raw, cfg, image_w, image_h)

    def _is_ready(self) -> bool:
        """Return True if the MTCNN model is loaded."""
        return self._model is not None and _MTCNN_AVAILABLE

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to 'cuda' if available, else 'cpu'.

        Args:
            device: Requested device string.

        Returns:
            Resolved device string.
        """
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
