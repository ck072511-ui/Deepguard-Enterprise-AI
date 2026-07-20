"""
DeepGuard — vision/face_extraction/detectors/retina_face.py

RetinaFace face detector backend via the ``insightface`` library.

RetinaFace is a state-of-the-art single-stage detector that jointly
predicts face score, bounding box, and 5-point landmarks. It outperforms
MTCNN on challenging faces (occlusion, extreme poses, small faces).

GPU support: pass ``ctx_id=0`` (GPU) or ``ctx_id=-1`` (CPU).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from vision.face_extraction.base import DetectionConfig, FaceDetection, IDetector

logger = logging.getLogger(__name__)

_INSIGHTFACE_AVAILABLE = False
try:
    import insightface  # type: ignore[import]
    from insightface.app import FaceAnalysis  # type: ignore[import]

    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None  # type: ignore[assignment,misc]


class RetinaFaceDetector(IDetector):
    """Face detector backed by InsightFace RetinaFace.

    Args:
        device:     'cpu', 'cuda', 'cuda:N', or 'auto'.
        config:     Default detection configuration.
        model_name: InsightFace app name (default: 'buffalo_sc' for speed).
        det_size:   Detection input resolution (width, height).
    """

    def __init__(
        self,
        device: str = "auto",
        config: DetectionConfig | None = None,
        model_name: str = "buffalo_sc",
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._config = config or DetectionConfig.default()
        self._model_name = model_name
        self._det_size = det_size
        self._device_str = device
        self._ctx_id = self._resolve_ctx_id(device)
        self._app: FaceAnalysis | None = None

        if _INSIGHTFACE_AVAILABLE:
            self._load_model()
        else:
            logger.warning("insightface not installed; RetinaFaceDetector is unavailable.")

    # ------------------------------------------------------------------
    # IDetector interface
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Return 'retinaface'."""
        return "retinaface"

    @property
    def device(self) -> str:
        """Return active device string."""
        return "cpu" if self._ctx_id < 0 else f"cuda:{self._ctx_id}"

    @property
    def is_available(self) -> bool:
        """Return True if insightface is installed and model loaded."""
        return _INSIGHTFACE_AVAILABLE and self._app is not None

    def detect(
        self,
        image: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[FaceDetection]:
        """Detect faces in a single RGB image.

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
        # InsightFace expects BGR
        bgr = image[:, :, ::-1].astype(np.uint8)
        try:
            faces = self._app.get(bgr)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("RetinaFaceDetector.detect failed: %s", exc)
            return []

        h, w = image.shape[:2]
        return self._parse(faces, w, h, cfg)

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        config: DetectionConfig | None = None,
    ) -> list[list[FaceDetection]]:
        """Detect faces in a batch (sequential; InsightFace has no batch API).

        Args:
            images: Sequence of RGB uint8 arrays.
            config: Optional config override.

        Returns:
            List of detection lists, one per image.
        """
        cfg = config or self._config
        return [self.detect(img, cfg) for img in images]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the InsightFace FaceAnalysis model.

        Logs a warning on failure instead of raising, so the selector
        can fall back to the next backend gracefully.
        """
        try:
            self._app = FaceAnalysis(
                name=self._model_name,
                allowed_modules=["detection"],
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self._ctx_id >= 0
                    else ["CPUExecutionProvider"]
                ),
            )
            self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
            logger.debug(
                "RetinaFaceDetector loaded model='%s' ctx_id=%d det_size=%s.",
                self._model_name,
                self._ctx_id,
                self._det_size,
            )
        except Exception as exc:
            logger.warning("Failed to load InsightFace model: %s", exc)
            self._app = None

    def _parse(
        self,
        faces: list,
        image_w: int,
        image_h: int,
        cfg: DetectionConfig,
    ) -> list[FaceDetection]:
        """Convert InsightFace face objects to FaceDetection instances.

        Args:
            faces:    List of InsightFace face objects.
            image_w:  Image width.
            image_h:  Image height.
            cfg:      Detection configuration.

        Returns:
            Filtered, sorted list of FaceDetection.
        """
        raw: list[FaceDetection] = []
        for j, face in enumerate(faces or []):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            conf = float(face.det_score)

            lm_arr: np.ndarray | None = None
            if hasattr(face, "kps") and face.kps is not None:
                lm_arr = np.array(face.kps, dtype=np.float32)  # shape (5, 2)

            raw.append(
                FaceDetection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=conf,
                    landmarks_5pt=lm_arr,
                    face_id=j,
                    backend=self.backend_name,
                )
            )

        return self._apply_config_filters(raw, cfg, image_w, image_h)

    @staticmethod
    def _resolve_ctx_id(device: str) -> int:
        """Map device string to InsightFace ctx_id (-1=CPU, >=0=GPU index).

        Args:
            device: Device string ('cpu', 'cuda', 'cuda:N', 'auto').

        Returns:
            ctx_id integer.
        """
        if device == "cpu":
            return -1
        if device in ("cuda", "auto"):
            try:
                import torch

                return 0 if torch.cuda.is_available() else -1
            except ImportError:
                return -1
        if device.startswith("cuda:"):
            return int(device.split(":")[1])
        return -1
