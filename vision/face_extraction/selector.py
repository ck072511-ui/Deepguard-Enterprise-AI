"""
DeepGuard — vision/face_extraction/selector.py

Automatic backend selection via capability probing.

The ``AutoSelector`` probes which detector libraries are installed and
available, then returns the best detector instance based on caller
preferences (accuracy vs. speed, GPU vs. CPU).

Priority order (accuracy):
  1. RetinaFace (insightface)   — SOTA accuracy, GPU
  2. MTCNN (facenet-pytorch)    — high accuracy, GPU
  3. MediaPipe                  — good accuracy, CPU-only, fast
  4. OpenCV FaceDetectorYN      — decent accuracy, CPU, no heavy deps

Priority order (speed):
  1. MediaPipe                  — fastest on CPU
  2. OpenCV FaceDetectorYN      — fast, no deps
  3. MTCNN                      — moderate speed
  4. RetinaFace                 — slowest (highest accuracy)
"""

from __future__ import annotations

import logging

from vision.face_extraction.base import DetectionConfig, IDetector

logger = logging.getLogger(__name__)

# Lazy imports used only during probing
_BACKEND_REGISTRY: dict[str, str] = {
    "retinaface": "vision.face_extraction.detectors.retina_face.RetinaFaceDetector",
    "mtcnn": "vision.face_extraction.detectors.mtcnn_detector.MTCNNDetector",
    "mediapipe": "vision.face_extraction.detectors.mediapipe_detector.MediaPipeDetector",
    "opencv": "vision.face_extraction.detectors.opencv_detector.OpenCVDetector",
}

_ACCURACY_ORDER = ["retinaface", "mtcnn", "mediapipe", "opencv"]
_SPEED_ORDER = ["mediapipe", "opencv", "mtcnn", "retinaface"]


def _import_detector(backend: str) -> type[IDetector] | None:
    """Dynamically import a detector class by backend name.

    Args:
        backend: Backend name (key in _BACKEND_REGISTRY).

    Returns:
        Detector class or None if import fails.
    """
    module_path = _BACKEND_REGISTRY.get(backend)
    if module_path is None:
        return None
    try:
        parts = module_path.rsplit(".", 1)
        import importlib

        mod = importlib.import_module(parts[0])
        return getattr(mod, parts[1])
    except (ImportError, AttributeError):
        return None


def probe_available_backends() -> list[str]:
    """Probe which detector backends are available on this system.

    Returns:
        List of available backend names, ordered by accuracy priority.
    """
    available: list[str] = []
    for name in _ACCURACY_ORDER:
        cls = _import_detector(name)
        if cls is None:
            continue
        try:
            instance = cls()
            if instance.is_available:
                available.append(name)
                logger.debug("Backend probe: '%s' available.", name)
            else:
                logger.debug("Backend probe: '%s' imported but not available.", name)
        except Exception as exc:
            logger.debug("Backend probe: '%s' failed to instantiate: %s.", name, exc)
    return available


class AutoSelector:
    """Selects and instantiates the best available face detector.

    Args:
        prefer_gpu:      Prefer GPU-capable backends (retinaface, mtcnn).
        prefer_accuracy: Sort by accuracy rather than speed.
        config:          Detection configuration to pass to the detector.
        device:          Compute device hint ('auto', 'cpu', 'cuda', etc.).
    """

    def __init__(
        self,
        prefer_gpu: bool = True,
        prefer_accuracy: bool = True,
        config: DetectionConfig | None = None,
        device: str = "auto",
    ) -> None:
        self._prefer_gpu = prefer_gpu
        self._prefer_accuracy = prefer_accuracy
        self._config = config or DetectionConfig.default()
        self._device = device

    def select(self) -> IDetector:
        """Probe backends and return the best available detector.

        Returns:
            Instantiated IDetector from the highest-priority available backend.

        Raises:
            RuntimeError: If no backend is available on this system.
        """
        priority = _ACCURACY_ORDER if self._prefer_accuracy else _SPEED_ORDER
        logger.info(
            "AutoSelector probing backends (prefer_accuracy=%s, prefer_gpu=%s).",
            self._prefer_accuracy,
            self._prefer_gpu,
        )

        for name in priority:
            cls = _import_detector(name)
            if cls is None:
                logger.debug("AutoSelector: '%s' not importable, skipping.", name)
                continue
            try:
                detector: IDetector = cls(device=self._device, config=self._config)
                if detector.is_available:
                    logger.info(
                        "AutoSelector selected backend='%s' device='%s'.",
                        detector.backend_name,
                        detector.device,
                    )
                    return detector
                logger.debug(
                    "AutoSelector: '%s' not available (missing deps).", name
                )
            except Exception as exc:
                logger.debug(
                    "AutoSelector: '%s' raised during init: %s.", name, exc
                )

        raise RuntimeError(
            "No face detection backend is available. "
            "Install at least one of: insightface, facenet-pytorch, mediapipe, opencv-python."
        )

    @classmethod
    def create_detector(
        cls,
        backend: str = "auto",
        device: str = "auto",
        config: DetectionConfig | None = None,
        prefer_accuracy: bool = True,
    ) -> IDetector:
        """Factory shortcut: create a named or auto-selected detector.

        Args:
            backend:        Backend name or 'auto' for automatic selection.
            device:         Compute device hint.
            config:         Detection configuration.
            prefer_accuracy: When backend='auto', whether to prioritise accuracy.

        Returns:
            Instantiated IDetector.

        Raises:
            ValueError:  If a named backend is not recognised.
            RuntimeError: If 'auto' and no backend is available.
        """
        if backend == "auto":
            return cls(
                prefer_accuracy=prefer_accuracy,
                config=config,
                device=device,
            ).select()

        detector_cls = _import_detector(backend)
        if detector_cls is None:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Valid names: {list(_BACKEND_REGISTRY.keys())} or 'auto'."
            )
        detector = detector_cls(device=device, config=config)  # type: ignore[call-arg]
        if not detector.is_available:
            raise RuntimeError(
                f"Backend '{backend}' is not available on this system. "
                "Check that the required library is installed."
            )
        return detector
