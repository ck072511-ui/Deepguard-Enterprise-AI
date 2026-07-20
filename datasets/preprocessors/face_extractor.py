"""
DeepGuard — datasets/preprocessors/face_extractor.py

Face detection and region extraction with multiple backends:
  - MTCNN (facenet-pytorch)    — high accuracy, GPU-accelerated
  - OpenCV FaceDetectorYN      — DNN-based, works with OpenCV 5+
  - OpenCV Haar Cascade        — legacy fallback for OpenCV <5

The extractor implements the IFaceExtractor interface and returns
cropped face numpy arrays ready for downstream model input.

Usage:
    >>> extractor = FaceExtractor(backend="mtcnn", device="cpu")
    >>> faces = extractor.extract_faces(image_rgb, margin=0.3)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.domain.entities.dataset_entity import FaceRegionEntity
from core.exceptions.dataset_exceptions import FaceExtractionError
from core.interfaces.dataset_interface import IFaceExtractor

logger = logging.getLogger(__name__)

# Detect OpenCV version for backend capability checks
_CV2_VERSION = tuple(int(x) for x in cv2.__version__.split(".")[:2])
_HAS_CASCADE_CLASSIFIER = hasattr(cv2, "CascadeClassifier")
_HAS_FACE_DETECTOR_YN = hasattr(cv2, "FaceDetectorYN")

# Path to OpenCV's bundled Haar Cascade XML (legacy, OpenCV < 5)
_HAAR_CASCADE_PATH = (
    Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
)


class FaceExtractor(IFaceExtractor):
    """Multi-backend face detection and extraction pipeline.

    Supports MTCNN (high accuracy) and OpenCV Haar Cascade (fast fallback).
    Automatically selects the best available backend when ``backend="auto"``.

    Args:
        backend:          Detection backend: 'mtcnn' | 'opencv_haar' | 'auto'.
        device:           Compute device for MTCNN: 'cpu' | 'cuda' | 'auto'.
        keep_largest_only: If True, returns only the largest detected face.
        fallback_to_full:  If True, returns the full image when no face found.
        min_face_size:    Minimum face size in pixels for MTCNN.
        image_size:       Output face crop size (square, pixels).
    """

    def __init__(
        self,
        backend: str = "auto",
        device: str = "auto",
        *,
        keep_largest_only: bool = False,
        fallback_to_full: bool = True,
        min_face_size: int = 40,
        image_size: int = 224,
    ) -> None:
        self._backend_name = backend
        self._keep_largest_only = keep_largest_only
        self._fallback_to_full = fallback_to_full
        self._min_face_size = min_face_size
        self._image_size = image_size
        self._device = self._resolve_device(device)

        self._mtcnn: Any = None
        self._haar_cascade: cv2.CascadeClassifier | None = None

        resolved = self._resolve_backend(backend)
        self._active_backend = resolved
        self._initialise_backend(resolved)

        logger.info(
            "FaceExtractor initialised | backend=%s device=%s keep_largest=%s",
            self._active_backend,
            self._device,
            keep_largest_only,
        )

    # ------------------------------------------------------------------
    # IFaceExtractor interface
    # ------------------------------------------------------------------

    def extract_faces(
        self,
        image: np.ndarray,
        margin: float = 0.3,
        min_confidence: float = 0.9,
    ) -> list[np.ndarray]:
        """Detect faces and return cropped face regions.

        Args:
            image:          RGB numpy array (H, W, 3) uint8.
            margin:         Fractional margin added around the face bbox.
            min_confidence: Minimum detection confidence threshold.

        Returns:
            List of cropped face RGB numpy arrays resized to image_size.
            Returns [full_image] if no face found and fallback_to_full=True.
            Returns [] if no face found and fallback_to_full=False.

        Raises:
            FaceExtractionError: On unexpected internal extraction failure.
        """
        if image is None or image.size == 0:
            raise FaceExtractionError("<unknown>", "Input image is None or empty.")

        try:
            regions = self.detect(image, min_confidence=min_confidence)
        except FaceExtractionError:
            raise
        except Exception as exc:
            raise FaceExtractionError(
                "<unknown>", f"Detection backend '{self._active_backend}' raised: {exc}"
            ) from exc

        if not regions:
            logger.debug("No face detected (backend=%s).", self._active_backend)
            if self._fallback_to_full:
                return [self._resize(image)]
            return []

        if self._keep_largest_only:
            regions = [max(regions, key=lambda r: r.area)]

        crops: list[np.ndarray] = []
        h, w = image.shape[:2]
        for region in regions:
            expanded = region.with_margin(margin, image_w=w, image_h=h)
            x1, y1, x2, y2 = expanded.to_xyxy()
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning("Face crop produced empty array; skipping.")
                continue
            crops.append(self._resize(crop))

        if not crops and self._fallback_to_full:
            return [self._resize(image)]

        return crops

    def detect(
        self,
        image: np.ndarray,
        min_confidence: float = 0.9,
    ) -> list[FaceRegionEntity]:
        """Detect face bounding boxes without cropping.

        Args:
            image:          RGB numpy array (H, W, 3) uint8.
            min_confidence: Minimum detection confidence to accept.

        Returns:
            List of FaceRegionEntity, sorted by confidence descending.
        """
        if self._active_backend == "mtcnn":
            return self._detect_mtcnn(image, min_confidence)
        return self._detect_haar(image)

    @property
    def backend_name(self) -> str:
        """Return the active detection backend name."""
        return self._active_backend

    # ------------------------------------------------------------------
    # MTCNN detection
    # ------------------------------------------------------------------

    def _detect_mtcnn(
        self, image: np.ndarray, min_confidence: float
    ) -> list[FaceRegionEntity]:
        """Run MTCNN detection on an RGB image.

        Args:
            image:          RGB numpy array.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of FaceRegionEntity objects.
        """
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        try:
            boxes, probs, landmarks = self._mtcnn.detect(pil_img, landmarks=True)
        except Exception as exc:
            logger.warning("MTCNN detection failed: %s. Returning empty.", exc)
            return []

        if boxes is None or probs is None:
            return []

        regions: list[FaceRegionEntity] = []
        for box, prob, lm in zip(boxes, probs, landmarks or [None] * len(boxes)):
            if prob is None or float(prob) < min_confidence:
                continue
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            landmark_dict: dict[str, tuple[int, int]] = {}
            if lm is not None:
                names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                for name, point in zip(names, lm):
                    landmark_dict[name] = (int(point[0]), int(point[1]))

            regions.append(
                FaceRegionEntity(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    confidence=float(prob),
                    landmarks=landmark_dict,
                )
            )

        regions.sort(key=lambda r: r.confidence, reverse=True)
        return regions

    # ------------------------------------------------------------------
    # OpenCV detection — FaceDetectorYN (OpenCV 5+) or Haar (OpenCV <5)
    # ------------------------------------------------------------------

    def _detect_haar(self, image: np.ndarray) -> list[FaceRegionEntity]:
        """Run face detection using the best available OpenCV backend.

        On OpenCV 5+, uses FaceDetectorYN (DNN-based).
        On older OpenCV, falls back to Haar Cascade.
        If neither is available, returns an empty list.

        Args:
            image: RGB numpy array (H, W, 3).

        Returns:
            List of FaceRegionEntity, sorted by confidence/area descending.
        """
        if _HAS_FACE_DETECTOR_YN and self._face_detector_yn is not None:
            return self._detect_opencv_dnn(image)

        if _HAS_CASCADE_CLASSIFIER and self._haar_cascade is not None:
            return self._detect_cascade(image)

        # No backend available — return empty list (fallback handles this)
        logger.debug("No OpenCV face detection backend available; returning empty.")
        return []

    def _detect_opencv_dnn(self, image: np.ndarray) -> list[FaceRegionEntity]:
        """Run FaceDetectorYN (OpenCV DNN) detection on an RGB image.

        Args:
            image: RGB numpy array.

        Returns:
            List of FaceRegionEntity sorted by score descending.
        """
        h, w = image.shape[:2]
        self._face_detector_yn.setInputSize((w, h))  # type: ignore[union-attr]
        try:
            _, faces = self._face_detector_yn.detect(image)  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("FaceDetectorYN failed: %s", exc)
            return []

        if faces is None or len(faces) == 0:
            return []

        regions: list[FaceRegionEntity] = []
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[14]) if len(face) > 14 else 1.0
            regions.append(
                FaceRegionEntity(
                    x=max(0, x),
                    y=max(0, y),
                    width=min(fw, w - x),
                    height=min(fh, h - y),
                    confidence=score,
                )
            )

        regions.sort(key=lambda r: r.confidence, reverse=True)
        return regions

    def _detect_cascade(self, image: np.ndarray) -> list[FaceRegionEntity]:
        """Run Haar Cascade detection (OpenCV < 5 only).

        Args:
            image: RGB numpy array.

        Returns:
            List of FaceRegionEntity with confidence=1.0.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = self._haar_cascade.detectMultiScale(  # type: ignore[union-attr]
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self._min_face_size, self._min_face_size),
        )

        if len(detections) == 0:
            return []

        regions = [
            FaceRegionEntity(x=int(x), y=int(y), width=int(w), height=int(h), confidence=1.0)
            for x, y, w, h in detections
        ]
        regions.sort(key=lambda r: r.area, reverse=True)
        return regions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize an image to (image_size, image_size) using bicubic interpolation.

        Args:
            image: RGB numpy array.

        Returns:
            Resized RGB numpy array.
        """
        return cv2.resize(
            image,
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_CUBIC,
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' to the best available torch device string.

        Args:
            device: 'auto' | 'cpu' | 'cuda' | 'mps'.

        Returns:
            Resolved device string.
        """
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _resolve_backend(self, backend: str) -> str:
        """Resolve 'auto' backend by testing available backends.

        Priority: mtcnn → opencv_dnn → opencv_haar.

        Args:
            backend: Requested backend name.

        Returns:
            Resolved backend name.
        """
        if backend != "auto":
            return backend
        try:
            import facenet_pytorch  # noqa: F401
            return "mtcnn"
        except ImportError:
            pass

        if _HAS_FACE_DETECTOR_YN:
            logger.info("Using OpenCV FaceDetectorYN backend.")
            return "opencv_dnn"

        if _HAS_CASCADE_CLASSIFIER:
            logger.info("Using OpenCV Haar Cascade backend.")
            return "opencv_haar"

        logger.warning("No face detection backend available; using null fallback.")
        return "opencv_haar"  # Will produce empty list gracefully

    def _initialise_backend(self, backend: str) -> None:
        """Initialise the chosen detection backend.

        Args:
            backend: Resolved backend name.

        Raises:
            FaceExtractionError: If the backend cannot be initialised.
        """
        self._face_detector_yn: Any = None
        self._haar_cascade: Any = None

        if backend == "mtcnn":
            try:
                from facenet_pytorch import MTCNN

                self._mtcnn = MTCNN(
                    min_face_size=self._min_face_size,
                    thresholds=[0.6, 0.7, 0.7],
                    keep_all=True,
                    device=self._device,
                    post_process=False,
                )
                logger.debug("MTCNN initialised on device=%s.", self._device)
            except Exception as exc:
                raise FaceExtractionError(
                    "<init>", f"Failed to initialise MTCNN: {exc}"
                ) from exc

        elif backend in ("opencv_dnn", "opencv_haar"):
            if _HAS_FACE_DETECTOR_YN:
                # OpenCV 5+ — use DNN-based detector (no model file needed for basic)
                try:
                    # Try lightweight model bundled with OpenCV if available
                    model_path = Path(cv2.__file__).parent / "data" / "face_detection_yunet_2023mar.onnx"
                    if model_path.exists():
                        self._face_detector_yn = cv2.FaceDetectorYN.create(
                            str(model_path),
                            "",
                            (320, 320),
                            score_threshold=0.6,
                            nms_threshold=0.3,
                        )
                        logger.debug("OpenCV FaceDetectorYN initialised from '%s'.", model_path)
                    else:
                        logger.debug(
                            "YuNet model not found; FaceDetectorYN will be None (fallback to empty)."
                        )
                except Exception as exc:
                    logger.debug("FaceDetectorYN init failed: %s", exc)

            elif _HAS_CASCADE_CLASSIFIER and _HAAR_CASCADE_PATH.exists():
                # OpenCV < 5 — use legacy Haar Cascade
                self._haar_cascade = cv2.CascadeClassifier(str(_HAAR_CASCADE_PATH))  # type: ignore[attr-defined]
                if self._haar_cascade.empty():
                    logger.warning("Haar cascade failed to load; face detection will return empty.")
                    self._haar_cascade = None
                else:
                    logger.debug("OpenCV Haar Cascade initialised.")
            else:
                logger.warning(
                    "No OpenCV face detection model available. "
                    "FaceExtractor will use fallback (full frame)."
                )

        else:
            raise FaceExtractionError(
                "<init>",
                f"Unknown backend '{backend}'. Valid: 'mtcnn', 'opencv_dnn', 'opencv_haar', 'auto'.",
            )


class NullFaceExtractor(IFaceExtractor):
    """Pass-through extractor that returns the full image without detection.

    Useful when the dataset already contains pre-cropped face images
    and no detection step is needed.

    Args:
        image_size: Output resize target.
    """

    def __init__(self, image_size: int = 224) -> None:
        self._image_size = image_size

    def extract_faces(
        self,
        image: np.ndarray,
        margin: float = 0.3,
        min_confidence: float = 0.9,
    ) -> list[np.ndarray]:
        """Return the full image resized — no face detection performed.

        Args:
            image:          Input RGB numpy array.
            margin:         Unused (interface compatibility).
            min_confidence: Unused (interface compatibility).

        Returns:
            Single-element list containing the resized image.
        """
        resized = cv2.resize(
            image,
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        return [resized]

    def detect(
        self,
        image: np.ndarray,
        min_confidence: float = 0.9,
    ) -> list[FaceRegionEntity]:
        """Return an empty list (no detection in pass-through mode).

        Args:
            image:          Unused.
            min_confidence: Unused.

        Returns:
            Empty list.
        """
        return []

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "null"
