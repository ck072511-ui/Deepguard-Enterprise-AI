"""
DeepGuard — vision/face_extraction/batch_processor.py

Batch image and video frame face extraction.

Provides:
  - ``BatchProcessor.process_images``     — parallel I/O + sequential detection
  - ``BatchProcessor.extract_from_video`` — streaming generator for video files

Design goals:
  - Memory-efficient: processes images lazily, never loads all at once
  - Parallelised I/O: uses ThreadPoolExecutor for disk reads
  - Configurable: detection config, quality filter, alignment, output size
  - Idempotent: skips missing files rather than crashing
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator, Sequence

import cv2
import numpy as np

from vision.face_extraction.alignment import FaceAligner
from vision.face_extraction.base import DetectionConfig, FaceDetection, IDetector
from vision.face_extraction.quality_filter import QualityFilter, QualityThresholds

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Result of extracting faces from a single video frame.

    Attributes:
        frame_index:  Frame number in the source video (0-based).
        timestamp_ms: Frame timestamp in milliseconds.
        source_path:  Path to the source video.
        detections:   All face detections in this frame.
        faces:        Cropped/aligned face arrays (one per detection).
        quality_ok:   List of booleans — True if crop passed quality filter.
    """

    frame_index: int
    timestamp_ms: float
    source_path: Path
    detections: list[FaceDetection] = field(default_factory=list)
    faces: list[np.ndarray] = field(default_factory=list)
    quality_ok: list[bool] = field(default_factory=list)

    @property
    def n_faces(self) -> int:
        """Return the number of detected faces in this frame."""
        return len(self.detections)

    @property
    def n_quality_faces(self) -> int:
        """Return the number of faces that passed quality filtering."""
        return sum(self.quality_ok)


@dataclass
class ImageResult:
    """Result of processing a single image file.

    Attributes:
        source_path: Path to the source image.
        detections:  Face detections.
        faces:       Cropped/aligned face arrays.
        quality_ok:  Per-face quality flags.
        error:       Error message if loading/detection failed, else ''.
    """

    source_path: Path
    detections: list[FaceDetection] = field(default_factory=list)
    faces: list[np.ndarray] = field(default_factory=list)
    quality_ok: list[bool] = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> bool:
        """Return True if processing succeeded (no error)."""
        return self.error == ""


class BatchProcessor:
    """Processes batches of images or video files for face extraction.

    Args:
        detector:       Configured IDetector backend instance.
        config:         Detection configuration.
        aligner:        FaceAligner for alignment (None = crop without alignment).
        quality_filter: QualityFilter instance (None = skip quality check).
        output_size:    Output face crop size in pixels (square).
        num_workers:    Number of parallel I/O worker threads.
        fallback_to_full_frame: If True and no face detected, return the whole
                                resized frame as a synthetic 'face'.
    """

    def __init__(
        self,
        detector: IDetector,
        config: DetectionConfig | None = None,
        aligner: FaceAligner | None = None,
        quality_filter: QualityFilter | None = None,
        output_size: int = 224,
        num_workers: int = 4,
        fallback_to_full_frame: bool = False,
    ) -> None:
        self._detector = detector
        self._config = config or DetectionConfig.default()
        self._aligner = aligner or FaceAligner(output_size=output_size)
        self._quality_filter = quality_filter or QualityFilter()
        self._output_size = output_size
        self._num_workers = max(1, num_workers)
        self._fallback = fallback_to_full_frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_images(
        self,
        image_paths: Sequence[Path | str],
        *,
        run_quality_filter: bool = True,
    ) -> list[ImageResult]:
        """Detect and extract faces from a collection of image files.

        Loads images in parallel (I/O bound) then runs detection
        sequentially on the main thread (compute bound).

        Args:
            image_paths:       Paths to image files (JPEG, PNG, BMP, …).
            run_quality_filter: If True, runs quality checks per face.

        Returns:
            List of ImageResult, one per input path (same order).
        """
        paths = [Path(p) for p in image_paths]
        loaded: dict[int, np.ndarray | None] = {}
        errors: dict[int, str] = {}

        logger.info(
            "BatchProcessor: loading %d images with %d workers.",
            len(paths),
            self._num_workers,
        )

        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            future_to_idx = {
                pool.submit(self._load_image_rgb, p): i
                for i, p in enumerate(paths)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                img, err = future.result()
                loaded[idx] = img
                if err:
                    errors[idx] = err

        results: list[ImageResult] = []
        for i, path in enumerate(paths):
            img = loaded.get(i)
            err = errors.get(i, "")

            if img is None or err:
                results.append(
                    ImageResult(source_path=path, error=err or "LOAD_FAILED")
                )
                continue

            detections = self._detector.detect(img, self._config)

            if not detections and self._fallback:
                fallback_face = cv2.resize(
                    img, (self._output_size, self._output_size), interpolation=cv2.INTER_CUBIC
                )
                results.append(
                    ImageResult(
                        source_path=path,
                        faces=[fallback_face],
                        quality_ok=[True],
                    )
                )
                continue

            faces, quality_ok = self._extract_crops(img, detections, run_quality_filter)
            results.append(
                ImageResult(
                    source_path=path,
                    detections=detections,
                    faces=faces,
                    quality_ok=quality_ok,
                )
            )

        logger.info(
            "BatchProcessor: processed %d images, %d with errors.",
            len(paths),
            len(errors),
        )
        return results

    def extract_from_video(
        self,
        video_path: Path | str,
        *,
        target_fps: float | None = None,
        max_frames: int | None = None,
        run_quality_filter: bool = True,
        skip_frames_without_face: bool = False,
    ) -> Generator[FrameResult, None, None]:
        """Stream face extractions from a video file frame-by-frame.

        Yields one FrameResult per sampled frame, making it memory-efficient
        even for hour-long videos.

        Args:
            video_path:                Path to the video file.
            target_fps:                Sampling rate in frames per second.
                                       None = use native video FPS (all frames).
            max_frames:                Stop after this many sampled frames (None = all).
            run_quality_filter:        Run quality check on each face crop.
            skip_frames_without_face:  If True, only yield frames with ≥1 detection.

        Yields:
            FrameResult for each sampled frame.

        Raises:
            FileNotFoundError: If video_path does not exist.
            RuntimeError:      If the video cannot be opened.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: '{video_path}'")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: '{video_path}'")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_fps = target_fps if target_fps else native_fps
        step = max(1, round(native_fps / sample_fps))

        logger.info(
            "BatchProcessor: video='%s' native_fps=%.1f sample_fps=%.1f step=%d total=%d",
            video_path.name,
            native_fps,
            sample_fps,
            step,
            total_frames,
        )

        frame_idx = 0
        yielded = 0

        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                if frame_idx % step == 0:
                    ts_ms = frame_idx / max(native_fps, 1e-6) * 1000.0
                    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    detections = self._detector.detect(frame_rgb, self._config)

                    if skip_frames_without_face and not detections:
                        frame_idx += 1
                        continue

                    faces, quality_ok = self._extract_crops(
                        frame_rgb, detections, run_quality_filter
                    )

                    yield FrameResult(
                        frame_index=frame_idx,
                        timestamp_ms=ts_ms,
                        source_path=video_path,
                        detections=detections,
                        faces=faces,
                        quality_ok=quality_ok,
                    )
                    yielded += 1

                    if max_frames is not None and yielded >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()

        logger.info(
            "BatchProcessor: finished video '%s', yielded %d frames.",
            video_path.name,
            yielded,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_crops(
        self,
        image: np.ndarray,
        detections: list[FaceDetection],
        run_quality_filter: bool,
    ) -> tuple[list[np.ndarray], list[bool]]:
        """Crop, align, and optionally quality-filter detected faces.

        Args:
            image:             Source RGB image.
            detections:        Face detections from the detector.
            run_quality_filter: Whether to run quality checks.

        Returns:
            Tuple of (face_crops, quality_flags).
        """
        faces: list[np.ndarray] = []
        quality_ok: list[bool] = []

        for det in detections:
            if det.has_landmarks:
                crop = self._aligner.align(image, det.landmarks_5pt)  # type: ignore[arg-type]
            else:
                crop = self._aligner.crop_without_alignment(image, det.bbox_xyxy)

            if run_quality_filter:
                qr = self._quality_filter.evaluate(crop)
                quality_ok.append(qr.passed)
            else:
                quality_ok.append(True)

            faces.append(crop)

        return faces, quality_ok

    @staticmethod
    def _load_image_rgb(path: Path) -> tuple[np.ndarray | None, str]:
        """Load an image file and convert to RGB.

        Args:
            path: File path to load.

        Returns:
            Tuple of (RGB array or None, error string or '').
        """
        if not path.exists():
            return None, f"FILE_NOT_FOUND:{path}"
        try:
            bgr = cv2.imread(str(path))
            if bgr is None:
                return None, f"DECODE_FAILED:{path}"
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ""
        except Exception as exc:
            return None, f"LOAD_ERROR:{exc}"
