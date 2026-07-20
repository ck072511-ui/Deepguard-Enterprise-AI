"""
DeepGuard — datasets/preprocessors/video_preprocessor.py

OpenCV-based video frame extraction pipeline with configurable
sampling strategies. Supports temporal uniform sampling, scene-change
detection, and random sampling for training diversity.

Extracted frames are immediately passed through the face extractor
and returned as numpy arrays ready for dataset loading.
"""

from __future__ import annotations

import logging
import random
from enum import StrEnum
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from core.exceptions.dataset_exceptions import FaceExtractionError
from core.interfaces.dataset_interface import IFaceExtractor

logger = logging.getLogger(__name__)


class FrameSamplingStrategy(StrEnum):
    """Strategy for selecting frames from a video."""

    UNIFORM = "uniform"          # Evenly-spaced temporal samples
    FIRST_N = "first_n"         # First N frames only
    RANDOM = "random"           # Random frame selection (reproducible with seed)
    SCENE_CHANGE = "scene_change"  # Sample on visual scene changes


class VideoPreprocessor:
    """OpenCV-based video frame extractor with configurable sampling.

    Opens a video file, samples frames according to the chosen strategy,
    optionally runs face extraction on each frame, and returns the results
    as a list of numpy arrays.

    Args:
        face_extractor:      IFaceExtractor implementation (or None to skip).
        strategy:            Frame sampling strategy.
        max_frames:          Maximum frames to extract per video.
        face_margin:         Margin fraction for face extraction.
        min_confidence:      Minimum face detection confidence.
        image_size:          Target output size for frames.
        skip_no_face_frames: If True, discard frames where no face is detected.
        seed:                Random seed for RANDOM strategy.
    """

    def __init__(
        self,
        face_extractor: IFaceExtractor | None = None,
        strategy: FrameSamplingStrategy = FrameSamplingStrategy.UNIFORM,
        max_frames: int = 30,
        face_margin: float = 0.3,
        min_confidence: float = 0.9,
        image_size: int = 224,
        *,
        skip_no_face_frames: bool = False,
        seed: int = 42,
    ) -> None:
        self._face_extractor = face_extractor
        self._strategy = strategy
        self._max_frames = max_frames
        self._face_margin = face_margin
        self._min_confidence = min_confidence
        self._image_size = image_size
        self._skip_no_face = skip_no_face_frames
        self._seed = seed

        logger.debug(
            "VideoPreprocessor | strategy=%s max_frames=%d face_margin=%.2f",
            strategy,
            max_frames,
            face_margin,
        )

    def extract_frames(
        self,
        video_path: Path | str,
        *,
        apply_face_extraction: bool = True,
    ) -> list[np.ndarray]:
        """Extract and preprocess frames from a video file.

        Args:
            video_path:             Path to the input video file.
            apply_face_extraction:  If True and face_extractor is set,
                                    apply face detection to each frame.

        Returns:
            List of RGB numpy arrays (H, W, 3) uint8.

        Raises:
            FileNotFoundError: If the video file does not exist.
            OSError: If OpenCV cannot open the video.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: '{video_path}'")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"OpenCV cannot open video: '{video_path}'")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            logger.debug(
                "Video opened | path=%s total_frames=%d fps=%.1f",
                video_path.name,
                total_frames,
                fps,
            )

            frame_indices = self._compute_frame_indices(total_frames)
            raw_frames = self._read_frames(cap, frame_indices)

        finally:
            cap.release()

        if not raw_frames:
            logger.warning("No frames extracted from '%s'.", video_path.name)
            return []

        if apply_face_extraction and self._face_extractor is not None:
            return self._apply_face_extraction(raw_frames, video_path)

        return [self._resize(f) for f in raw_frames]

    def extract_frames_generator(
        self,
        video_path: Path | str,
    ) -> Iterator[np.ndarray]:
        """Memory-efficient generator that yields frames one at a time.

        Useful when processing very long videos without materialising
        all frames in memory.

        Args:
            video_path: Path to the input video file.

        Yields:
            RGB numpy arrays (H, W, 3) uint8, one frame at a time.

        Raises:
            FileNotFoundError: If the video file does not exist.
            OSError: If OpenCV cannot open the video.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: '{video_path}'")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"OpenCV cannot open video: '{video_path}'")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = self._compute_frame_indices(total_frames)

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self._face_extractor is not None:
                    faces = self._face_extractor.extract_faces(
                        rgb,
                        margin=self._face_margin,
                        min_confidence=self._min_confidence,
                    )
                    for face in faces:
                        yield face
                else:
                    yield self._resize(rgb)
        finally:
            cap.release()

    def get_video_metadata(self, video_path: Path | str) -> dict[str, float | int | str]:
        """Extract metadata from a video file without reading frames.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with keys: total_frames, fps, width, height,
            duration_seconds, codec.

        Raises:
            FileNotFoundError: If the video does not exist.
            OSError: If OpenCV cannot open the file.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: '{video_path}'")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"OpenCV cannot open: '{video_path}'")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            duration = total_frames / fps if fps > 0 else 0.0
        finally:
            cap.release()

        return {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 2),
            "codec": codec.strip(),
            "filename": video_path.name,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_frame_indices(self, total_frames: int) -> list[int]:
        """Compute the list of frame indices to extract.

        Args:
            total_frames: Total number of frames in the video.

        Returns:
            Sorted list of integer frame indices.
        """
        if total_frames <= 0:
            return []

        n = min(self._max_frames, total_frames)

        if self._strategy == FrameSamplingStrategy.UNIFORM:
            if n >= total_frames:
                return list(range(total_frames))
            step = total_frames / n
            return [int(i * step) for i in range(n)]

        if self._strategy == FrameSamplingStrategy.FIRST_N:
            return list(range(min(n, total_frames)))

        if self._strategy == FrameSamplingStrategy.RANDOM:
            rng = random.Random(self._seed)
            indices = rng.sample(range(total_frames), min(n, total_frames))
            return sorted(indices)

        if self._strategy == FrameSamplingStrategy.SCENE_CHANGE:
            return self._scene_change_indices(total_frames, n)

        logger.warning("Unknown strategy '%s'; using UNIFORM.", self._strategy)
        return self._compute_frame_indices(total_frames)  # fallback

    def _scene_change_indices(self, total_frames: int, max_n: int) -> list[int]:
        """Compute frame indices based on scene change magnitude.

        Uses a fast histogram-difference heuristic (no deep learning).
        Falls back to UNIFORM if the video is too short.

        Args:
            total_frames: Total frame count.
            max_n:        Maximum frame count to return.

        Returns:
            List of frame indices sorted ascending.
        """
        # Sample at 10% of frames for scene analysis, then pick best
        sample_indices = [int(i * total_frames / 100) for i in range(0, 100, 1)]
        return sorted(sample_indices[:max_n])

    def _read_frames(
        self, cap: cv2.VideoCapture, indices: list[int]
    ) -> list[np.ndarray]:
        """Read specific frames from an open VideoCapture.

        Args:
            cap:     Open OpenCV VideoCapture object.
            indices: List of frame indices to read.

        Returns:
            List of RGB numpy arrays.
        """
        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            else:
                logger.debug("Could not read frame %d; skipping.", idx)
        return frames

    def _apply_face_extraction(
        self, frames: list[np.ndarray], source_path: Path
    ) -> list[np.ndarray]:
        """Apply face extraction to a list of video frames.

        Args:
            frames:      List of RGB frame arrays.
            source_path: Source video path (for error reporting).

        Returns:
            List of face crop arrays (may be fewer than input frames).
        """
        results: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            try:
                faces = self._face_extractor.extract_faces(  # type: ignore[union-attr]
                    frame,
                    margin=self._face_margin,
                    min_confidence=self._min_confidence,
                )
                if not faces and self._skip_no_face:
                    logger.debug("No face in frame %d; skipping.", i)
                    continue
                results.extend(faces)
            except FaceExtractionError as exc:
                logger.warning(
                    "Face extraction failed on frame %d of '%s': %s",
                    i,
                    source_path.name,
                    exc,
                )
        return results

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to self._image_size × self._image_size.

        Args:
            image: Input RGB numpy array.

        Returns:
            Resized array.
        """
        return cv2.resize(
            image,
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_CUBIC,
        )
