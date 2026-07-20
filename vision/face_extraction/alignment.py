"""
DeepGuard — vision/face_extraction/alignment.py

Face alignment via affine transformation to a canonical 5-point template.

The canonical template is based on the FFHQ dataset landmark positions
(scaled to 112×112 by default). Given 5 facial landmarks from any detector,
we compute an affine warp that maps the face into a standardised pose:
  - Eyes aligned horizontally
  - Face centred and upright
  - Consistent scale

This dramatically reduces intra-class variance and improves model accuracy.

Reference template (ArcFace 112×112):
  Left eye:    [38.29, 51.70]
  Right eye:   [73.53, 51.50]
  Nose:        [56.02, 71.74]
  Left mouth:  [41.55, 92.37]
  Right mouth: [70.73, 92.20]
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FFHQ / ArcFace canonical 5-point template, normalised for 112×112
# ---------------------------------------------------------------------------
_ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],   # left eye centre
        [73.5318, 51.5014],   # right eye centre
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


def _scale_template(target_size: int) -> np.ndarray:
    """Scale the 112×112 template to an arbitrary square output size.

    Args:
        target_size: Side length of the output image in pixels.

    Returns:
        (5, 2) float32 array scaled to target_size.
    """
    scale = target_size / 112.0
    return (_ARCFACE_TEMPLATE_112 * scale).astype(np.float32)


class FaceAligner:
    """Aligns face crops to a canonical pose using a 5-point affine warp.

    Args:
        output_size:  Side length of the output square image (pixels).
        border_mode:  OpenCV border mode for out-of-bounds pixels.
        border_value: Fill value for BORDER_CONSTANT mode.
    """

    def __init__(
        self,
        output_size: int = 112,
        border_mode: int = cv2.BORDER_REFLECT,
        border_value: int = 0,
    ) -> None:
        self._output_size = output_size
        self._border_mode = border_mode
        self._border_value = border_value
        self._template = _scale_template(output_size)

    @property
    def output_size(self) -> int:
        """Return the configured output image size."""
        return self._output_size

    def align(
        self,
        image: np.ndarray,
        landmarks_5pt: np.ndarray,
    ) -> np.ndarray:
        """Affine-warp a face region to canonical pose.

        Estimates a partial affine transform (no shear) mapping the 5
        detected landmarks onto the canonical template positions.

        Args:
            image:         Full RGB image (H, W, 3) uint8.
            landmarks_5pt: Detected (5, 2) float32 landmark array.

        Returns:
            Aligned face crop as uint8 RGB array of shape
            (output_size, output_size, 3).

        Raises:
            ValueError: If landmarks shape is wrong or image is empty.
        """
        self._validate_inputs(image, landmarks_5pt)

        src = landmarks_5pt.astype(np.float32)
        dst = self._template

        transform, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.LMEDS
        )
        if transform is None:
            logger.warning(
                "Affine estimation failed (degenerate landmarks). "
                "Returning centre crop."
            )
            return self._centre_crop(image)

        aligned = cv2.warpAffine(
            image,
            transform,
            (self._output_size, self._output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=self._border_mode,
            borderValue=self._border_value,
        )
        return aligned

    def align_batch(
        self,
        image: np.ndarray,
        landmarks_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Align multiple faces from the same image.

        Args:
            image:          Full RGB image (H, W, 3).
            landmarks_list: List of (5, 2) float32 landmark arrays.

        Returns:
            List of aligned face crop arrays.
        """
        return [self.align(image, lm) for lm in landmarks_list]

    def crop_without_alignment(
        self,
        image: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop and resize a face region without alignment.

        Used as fallback when landmarks are unavailable.

        Args:
            image:     RGB image (H, W, 3).
            bbox_xyxy: Bounding box (x1, y1, x2, y2).

        Returns:
            Cropped, resized face array of shape (output_size, output_size, 3).
        """
        x1, y1, x2, y2 = bbox_xyxy
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return self._centre_crop(image)

        crop = image[y1:y2, x1:x2]
        return cv2.resize(
            crop,
            (self._output_size, self._output_size),
            interpolation=cv2.INTER_CUBIC,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _centre_crop(self, image: np.ndarray) -> np.ndarray:
        """Return a centre crop resized to output_size × output_size.

        Args:
            image: RGB image array.

        Returns:
            Resized square crop.
        """
        h, w = image.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = image[y0 : y0 + side, x0 : x0 + side]
        return cv2.resize(
            crop,
            (self._output_size, self._output_size),
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def _validate_inputs(image: np.ndarray, landmarks: np.ndarray) -> None:
        """Validate image and landmark inputs.

        Args:
            image:     Input image.
            landmarks: Landmark array.

        Raises:
            ValueError: On invalid input.
        """
        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3), got shape {image.shape}")
        if landmarks is None or landmarks.shape != (5, 2):
            raise ValueError(
                f"landmarks_5pt must be a (5, 2) array, got {getattr(landmarks, 'shape', None)}"
            )


def align_face(
    image: np.ndarray,
    landmarks_5pt: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """Convenience function: align a single face in one call.

    Args:
        image:         RGB image (H, W, 3) uint8.
        landmarks_5pt: (5, 2) float32 landmark array.
        output_size:   Output image side length in pixels.

    Returns:
        Aligned face crop (output_size, output_size, 3) uint8.
    """
    return FaceAligner(output_size=output_size).align(image, landmarks_5pt)


def compute_alignment_score(
    landmarks_5pt: np.ndarray,
    output_size: int = 112,
) -> float:
    """Estimate how well detected landmarks match the canonical template.

    A score near 1.0 means the face is already well-aligned (frontal).
    A score near 0.0 means heavily rotated or occluded.

    Args:
        landmarks_5pt: Detected (5, 2) float32 array in pixel coords.
        output_size:   Reference output size.

    Returns:
        Alignment score in [0.0, 1.0].
    """
    if landmarks_5pt is None or landmarks_5pt.shape != (5, 2):
        return 0.0

    template = _scale_template(output_size)
    # Normalise both to unit scale before comparing
    src_centred = landmarks_5pt - landmarks_5pt.mean(axis=0)
    dst_centred = template - template.mean(axis=0)

    src_scale = max(np.linalg.norm(src_centred), 1e-6)
    dst_scale = max(np.linalg.norm(dst_centred), 1e-6)

    src_norm = src_centred / src_scale
    dst_norm = dst_centred / dst_scale

    # Cosine similarity in flattened space
    score = float(np.clip(np.dot(src_norm.ravel(), dst_norm.ravel()), 0.0, 1.0))
    return score
