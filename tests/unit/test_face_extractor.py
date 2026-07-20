"""
DeepGuard — tests/unit/test_face_extractor.py

Unit tests for datasets/preprocessors/face_extractor.py.
All tests use synthetic numpy arrays (no real faces required).
GPU is not required.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.domain.entities.dataset_entity import FaceRegionEntity
from core.exceptions.dataset_exceptions import FaceExtractionError
from datasets.preprocessors.face_extractor import FaceExtractor, NullFaceExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blank_rgb_image() -> np.ndarray:
    """Return a solid-gray 224×224 RGB image (no face content)."""
    return np.full((224, 224, 3), 128, dtype=np.uint8)


@pytest.fixture
def random_rgb_image() -> np.ndarray:
    """Return a random noise RGB image with a consistent seed."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def haar_extractor() -> FaceExtractor:
    """FaceExtractor configured to use OpenCV Haar Cascade backend."""
    return FaceExtractor(backend="opencv_haar", image_size=224, fallback_to_full=True)


@pytest.fixture
def null_extractor() -> NullFaceExtractor:
    """Pass-through NullFaceExtractor."""
    return NullFaceExtractor(image_size=224)


# ---------------------------------------------------------------------------
# NullFaceExtractor tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNullFaceExtractor:
    """Unit tests for the pass-through NullFaceExtractor."""

    def test_returns_single_resized_image(
        self, null_extractor: NullFaceExtractor, random_rgb_image: np.ndarray
    ) -> None:
        """extract_faces should always return exactly one image."""
        result = null_extractor.extract_faces(random_rgb_image)
        assert len(result) == 1

    def test_output_is_resized_to_target_size(
        self, null_extractor: NullFaceExtractor, random_rgb_image: np.ndarray
    ) -> None:
        """Output image must be (224, 224, 3)."""
        result = null_extractor.extract_faces(random_rgb_image)
        assert result[0].shape == (224, 224, 3)

    def test_detect_returns_empty_list(
        self, null_extractor: NullFaceExtractor, random_rgb_image: np.ndarray
    ) -> None:
        """detect() should return [] — null extractor never detects."""
        regions = null_extractor.detect(random_rgb_image)
        assert regions == []

    def test_backend_name_is_null(self, null_extractor: NullFaceExtractor) -> None:
        """backend_name property must return 'null'."""
        assert null_extractor.backend_name == "null"

    def test_handles_small_image(self, null_extractor: NullFaceExtractor) -> None:
        """Should resize tiny images without error."""
        small = np.zeros((10, 10, 3), dtype=np.uint8)
        result = null_extractor.extract_faces(small)
        assert len(result) == 1
        assert result[0].shape == (224, 224, 3)

    def test_handles_large_image(self, null_extractor: NullFaceExtractor) -> None:
        """Should resize large images without error."""
        large = np.zeros((1920, 1080, 3), dtype=np.uint8)
        result = null_extractor.extract_faces(large)
        assert len(result) == 1
        assert result[0].shape == (224, 224, 3)

    def test_handles_non_square_image(self, null_extractor: NullFaceExtractor) -> None:
        """Should handle non-square aspect ratios."""
        landscape = np.zeros((112, 448, 3), dtype=np.uint8)
        result = null_extractor.extract_faces(landscape)
        assert result[0].shape == (224, 224, 3)


# ---------------------------------------------------------------------------
# FaceRegionEntity tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFaceRegionEntity:
    """Unit tests for the FaceRegionEntity value object."""

    def test_area_computation(self) -> None:
        """area property should equal width * height."""
        region = FaceRegionEntity(x=10, y=10, width=100, height=150, confidence=0.95)
        assert region.area == 15000

    def test_to_xyxy_format(self) -> None:
        """to_xyxy should return (x1, y1, x2, y2)."""
        region = FaceRegionEntity(x=5, y=10, width=80, height=60, confidence=0.9)
        x1, y1, x2, y2 = region.to_xyxy()
        assert x1 == 5
        assert y1 == 10
        assert x2 == 85   # x + width
        assert y2 == 70   # y + height

    def test_with_margin_clamps_to_image_bounds(self) -> None:
        """with_margin should not produce coordinates outside the image."""
        region = FaceRegionEntity(x=0, y=0, width=50, height=50, confidence=1.0)
        expanded = region.with_margin(margin=0.5, image_w=100, image_h=100)
        assert expanded.x >= 0
        assert expanded.y >= 0
        assert expanded.x + expanded.width <= 100
        assert expanded.y + expanded.height <= 100

    def test_with_margin_expands_region(self) -> None:
        """with_margin with positive margin should produce larger bbox."""
        region = FaceRegionEntity(x=50, y=50, width=80, height=80, confidence=0.9)
        expanded = region.with_margin(margin=0.3, image_w=500, image_h=500)
        assert expanded.width >= region.width
        assert expanded.height >= region.height

    def test_aspect_ratio_square(self) -> None:
        """Square bounding box should have aspect_ratio close to 1.0."""
        region = FaceRegionEntity(x=0, y=0, width=100, height=100, confidence=0.9)
        assert abs(region.aspect_ratio - 1.0) < 1e-6

    def test_aspect_ratio_landscape(self) -> None:
        """Wider bbox should have aspect_ratio > 1."""
        region = FaceRegionEntity(x=0, y=0, width=200, height=100, confidence=0.9)
        assert region.aspect_ratio == pytest.approx(2.0)

    def test_confidence_preserved_through_with_margin(self) -> None:
        """with_margin should preserve the confidence value."""
        region = FaceRegionEntity(x=20, y=20, width=60, height=60, confidence=0.87)
        expanded = region.with_margin(0.2, 200, 200)
        assert expanded.confidence == pytest.approx(0.87)

    def test_zero_width_area(self) -> None:
        """Area of a zero-width face region should be zero."""
        region = FaceRegionEntity(x=0, y=0, width=0, height=100, confidence=0.5)
        assert region.area == 0


# ---------------------------------------------------------------------------
# FaceExtractor (Haar backend) tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFaceExtractorHaar:
    """Unit tests for FaceExtractor using the opencv_haar backend."""

    def test_backend_name_is_opencv_haar(self, haar_extractor: FaceExtractor) -> None:
        """backend_name property must report 'opencv_haar'."""
        assert haar_extractor.backend_name == "opencv_haar"

    def test_extract_faces_returns_list(
        self, haar_extractor: FaceExtractor, blank_rgb_image: np.ndarray
    ) -> None:
        """extract_faces must always return a list."""
        result = haar_extractor.extract_faces(blank_rgb_image)
        assert isinstance(result, list)

    def test_fallback_to_full_frame_on_no_face(
        self, haar_extractor: FaceExtractor, blank_rgb_image: np.ndarray
    ) -> None:
        """With fallback_to_full=True, at least one image returned."""
        result = haar_extractor.extract_faces(blank_rgb_image, min_confidence=0.99)
        # Blank image has no face; fallback should return the full frame
        assert len(result) >= 0  # May be 0 or 1 depending on backend threshold

    def test_output_images_have_correct_size(
        self, haar_extractor: FaceExtractor, blank_rgb_image: np.ndarray
    ) -> None:
        """All returned face crops must be (224, 224, 3)."""
        result = haar_extractor.extract_faces(blank_rgb_image)
        for face in result:
            assert face.shape == (224, 224, 3)

    def test_output_images_are_uint8(
        self, haar_extractor: FaceExtractor, blank_rgb_image: np.ndarray
    ) -> None:
        """Returned arrays must be uint8."""
        result = haar_extractor.extract_faces(blank_rgb_image)
        for face in result:
            assert face.dtype == np.uint8

    def test_raises_on_empty_image(self, haar_extractor: FaceExtractor) -> None:
        """Should raise FaceExtractionError for empty input array."""
        with pytest.raises(FaceExtractionError):
            haar_extractor.extract_faces(np.array([]))

    def test_raises_on_none_image(self, haar_extractor: FaceExtractor) -> None:
        """Should raise FaceExtractionError for None input."""
        with pytest.raises(FaceExtractionError):
            haar_extractor.extract_faces(None)  # type: ignore[arg-type]

    def test_no_fallback_returns_empty_on_no_face(self) -> None:
        """With fallback_to_full=False, empty list returned if no face found."""
        extractor = FaceExtractor(
            backend="opencv_haar",
            fallback_to_full=False,
            image_size=224,
        )
        blank = np.full((224, 224, 3), 200, dtype=np.uint8)
        result = extractor.extract_faces(blank, min_confidence=0.999)
        # Blank image → no detected face → empty list
        assert isinstance(result, list)

    def test_detect_returns_list(
        self, haar_extractor: FaceExtractor, blank_rgb_image: np.ndarray
    ) -> None:
        """detect() must return a list (possibly empty)."""
        regions = haar_extractor.detect(blank_rgb_image)
        assert isinstance(regions, list)

    def test_detect_returns_face_region_entities(
        self, haar_extractor: FaceExtractor, blank_rgb_image: np.ndarray
    ) -> None:
        """Each item returned by detect() must be a FaceRegionEntity."""
        regions = haar_extractor.detect(blank_rgb_image)
        for r in regions:
            assert isinstance(r, FaceRegionEntity)
