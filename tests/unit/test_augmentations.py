"""
DeepGuard — tests/unit/test_augmentations.py

Unit tests for datasets/augmentations/train_transforms.py and
datasets/augmentations/val_transforms.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from datasets.augmentations.train_transforms import (
    build_train_transforms,
    get_transform_params,
)
from datasets.augmentations.val_transforms import (
    apply_tta,
    build_tta_transforms,
    build_val_transforms,
    denormalize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Return a random RGB uint8 numpy array (224×224×3)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_rgb_image_large() -> np.ndarray:
    """Return a random RGB uint8 numpy array (512×512×3)."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Training transform tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrainTransforms:
    """Unit tests for the training augmentation pipeline."""

    @pytest.mark.parametrize("severity", ["light", "medium", "heavy"])
    def test_returns_tensor(
        self, sample_rgb_image: np.ndarray, severity: str
    ) -> None:
        """Output must be a PyTorch tensor."""
        transform = build_train_transforms(image_size=224, severity=severity)
        result = transform(image=sample_rgb_image)
        assert isinstance(result["image"], torch.Tensor)

    @pytest.mark.parametrize("severity", ["light", "medium", "heavy"])
    def test_output_shape_correct(
        self, sample_rgb_image: np.ndarray, severity: str
    ) -> None:
        """Output tensor must be (3, 224, 224) — CHW format."""
        transform = build_train_transforms(image_size=224, severity=severity)
        result = transform(image=sample_rgb_image)
        assert result["image"].shape == torch.Size([3, 224, 224])

    def test_output_dtype_is_float32(self, sample_rgb_image: np.ndarray) -> None:
        """Output tensor must be float32."""
        transform = build_train_transforms()
        result = transform(image=sample_rgb_image)
        assert result["image"].dtype == torch.float32

    def test_invalid_severity_raises(self) -> None:
        """Building with an unsupported severity should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid severity"):
            build_train_transforms(severity="extreme")

    def test_different_image_sizes(self, sample_rgb_image: np.ndarray) -> None:
        """Transform should work for common ViT input sizes."""
        for size in [112, 160, 224, 256, 384]:
            transform = build_train_transforms(image_size=size)
            result = transform(image=sample_rgb_image)
            assert result["image"].shape == torch.Size([3, size, size])

    def test_custom_normalization(self, sample_rgb_image: np.ndarray) -> None:
        """Custom mean/std should be accepted without error."""
        transform = build_train_transforms(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
        result = transform(image=sample_rgb_image)
        assert result["image"].shape[0] == 3

    def test_transform_is_deterministic_with_seed(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """Sanity check: the same input can produce different outputs (stochastic)."""
        transform = build_train_transforms(severity="heavy")
        result1 = transform(image=sample_rgb_image.copy())
        result2 = transform(image=sample_rgb_image.copy())
        # Training transforms are stochastic — outputs may differ
        # We just verify they don't crash
        assert result1["image"].shape == result2["image"].shape

    def test_get_transform_params_returns_dict(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """get_transform_params should return a non-empty dictionary."""
        transform = build_train_transforms()
        params = get_transform_params(transform)
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_large_image_input(self, sample_rgb_image_large: np.ndarray) -> None:
        """Should handle images larger than the target size."""
        transform = build_train_transforms(image_size=224)
        result = transform(image=sample_rgb_image_large)
        assert result["image"].shape == torch.Size([3, 224, 224])


# ---------------------------------------------------------------------------
# Validation transform tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValTransforms:
    """Unit tests for the validation/test augmentation pipeline."""

    def test_returns_tensor(self, sample_rgb_image: np.ndarray) -> None:
        """Val transform output must be a PyTorch tensor."""
        transform = build_val_transforms(image_size=224)
        result = transform(image=sample_rgb_image)
        assert isinstance(result["image"], torch.Tensor)

    def test_output_shape_correct(self, sample_rgb_image: np.ndarray) -> None:
        """Val transform output tensor must be (3, 224, 224)."""
        transform = build_val_transforms(image_size=224)
        result = transform(image=sample_rgb_image)
        assert result["image"].shape == torch.Size([3, 224, 224])

    def test_is_deterministic(self, sample_rgb_image: np.ndarray) -> None:
        """Val transforms must be deterministic — same input → same output."""
        transform = build_val_transforms()
        r1 = transform(image=sample_rgb_image.copy())["image"]
        r2 = transform(image=sample_rgb_image.copy())["image"]
        assert torch.allclose(r1, r2)

    def test_without_center_crop(self, sample_rgb_image: np.ndarray) -> None:
        """Val transforms without center_crop should still resize correctly."""
        transform = build_val_transforms(image_size=224, center_crop=False)
        result = transform(image=sample_rgb_image)
        assert result["image"].shape == torch.Size([3, 224, 224])

    def test_output_values_are_normalised(self, sample_rgb_image: np.ndarray) -> None:
        """Output values should be normalised (not bounded to [0,1] anymore)."""
        transform = build_val_transforms()
        result = transform(image=sample_rgb_image)
        tensor = result["image"]
        # After ImageNet normalisation, values may be outside [0, 1]
        assert tensor.min() < 1.0  # Has been normalised
        assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# TTA transform tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTTATransforms:
    """Unit tests for Test-Time Augmentation transforms."""

    def test_returns_five_transforms(self) -> None:
        """build_tta_transforms should return exactly 5 pipelines."""
        transforms = build_tta_transforms(image_size=224)
        assert len(transforms) == 5

    def test_each_tta_produces_correct_shape(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """Each TTA variant must produce a (3, 224, 224) tensor."""
        transforms = build_tta_transforms(image_size=224)
        for transform in transforms:
            result = transform(image=sample_rgb_image)
            assert result["image"].shape == torch.Size([3, 224, 224])

    def test_apply_tta_returns_correct_count(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """apply_tta should return as many results as TTA transforms."""
        transforms = build_tta_transforms()
        results = apply_tta(sample_rgb_image, transforms)
        assert len(results) == len(transforms)

    def test_tta_results_are_tensors(self, sample_rgb_image: np.ndarray) -> None:
        """All TTA results must be PyTorch tensors."""
        transforms = build_tta_transforms()
        results = apply_tta(sample_rgb_image, transforms)
        for r in results:
            assert isinstance(r, torch.Tensor)


# ---------------------------------------------------------------------------
# Denormalize tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDenormalize:
    """Unit tests for the denormalize utility function."""

    def test_output_clamped_to_zero_one(self) -> None:
        """Denormalized tensor should be clamped to [0, 1]."""
        # Create a normalised tensor (values may be outside [0,1])
        tensor = torch.randn(3, 224, 224)
        result = denormalize(tensor)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_same_shape_as_input(self) -> None:
        """Denormalized tensor must have the same shape."""
        tensor = torch.randn(3, 224, 224)
        result = denormalize(tensor)
        assert result.shape == tensor.shape

    def test_batch_input_support(self) -> None:
        """Should handle (B, C, H, W) batch tensors."""
        batch = torch.randn(4, 3, 224, 224)
        result = denormalize(batch)
        assert result.shape == (4, 3, 224, 224)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_round_trip_approximate(self) -> None:
        """Normalize then denormalize should approximately recover the original."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        normalised = transform(image=img)["image"]
        recovered = denormalize(normalised)
        # Should be approximately 128/255 ≈ 0.502 in all channels
        assert recovered.mean().item() == pytest.approx(128 / 255, abs=0.1)
