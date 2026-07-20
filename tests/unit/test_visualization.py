"""
DeepGuard — tests/unit/test_visualization.py

Unit tests for dataset visualization helper functions.
"""

from pathlib import Path
import numpy as np
import pytest
import cv2
import matplotlib.pyplot as plt

from core.domain.entities.dataset_entity import Label, SampleEntity, DatasetName, SplitName, ManipulationType
from datasets.visualization import (
    plot_class_distribution,
    plot_split_distribution,
    plot_sample_grid,
    plot_augmentation_comparison,
    plot_manipulation_distribution,
)


@pytest.fixture
def dummy_samples(temp_dir: Path) -> list[SampleEntity]:
    """Provide a list of dummy SampleEntity objects with existing image files on disk."""
    samples = []
    # Create 3 real and 3 fake samples
    for i in range(6):
        label = Label.REAL if i % 2 == 0 else Label.FAKE
        split = SplitName.TRAIN if i < 4 else SplitName.VAL
        manipulation = ManipulationType.NONE if label == Label.REAL else ManipulationType.DEEPFAKES
        
        # Create a dummy image file
        img_path = temp_dir / f"sample_{i}.jpg"
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        
        sample = SampleEntity.create(
            path=img_path,
            label=label,
            dataset_name=DatasetName.FF_PLUS_PLUS,
            split=split,
            manipulation=manipulation,
        )
        samples.append(sample)
    return samples


def test_plot_class_distribution(dummy_samples: list[SampleEntity], temp_dir: Path) -> None:
    output_path = temp_dir / "class_dist.png"
    fig = plot_class_distribution(dummy_samples, output_path=output_path)
    assert fig is not None
    assert output_path.exists()


def test_plot_split_distribution(dummy_samples: list[SampleEntity], temp_dir: Path) -> None:
    output_path = temp_dir / "split_dist.png"
    fig = plot_split_distribution(dummy_samples, output_path=output_path)
    assert fig is not None
    assert output_path.exists()


def test_plot_sample_grid(dummy_samples: list[SampleEntity], temp_dir: Path) -> None:
    output_path = temp_dir / "sample_grid.png"
    fig = plot_sample_grid(dummy_samples, n_rows=2, n_cols=2, output_path=output_path)
    assert fig is not None
    assert output_path.exists()


def test_plot_augmentation_comparison(temp_dir: Path) -> None:
    output_path = temp_dir / "aug_comparison.png"
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    aug_imgs = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
    
    fig = plot_augmentation_comparison(img, augmented_images=aug_imgs, output_path=output_path)
    assert fig is not None
    assert output_path.exists()


def test_plot_manipulation_distribution(dummy_samples: list[SampleEntity], temp_dir: Path) -> None:
    output_path = temp_dir / "manip_dist.png"
    fig = plot_manipulation_distribution(dummy_samples, output_path=output_path)
    assert fig is not None
    assert output_path.exists()


def test_plot_manipulation_distribution_empty(temp_dir: Path) -> None:
    # Test plot_manipulation_distribution with no fake samples
    samples = [
        SampleEntity.create(
            path=temp_dir / "only_real.jpg",
            label=Label.REAL,
            dataset_name=DatasetName.FF_PLUS_PLUS,
        )
    ]
    fig = plot_manipulation_distribution(samples)
    assert fig is not None
