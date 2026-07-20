"""
DeepGuard — tests/unit/test_loaders.py

Unit tests for dataset loader utilities:
  - CustomDataset with a real temporary directory structure
  - DatasetVersioner (save/load/list)
  - ImagePreprocessor
  - VideoPreprocessor (metadata, index computation)
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.domain.entities.dataset_entity import (
    DatasetMetadataEntity,
    DatasetName,
    Label,
    SampleEntity,
    SplitName,
)
from datasets.loaders.custom_loader import CustomDataset
from datasets.preprocessors.image_preprocessor import ImagePreprocessor, NormalizationParams
from datasets.preprocessors.video_preprocessor import FrameSamplingStrategy, VideoPreprocessor
from datasets.versioning import DatasetVersioner
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def custom_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal custom dataset with 10 real + 10 fake JPEG images."""
    (tmp_path / "real").mkdir()
    (tmp_path / "fake").mkdir()

    for i in range(10):
        # Write a minimal valid JPEG (just header bytes + padding)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 1] = 128  # Green channel for real
        cv2.imwrite(str(tmp_path / "real" / f"real_{i:03d}.jpg"), img)

        img2 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2[:, :, 2] = 128  # Red channel for fake
        cv2.imwrite(str(tmp_path / "fake" / f"fake_{i:03d}.jpg"), img2)

    return tmp_path


@pytest.fixture
def custom_dataset_with_csv(tmp_path: Path, custom_dataset_dir: Path) -> Path:
    """Create a CSV manifest for the custom dataset directory."""
    manifest_path = custom_dataset_dir / "manifest.csv"
    rows = []
    for f in (custom_dataset_dir / "real").iterdir():
        rows.append({"path": f"real/{f.name}", "label": "0"})
    for f in (custom_dataset_dir / "fake").iterdir():
        rows.append({"path": f"fake/{f.name}", "label": "1"})

    with manifest_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return custom_dataset_dir


@pytest.fixture
def sample_metadata() -> DatasetMetadataEntity:
    """Return a minimal DatasetMetadataEntity for versioning tests."""
    return DatasetMetadataEntity(
        dataset_id=str(uuid.uuid4()),
        name=DatasetName.CUSTOM,
        version="1.0",
        root_path=Path("/fake/root"),
        total_samples=20,
        real_count=10,
        fake_count=10,
        created_at=datetime.now(tz=timezone.utc),
        description="Test metadata",
    )


@pytest.fixture
def sample_entities() -> list[SampleEntity]:
    """Return a list of 5 synthetic SampleEntity objects."""
    return [
        SampleEntity.create(
            path=Path(f"/fake/path/sample_{i}.jpg"),
            label=Label.REAL if i % 2 == 0 else Label.FAKE,
            dataset_name=DatasetName.CUSTOM,
        )
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# CustomDataset tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCustomDataset:
    """Tests for the CustomDataset loader using a real temp directory."""

    def test_loads_train_samples(self, custom_dataset_dir: Path) -> None:
        """Train split should contain samples from real/ and fake/ dirs."""
        ds = CustomDataset(
            root=custom_dataset_dir,
            split=SplitName.TRAIN,
            max_samples=None,
        )
        assert len(ds) > 0

    def test_real_and_fake_labels_present(
        self, custom_dataset_dir: Path
    ) -> None:
        """Both REAL and FAKE labels should be discoverable."""
        # Load all by combining splits
        all_samples: list[SampleEntity] = []
        for split in SplitName:
            try:
                ds = CustomDataset(root=custom_dataset_dir, split=split)
                all_samples.extend(ds.samples)
            except Exception:
                pass

        labels = {s.label for s in all_samples}
        assert Label.REAL in labels
        assert Label.FAKE in labels

    def test_getitem_returns_tensor_and_label(
        self, custom_dataset_dir: Path
    ) -> None:
        """__getitem__ must return (FloatTensor, int) tuple."""
        import torch

        ds = CustomDataset(root=custom_dataset_dir, split=SplitName.TRAIN)
        if len(ds) == 0:
            pytest.skip("No samples in train split for this seed")
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert isinstance(label, int)
        assert label in (0, 1)

    def test_getitem_tensor_shape(self, custom_dataset_dir: Path) -> None:
        """Tensor shape must be (3, image_size, image_size)."""
        import torch

        ds = CustomDataset(root=custom_dataset_dir, split=SplitName.TRAIN, image_size=64)
        if len(ds) == 0:
            pytest.skip("No samples in train split")
        tensor, _ = ds[0]
        assert tensor.shape == torch.Size([3, 64, 64])

    def test_dataset_name_is_custom(self, custom_dataset_dir: Path) -> None:
        """dataset_name property should return 'custom'."""
        ds = CustomDataset(root=custom_dataset_dir, split=SplitName.TRAIN)
        assert ds.dataset_name == DatasetName.CUSTOM

    def test_class_counts_dict(self, custom_dataset_dir: Path) -> None:
        """class_counts should return dict with 'real', 'fake', 'total'."""
        ds = CustomDataset(root=custom_dataset_dir, split=SplitName.TRAIN)
        counts = ds.class_counts
        assert "real" in counts
        assert "fake" in counts
        assert "total" in counts
        assert counts["total"] == counts["real"] + counts["fake"]

    def test_get_labels_list(self, custom_dataset_dir: Path) -> None:
        """get_labels() should return a list of 0/1 integers."""
        ds = CustomDataset(root=custom_dataset_dir, split=SplitName.TRAIN)
        labels = ds.get_labels()
        assert all(lbl in (0, 1) for lbl in labels)
        assert len(labels) == len(ds)

    def test_out_of_range_index_raises(self, custom_dataset_dir: Path) -> None:
        """Accessing index beyond length should raise IndexError."""
        ds = CustomDataset(root=custom_dataset_dir, split=SplitName.TRAIN)
        with pytest.raises(IndexError):
            _ = ds[len(ds) + 100]

    def test_csv_manifest_layout_detection(
        self, custom_dataset_with_csv: Path
    ) -> None:
        """Dataset with manifest.csv should load via CSV layout."""
        ds = CustomDataset(
            root=custom_dataset_with_csv,
            split=SplitName.TRAIN,
            manifest_file="manifest.csv",
        )
        assert len(ds) > 0

    def test_class_weights_shape(self, custom_dataset_dir: Path) -> None:
        """class_weights should be a tensor of shape (2,)."""
        import torch

        ds = CustomDataset(root=custom_dataset_dir, split=SplitName.TRAIN)
        weights = ds.class_weights
        assert weights.shape == torch.Size([2])
        assert (weights > 0).all()

    def test_repr_contains_dataset_info(self, custom_dataset_dir: Path) -> None:
        """__repr__ should contain split and sample count."""
        ds = CustomDataset(root=custom_dataset_dir, split=SplitName.TRAIN)
        r = repr(ds)
        assert "train" in r.lower() or "split" in r.lower()


# ---------------------------------------------------------------------------
# ImagePreprocessor tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImagePreprocessor:
    """Tests for the ImagePreprocessor."""

    def test_preprocess_returns_rgb_array(self) -> None:
        """preprocess should return an RGB numpy array."""
        proc = ImagePreprocessor(target_size=224, input_is_bgr=True)
        bgr_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = proc.preprocess(bgr_img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)

    def test_preprocess_resizes_to_target(self) -> None:
        """Output must be target_size × target_size."""
        for size in [112, 224, 384]:
            proc = ImagePreprocessor(target_size=size)
            img = np.zeros((50, 200, 3), dtype=np.uint8)
            result = proc.preprocess(img)
            assert result.shape == (size, size, 3)

    def test_preprocess_batch(self) -> None:
        """preprocess_batch should process all images."""
        proc = ImagePreprocessor(target_size=64)
        imgs = [np.zeros((100, 100, 3), dtype=np.uint8)] * 4
        results = proc.preprocess_batch(imgs)
        assert len(results) == 4
        assert all(r.shape == (64, 64, 3) for r in results)

    def test_normalize_returns_float32(self) -> None:
        """normalize_array should convert to float32."""
        proc = ImagePreprocessor(normalize=True)
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        result = proc.normalize_array(img)
        assert result.dtype == np.float32

    def test_validates_none_input(self) -> None:
        """preprocess should raise ValueError for None input."""
        proc = ImagePreprocessor()
        with pytest.raises(ValueError, match="None"):
            proc.preprocess(None)  # type: ignore[arg-type]

    def test_validates_wrong_dims(self) -> None:
        """preprocess should raise ValueError for 1D input."""
        proc = ImagePreprocessor()
        with pytest.raises(ValueError, match="2D or 3D"):
            proc.preprocess(np.zeros(100))

    def test_image_statistics_keys(self) -> None:
        """compute_image_statistics should return standard metric keys."""
        proc = ImagePreprocessor()
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        stats = proc.compute_image_statistics(img)
        for ch in ["R", "G", "B"]:
            assert f"{ch}_mean" in stats
            assert f"{ch}_std" in stats
        assert "sharpness" in stats
        assert "height" in stats
        assert "width" in stats

    def test_clahe_does_not_change_shape(self) -> None:
        """CLAHE transform should not alter the image dimensions."""
        proc = ImagePreprocessor(target_size=224, apply_clahe=True)
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = proc.preprocess(img)
        assert result.shape == (224, 224, 3)


# ---------------------------------------------------------------------------
# DatasetVersioner tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetVersioner:
    """Tests for the DatasetVersioner save/load/list cycle."""

    def test_save_creates_manifest_file(
        self,
        tmp_path: Path,
        sample_entities: list[SampleEntity],
        sample_metadata: DatasetMetadataEntity,
    ) -> None:
        """save_version should create a .manifest.json file."""
        versioner = DatasetVersioner()
        manifest_path = versioner.save_version(sample_entities, sample_metadata, tmp_path)
        assert manifest_path.exists()
        assert manifest_path.suffix == ".json"

    def test_saved_manifest_is_valid_json(
        self,
        tmp_path: Path,
        sample_entities: list[SampleEntity],
        sample_metadata: DatasetMetadataEntity,
    ) -> None:
        """Saved manifest should be parseable JSON."""
        versioner = DatasetVersioner()
        path = versioner.save_version(sample_entities, sample_metadata, tmp_path)
        with path.open() as f:
            data = json.load(f)
        assert "manifest_version" in data
        assert "total_samples" in data

    def test_load_returns_same_sample_count(
        self,
        tmp_path: Path,
        sample_entities: list[SampleEntity],
        sample_metadata: DatasetMetadataEntity,
    ) -> None:
        """load_version should return the same number of samples as saved."""
        versioner = DatasetVersioner()
        path = versioner.save_version(sample_entities, sample_metadata, tmp_path)
        loaded_samples, _ = versioner.load_version(path)
        assert len(loaded_samples) == len(sample_entities)

    def test_load_returns_metadata(
        self,
        tmp_path: Path,
        sample_entities: list[SampleEntity],
        sample_metadata: DatasetMetadataEntity,
    ) -> None:
        """load_version should return a DatasetMetadataEntity."""
        versioner = DatasetVersioner()
        path = versioner.save_version(sample_entities, sample_metadata, tmp_path)
        _, meta = versioner.load_version(path)
        assert isinstance(meta, DatasetMetadataEntity)
        assert meta.name == DatasetName.CUSTOM

    def test_list_versions_finds_manifests(
        self,
        tmp_path: Path,
        sample_entities: list[SampleEntity],
        sample_metadata: DatasetMetadataEntity,
    ) -> None:
        """list_versions should discover saved manifests."""
        versioner = DatasetVersioner()
        versioner.save_version(sample_entities, sample_metadata, tmp_path)
        versioner.save_version(sample_entities, sample_metadata, tmp_path)
        versions = versioner.list_versions(tmp_path)
        assert len(versions) == 2

    def test_list_versions_empty_dir(self, tmp_path: Path) -> None:
        """list_versions on empty directory should return empty list."""
        versioner = DatasetVersioner()
        versions = versioner.list_versions(tmp_path)
        assert versions == []

    def test_checksum_mismatch_raises(
        self,
        tmp_path: Path,
        sample_entities: list[SampleEntity],
        sample_metadata: DatasetMetadataEntity,
    ) -> None:
        """Corrupted manifest should raise DatasetVersionError on load."""
        from core.exceptions.dataset_exceptions import DatasetVersionError

        versioner = DatasetVersioner()
        path = versioner.save_version(sample_entities, sample_metadata, tmp_path)

        # Corrupt the manifest by overwriting the checksum
        with path.open() as f:
            data = json.load(f)
        data["manifest_checksum"] = "0" * 64
        with path.open("w") as f:
            json.dump(data, f)

        with pytest.raises(DatasetVersionError, match="checksum mismatch"):
            versioner.load_version(path)
