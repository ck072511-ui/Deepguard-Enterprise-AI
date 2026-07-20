"""
DeepGuard — tests/unit/test_statistics.py

Unit tests for datasets/statistics.py — DatasetStatistics.
Uses synthetic SampleEntity objects; no real files required for
most tests (file size stats use temp files).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from core.domain.entities.dataset_entity import (
    CompressionLevel,
    DatasetName,
    Label,
    ManipulationType,
    MediaType,
    SampleEntity,
    SplitName,
)
from datasets.statistics import DatasetStatistics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(
    label: Label,
    split: SplitName = SplitName.TRAIN,
    manipulation: ManipulationType = ManipulationType.NONE,
    compression: CompressionLevel = CompressionLevel.UNKNOWN,
    path: Path | None = None,
) -> SampleEntity:
    """Create a synthetic SampleEntity for testing.

    Args:
        label:       Classification label.
        split:       Dataset split assignment.
        manipulation: Manipulation type.
        compression:  Compression level.
        path:        Optional file path (for file-based tests).

    Returns:
        SampleEntity with synthetic data.
    """
    uid = str(uuid.uuid4())[:8]
    return SampleEntity(
        sample_id=uid,
        path=path or Path(f"/fake/{uid}.jpg"),
        label=label,
        dataset_name=DatasetName.CUSTOM,
        media_type=MediaType.IMAGE,
        manipulation=manipulation,
        compression=compression,
        split=split,
        video_id=uid,
    )


@pytest.fixture
def balanced_mixed_samples() -> list[SampleEntity]:
    """60 real + 60 fake samples across train/val/test splits."""
    samples = []
    for split, n in [(SplitName.TRAIN, 40), (SplitName.VAL, 10), (SplitName.TEST, 10)]:
        for _ in range(n):
            samples.append(_make_sample(Label.REAL, split=split))
            samples.append(_make_sample(Label.FAKE, split=split))
    return samples


@pytest.fixture
def stats() -> DatasetStatistics:
    """DatasetStatistics instance for testing."""
    return DatasetStatistics()


# ---------------------------------------------------------------------------
# DatasetStatistics tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetStatistics:
    """Unit tests for DatasetStatistics.compute()."""

    def test_empty_samples_returns_dict(self, stats: DatasetStatistics) -> None:
        """compute() with empty list should return a dict with total_samples=0."""
        result = stats.compute([])
        assert isinstance(result, dict)
        assert result["total_samples"] == 0

    def test_total_samples_correct(
        self, stats: DatasetStatistics, balanced_mixed_samples: list[SampleEntity]
    ) -> None:
        """total_samples should match the input count."""
        result = stats.compute(balanced_mixed_samples)
        assert result["total_samples"] == len(balanced_mixed_samples)

    def test_class_distribution_correct(
        self, stats: DatasetStatistics, balanced_mixed_samples: list[SampleEntity]
    ) -> None:
        """Class distribution should correctly count real/fake."""
        result = stats.compute(balanced_mixed_samples)
        cd = result["class_distribution"]
        n_real = sum(1 for s in balanced_mixed_samples if s.label == Label.REAL)
        n_fake = sum(1 for s in balanced_mixed_samples if s.label == Label.FAKE)
        assert cd["real"] == n_real
        assert cd["fake"] == n_fake
        assert cd["total"] == len(balanced_mixed_samples)

    def test_class_percentages_sum_to_100(
        self, stats: DatasetStatistics, balanced_mixed_samples: list[SampleEntity]
    ) -> None:
        """real_pct + fake_pct should sum to 100.0."""
        result = stats.compute(balanced_mixed_samples)
        cd = result["class_distribution"]
        total_pct = cd["real_pct"] + cd["fake_pct"]
        assert total_pct == pytest.approx(100.0, abs=0.01)

    def test_split_distribution_has_all_splits(
        self, stats: DatasetStatistics, balanced_mixed_samples: list[SampleEntity]
    ) -> None:
        """Split distribution should include train, val, and test keys."""
        result = stats.compute(balanced_mixed_samples)
        sd = result["split_distribution"]
        assert "train" in sd
        assert "val" in sd
        assert "test" in sd

    def test_split_distribution_totals_match(
        self, stats: DatasetStatistics, balanced_mixed_samples: list[SampleEntity]
    ) -> None:
        """Sum of split totals should equal total_samples."""
        result = stats.compute(balanced_mixed_samples)
        sd = result["split_distribution"]
        split_total = sum(v["total"] for v in sd.values())
        assert split_total == result["total_samples"]

    def test_manipulation_distribution_contains_none(
        self, stats: DatasetStatistics
    ) -> None:
        """Manipulation distribution should include 'none' for real samples."""
        samples = [_make_sample(Label.REAL, manipulation=ManipulationType.NONE)] * 10
        result = stats.compute(samples)
        md = result["manipulation_distribution"]
        assert ManipulationType.NONE in md or "none" in md

    def test_balanced_dataset_detected(
        self, stats: DatasetStatistics, balanced_mixed_samples: list[SampleEntity]
    ) -> None:
        """is_balanced should be True for a 50/50 dataset."""
        result = stats.compute(balanced_mixed_samples)
        cb = result["class_balance"]
        assert cb["is_balanced"] is True

    def test_imbalanced_dataset_detected(self, stats: DatasetStatistics) -> None:
        """is_balanced should be False for a highly imbalanced dataset."""
        samples = (
            [_make_sample(Label.REAL)] * 90
            + [_make_sample(Label.FAKE)] * 10
        )
        result = stats.compute(samples)
        cb = result["class_balance"]
        assert cb["is_balanced"] is False
        assert cb["imbalance_ratio"] == pytest.approx(9.0, abs=0.5)

    def test_minority_class_identified(self, stats: DatasetStatistics) -> None:
        """minority_class should be 'fake' when fakes are fewer."""
        samples = (
            [_make_sample(Label.REAL)] * 80
            + [_make_sample(Label.FAKE)] * 20
        )
        result = stats.compute(samples)
        cb = result["class_balance"]
        assert cb["minority_class"] == "fake"
        assert cb["minority_count"] == 20
        assert cb["majority_count"] == 80

    def test_file_size_stats_with_missing_files(
        self, stats: DatasetStatistics
    ) -> None:
        """File size stats should handle missing files gracefully."""
        samples = [_make_sample(Label.REAL)]  # Path doesn't exist
        result = stats.compute(samples)
        fs = result["file_size_stats"]
        assert fs["missing_files"] == 1
        assert fs["count"] == 0

    def test_file_size_stats_with_real_files(
        self, stats: DatasetStatistics, tmp_path: Path
    ) -> None:
        """File size stats should correctly measure existing files."""
        # Create two temp files of known sizes
        f1 = tmp_path / "r1.jpg"
        f1.write_bytes(b"\x00" * 2048)  # 2 KB
        f2 = tmp_path / "f1.jpg"
        f2.write_bytes(b"\x00" * 4096)  # 4 KB

        samples = [
            _make_sample(Label.REAL, path=f1),
            _make_sample(Label.FAKE, path=f2),
        ]
        result = stats.compute(samples)
        fs = result["file_size_stats"]
        assert fs["count"] == 2
        assert fs["min_bytes"] == 2048
        assert fs["max_bytes"] == 4096
        assert fs["mean_bytes"] == pytest.approx(3072, abs=1)

    def test_export_report_creates_file(
        self, stats: DatasetStatistics, tmp_path: Path
    ) -> None:
        """export_report should create a JSON file at the target path."""
        result = stats.compute([_make_sample(Label.REAL)] * 5)
        out = tmp_path / "stats.json"
        stats.export_report(result, out)
        assert out.exists()
        import json
        with out.open() as f:
            loaded = json.load(f)
        assert loaded["total_samples"] == 5

    def test_format_summary_returns_string(
        self, stats: DatasetStatistics, balanced_mixed_samples: list[SampleEntity]
    ) -> None:
        """format_summary should return a non-empty string."""
        result = stats.compute(balanced_mixed_samples)
        summary = stats.format_summary(result)
        assert isinstance(summary, str)
        assert len(summary) > 50

    def test_format_summary_contains_key_info(
        self, stats: DatasetStatistics, balanced_mixed_samples: list[SampleEntity]
    ) -> None:
        """Summary should mention sample counts."""
        result = stats.compute(balanced_mixed_samples)
        summary = stats.format_summary(result)
        assert "Total samples" in summary or "DATASET STATISTICS" in summary
