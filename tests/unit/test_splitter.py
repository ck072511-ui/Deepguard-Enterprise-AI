"""
DeepGuard — tests/unit/test_splitter.py

Unit tests for datasets/splitter.py — StratifiedSplitter and
SubjectAwareSplitter. Uses only synthetic SampleEntity objects.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from core.domain.entities.dataset_entity import (
    DatasetName,
    Label,
    ManipulationType,
    MediaType,
    SampleEntity,
    SplitName,
)
from core.domain.value_objects.dataset_split import SplitRatios
from core.exceptions.dataset_exceptions import DatasetSplitError
from datasets.splitter import StratifiedSplitter, SubjectAwareSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(
    label: Label,
    subject_id: str = "",
    path_suffix: str = "",
) -> SampleEntity:
    """Create a minimal SampleEntity for testing.

    Args:
        label:       Real or Fake label.
        subject_id:  Optional subject identity string.
        path_suffix: Unique suffix to make path distinct.

    Returns:
        SampleEntity with a synthetic path.
    """
    uid = str(uuid.uuid4())[:8]
    return SampleEntity(
        sample_id=uid,
        path=Path(f"/fake/path/{uid}{path_suffix}.jpg"),
        label=label,
        dataset_name=DatasetName.CUSTOM,
        media_type=MediaType.IMAGE,
        manipulation=ManipulationType.NONE,
        subject_id=subject_id,
        video_id=uid,
    )


def _make_balanced_samples(n_real: int = 100, n_fake: int = 100) -> list[SampleEntity]:
    """Create a balanced list of real + fake samples.

    Args:
        n_real: Number of real samples.
        n_fake: Number of fake samples.

    Returns:
        Combined list.
    """
    real = [_make_sample(Label.REAL) for _ in range(n_real)]
    fake = [_make_sample(Label.FAKE) for _ in range(n_fake)]
    return real + fake


@pytest.fixture
def balanced_samples() -> list[SampleEntity]:
    """100 real + 100 fake = 200 samples."""
    return _make_balanced_samples(100, 100)


@pytest.fixture
def standard_ratios() -> SplitRatios:
    """Standard 80/10/10 split ratios."""
    return SplitRatios(train=0.8, val=0.1, test=0.1)


# ---------------------------------------------------------------------------
# SplitRatios value object tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSplitRatios:
    """Tests for the SplitRatios value object."""

    def test_valid_ratios_accepted(self) -> None:
        """SplitRatios(0.8, 0.1, 0.1) should construct successfully."""
        ratios = SplitRatios(train=0.8, val=0.1, test=0.1)
        assert ratios.train == 0.8
        assert ratios.val == 0.1
        assert ratios.test == 0.1

    def test_ratios_must_sum_to_one(self) -> None:
        """Ratios that don't sum to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitRatios(train=0.7, val=0.1, test=0.1)  # Sum = 0.9

    def test_negative_ratio_raises(self) -> None:
        """Non-positive ratio should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            SplitRatios(train=-0.8, val=0.9, test=0.9)

    def test_from_train_val_factory(self) -> None:
        """from_train_val should derive test automatically."""
        ratios = SplitRatios.from_train_val(0.7, 0.15)
        assert abs(ratios.test - 0.15) < 1e-6

    def test_default_is_80_10_10(self) -> None:
        """default() should return 80/10/10 ratios."""
        ratios = SplitRatios.default()
        assert ratios.train == 0.8
        assert ratios.val == 0.1
        assert ratios.test == 0.1

    def test_compute_sizes_100_samples(self) -> None:
        """80/10/10 split of 100 samples → (80, 10, 10)."""
        ratios = SplitRatios.default()
        n_train, n_val, n_test = ratios.compute_sizes(100)
        assert n_train == 80
        assert n_val == 10
        assert n_test == 10
        assert n_train + n_val + n_test == 100

    def test_compute_sizes_minimal(self) -> None:
        """Should work with minimal sample counts (>=3)."""
        ratios = SplitRatios.default()
        n_train, n_val, n_test = ratios.compute_sizes(3)
        assert n_train + n_val + n_test == 3

    def test_str_representation(self) -> None:
        """__str__ should include percentage values."""
        ratios = SplitRatios.default()
        s = str(ratios)
        assert "80%" in s
        assert "10%" in s

    def test_to_cumulative(self) -> None:
        """to_cumulative should return (0.8, 0.9) for default ratios."""
        ratios = SplitRatios.default()
        train_end, val_end = ratios.to_cumulative()
        assert train_end == pytest.approx(0.8)
        assert val_end == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# StratifiedSplitter tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStratifiedSplitter:
    """Unit tests for the StratifiedSplitter."""

    def test_returns_three_splits(
        self,
        balanced_samples: list[SampleEntity],
        standard_ratios: SplitRatios,
    ) -> None:
        """split() should return all three SplitName keys."""
        splitter = StratifiedSplitter()
        result = splitter.split(balanced_samples, standard_ratios)
        assert SplitName.TRAIN in result
        assert SplitName.VAL in result
        assert SplitName.TEST in result

    def test_all_samples_present(
        self,
        balanced_samples: list[SampleEntity],
        standard_ratios: SplitRatios,
    ) -> None:
        """Total samples across splits should equal the input count."""
        splitter = StratifiedSplitter()
        result = splitter.split(balanced_samples, standard_ratios)
        total = sum(len(v) for v in result.values())
        assert total == len(balanced_samples)

    def test_no_sample_appears_twice(
        self,
        balanced_samples: list[SampleEntity],
        standard_ratios: SplitRatios,
    ) -> None:
        """Each sample should appear in exactly one split."""
        splitter = StratifiedSplitter()
        result = splitter.split(balanced_samples, standard_ratios)
        all_ids = [s.sample_id for split in result.values() for s in split]
        assert len(all_ids) == len(set(all_ids))

    def test_approximate_train_val_test_sizes(
        self,
        balanced_samples: list[SampleEntity],
        standard_ratios: SplitRatios,
    ) -> None:
        """Splits should approximately match the requested ratios."""
        splitter = StratifiedSplitter()
        result = splitter.split(balanced_samples, standard_ratios, seed=42)
        n = len(balanced_samples)
        # Allow ±5% tolerance per split
        assert len(result[SplitName.TRAIN]) == pytest.approx(n * 0.8, rel=0.1)
        assert len(result[SplitName.VAL]) == pytest.approx(n * 0.1, rel=0.5)
        assert len(result[SplitName.TEST]) == pytest.approx(n * 0.1, rel=0.5)

    def test_class_balance_preserved_in_train(
        self,
        balanced_samples: list[SampleEntity],
        standard_ratios: SplitRatios,
    ) -> None:
        """Train split should maintain real/fake balance."""
        splitter = StratifiedSplitter()
        result = splitter.split(balanced_samples, standard_ratios)
        train = result[SplitName.TRAIN]
        n_real = sum(1 for s in train if s.label == Label.REAL)
        n_fake = sum(1 for s in train if s.label == Label.FAKE)
        balance = n_real / max(len(train), 1)
        assert 0.45 <= balance <= 0.55, f"Train balance {balance:.2%} not near 50/50"

    def test_reproducible_with_same_seed(
        self,
        balanced_samples: list[SampleEntity],
        standard_ratios: SplitRatios,
    ) -> None:
        """Same seed must produce identical splits."""
        splitter = StratifiedSplitter()
        r1 = splitter.split(balanced_samples, standard_ratios, seed=0)
        r2 = splitter.split(balanced_samples, standard_ratios, seed=0)
        for split_name in SplitName:
            ids1 = sorted(s.sample_id for s in r1[split_name])
            ids2 = sorted(s.sample_id for s in r2[split_name])
            assert ids1 == ids2

    def test_different_seeds_produce_different_splits(
        self,
        balanced_samples: list[SampleEntity],
        standard_ratios: SplitRatios,
    ) -> None:
        """Different seeds should generally produce different assignments."""
        splitter = StratifiedSplitter()
        r1 = splitter.split(balanced_samples, standard_ratios, seed=1)
        r2 = splitter.split(balanced_samples, standard_ratios, seed=2)
        ids1 = sorted(s.sample_id for s in r1[SplitName.TRAIN])
        ids2 = sorted(s.sample_id for s in r2[SplitName.TRAIN])
        assert ids1 != ids2  # Very unlikely to be equal with 200 samples

    def test_raises_with_too_few_samples(
        self, standard_ratios: SplitRatios
    ) -> None:
        """Should raise DatasetSplitError with fewer than 3 samples."""
        splitter = StratifiedSplitter()
        with pytest.raises(DatasetSplitError):
            splitter.split([_make_sample(Label.REAL), _make_sample(Label.FAKE)], standard_ratios)


# ---------------------------------------------------------------------------
# SubjectAwareSplitter tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubjectAwareSplitter:
    """Unit tests for the SubjectAwareSplitter."""

    def test_no_subject_leakage_between_train_and_test(
        self, standard_ratios: SplitRatios
    ) -> None:
        """A subject should not appear in both train and test splits."""
        # 10 subjects, 10 samples each
        samples = [
            _make_sample(Label.REAL if i % 2 == 0 else Label.FAKE, subject_id=f"sub_{i // 10}")
            for i in range(100)
        ]
        splitter = SubjectAwareSplitter()
        result = splitter.split(samples, standard_ratios)

        train_subjs = {s.subject_id for s in result[SplitName.TRAIN]}
        test_subjs = {s.subject_id for s in result[SplitName.TEST]}
        overlap = train_subjs & test_subjs
        assert len(overlap) == 0, f"Subject leakage: {overlap}"

    def test_falls_back_gracefully_for_empty_subject_id(
        self, standard_ratios: SplitRatios
    ) -> None:
        """Samples without subject_id should still be split (via fallback)."""
        samples = [_make_sample(Label.REAL) for _ in range(30)] + [
            _make_sample(Label.FAKE) for _ in range(30)
        ]
        splitter = SubjectAwareSplitter()
        result = splitter.split(samples, standard_ratios)
        total = sum(len(v) for v in result.values())
        assert total == 60
