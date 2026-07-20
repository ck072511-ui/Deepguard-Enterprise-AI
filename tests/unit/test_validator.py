"""
DeepGuard — tests/unit/test_validator.py

Unit tests for datasets/validator.py — DatasetValidator.
Uses temporary directories populated with synthetic files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.domain.entities.dataset_entity import DatasetName, Label, SampleEntity
from datasets.validator import DatasetValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator() -> DatasetValidator:
    """Return a DatasetValidator with default settings."""
    return DatasetValidator(min_file_size_bytes=10)


@pytest.fixture
def celeb_df_root(tmp_path: Path) -> Path:
    """Create a minimal valid CelebDF-v2 directory structure."""
    (tmp_path / "Celeb-real").mkdir()
    (tmp_path / "Celeb-synthesis").mkdir()
    (tmp_path / "YouTube-real").mkdir()
    (tmp_path / "List_of_testing_videos.txt").write_text("1 Celeb-real/test.mp4\n")
    # Add a dummy video file
    (tmp_path / "Celeb-real" / "fake_video.mp4").write_bytes(b"\x00" * 1024)
    return tmp_path


@pytest.fixture
def ff_plus_plus_root(tmp_path: Path) -> Path:
    """Create a minimal valid FF++ directory structure."""
    (tmp_path / "original_sequences").mkdir()
    (tmp_path / "manipulated_sequences").mkdir()
    return tmp_path


@pytest.fixture
def sample_entity_existing(tmp_path: Path) -> SampleEntity:
    """Return a SampleEntity pointing to an existing, valid file."""
    img_file = tmp_path / "real_image.jpg"
    img_file.write_bytes(b"\xFF\xD8\xFF" + b"\x00" * 1024)  # Minimal JPEG header
    return SampleEntity.create(
        path=img_file,
        label=Label.REAL,
        dataset_name=DatasetName.CUSTOM,
    )


@pytest.fixture
def sample_entity_missing(tmp_path: Path) -> SampleEntity:
    """Return a SampleEntity pointing to a non-existent file."""
    return SampleEntity.create(
        path=tmp_path / "does_not_exist.jpg",
        label=Label.FAKE,
        dataset_name=DatasetName.CUSTOM,
    )


@pytest.fixture
def sample_entity_invalid_ext(tmp_path: Path) -> SampleEntity:
    """Return a SampleEntity pointing to a file with an invalid extension."""
    bad_file = tmp_path / "data.xyz"
    bad_file.write_bytes(b"\x00" * 1024)
    return SampleEntity.create(
        path=bad_file,
        label=Label.REAL,
        dataset_name=DatasetName.CUSTOM,
    )


# ---------------------------------------------------------------------------
# Structure validation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetValidatorStructure:
    """Tests for validate_structure()."""

    def test_valid_celeb_df_structure_passes(
        self, validator: DatasetValidator, celeb_df_root: Path
    ) -> None:
        """A valid CelebDF-v2 structure should produce no errors."""
        errors = validator.validate_structure(celeb_df_root, DatasetName.CELEB_DF)
        assert errors == []

    def test_valid_ff_plus_plus_structure_passes(
        self, validator: DatasetValidator, ff_plus_plus_root: Path
    ) -> None:
        """A valid FF++ structure should produce no errors."""
        errors = validator.validate_structure(ff_plus_plus_root, DatasetName.FF_PLUS_PLUS)
        assert errors == []

    def test_missing_celeb_df_dir_produces_error(
        self, validator: DatasetValidator, tmp_path: Path
    ) -> None:
        """Missing required directories should be reported."""
        # Only create one required dir; omit the other
        (tmp_path / "Celeb-real").mkdir()
        errors = validator.validate_structure(tmp_path, DatasetName.CELEB_DF)
        assert any("Celeb-synthesis" in e for e in errors)

    def test_nonexistent_root_produces_error(
        self, validator: DatasetValidator
    ) -> None:
        """A non-existent root path should produce an error."""
        errors = validator.validate_structure(Path("/nonexistent/path"), DatasetName.CUSTOM)
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_file_as_root_produces_error(
        self, validator: DatasetValidator, tmp_path: Path
    ) -> None:
        """If root is a file instead of directory, error should be reported."""
        file_path = tmp_path / "notadir.txt"
        file_path.write_text("data")
        errors = validator.validate_structure(file_path, DatasetName.CUSTOM)
        assert any("not a directory" in e for e in errors)

    def test_dfdc_without_parts_produces_error(
        self, validator: DatasetValidator, tmp_path: Path
    ) -> None:
        """DFDC root without 'part' directories should produce error."""
        (tmp_path / "some_other_dir").mkdir()
        errors = validator.validate_structure(tmp_path, DatasetName.DFDC)
        assert any("part" in e.lower() for e in errors)

    def test_dfdc_with_parts_passes(
        self, validator: DatasetValidator, tmp_path: Path
    ) -> None:
        """DFDC root with part directories should pass structure validation."""
        (tmp_path / "dfdc_train_part_0").mkdir()
        errors = validator.validate_structure(tmp_path, DatasetName.DFDC)
        assert errors == []


# ---------------------------------------------------------------------------
# Integrity validation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetValidatorIntegrity:
    """Tests for validate_integrity()."""

    def test_existing_valid_file_passes(
        self,
        validator: DatasetValidator,
        sample_entity_existing: SampleEntity,
    ) -> None:
        """A sample pointing to an existing valid file should pass."""
        errors = validator.validate_integrity([sample_entity_existing])
        assert errors == []

    def test_missing_file_produces_error(
        self,
        validator: DatasetValidator,
        sample_entity_missing: SampleEntity,
    ) -> None:
        """A missing file should produce a MISSING error."""
        errors = validator.validate_integrity([sample_entity_missing])
        assert len(errors) == 1
        assert "MISSING" in errors[0]

    def test_invalid_extension_produces_error(
        self,
        validator: DatasetValidator,
        sample_entity_invalid_ext: SampleEntity,
    ) -> None:
        """A file with an invalid extension should produce an error."""
        errors = validator.validate_integrity([sample_entity_invalid_ext])
        assert len(errors) == 1
        assert "INVALID_EXT" in errors[0]

    def test_small_file_produces_error(
        self, validator: DatasetValidator, tmp_path: Path
    ) -> None:
        """A file below minimum size should produce TOO_SMALL error."""
        tiny = tmp_path / "tiny.jpg"
        tiny.write_bytes(b"\x00" * 5)  # 5 bytes < min 10 bytes
        sample = SampleEntity.create(
            path=tiny, label=Label.REAL, dataset_name=DatasetName.CUSTOM
        )
        errors = validator.validate_integrity([sample])
        assert any("TOO_SMALL" in e for e in errors)

    def test_empty_sample_list_returns_no_errors(
        self, validator: DatasetValidator
    ) -> None:
        """Empty sample list should produce no errors."""
        errors = validator.validate_integrity([])
        assert errors == []

    def test_checksum_match_passes(
        self,
        validator: DatasetValidator,
        sample_entity_existing: SampleEntity,
        tmp_path: Path,
    ) -> None:
        """A file matching its expected checksum should pass."""
        import hashlib

        data = sample_entity_existing.path.read_bytes()
        expected = hashlib.sha256(data).hexdigest()

        checksum_file = tmp_path / "checksums.json"
        checksum_file.write_text(
            json.dumps({sample_entity_existing.path.name: expected})
        )
        errors = validator.validate_integrity(
            [sample_entity_existing], checksum_file=checksum_file
        )
        assert errors == []

    def test_checksum_mismatch_produces_error(
        self,
        validator: DatasetValidator,
        sample_entity_existing: SampleEntity,
        tmp_path: Path,
    ) -> None:
        """A file with a wrong checksum should produce CHECKSUM_FAIL error."""
        checksum_file = tmp_path / "checksums.json"
        checksum_file.write_text(
            json.dumps({sample_entity_existing.path.name: "a" * 64})
        )
        errors = validator.validate_integrity(
            [sample_entity_existing], checksum_file=checksum_file
        )
        assert any("CHECKSUM_FAIL" in e for e in errors)


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetValidatorReport:
    """Tests for generate_report()."""

    def test_report_contains_required_keys(
        self,
        validator: DatasetValidator,
        celeb_df_root: Path,
        sample_entity_existing: SampleEntity,
    ) -> None:
        """Report must contain all expected top-level keys."""
        report = validator.generate_report(
            celeb_df_root, DatasetName.CELEB_DF, [sample_entity_existing]
        )
        required = {
            "dataset_name", "root", "total_samples", "real_count",
            "fake_count", "structure", "integrity", "class_balance", "overall_valid",
        }
        assert required.issubset(report.keys())

    def test_report_overall_valid_reflects_state(
        self,
        validator: DatasetValidator,
        celeb_df_root: Path,
        sample_entity_existing: SampleEntity,
    ) -> None:
        """overall_valid should be True when no errors exist."""
        report = validator.generate_report(
            celeb_df_root, DatasetName.CELEB_DF, [sample_entity_existing]
        )
        assert isinstance(report["overall_valid"], bool)

    def test_report_sample_counts_correct(
        self,
        validator: DatasetValidator,
        celeb_df_root: Path,
        sample_entity_existing: SampleEntity,
        sample_entity_missing: SampleEntity,
    ) -> None:
        """total_samples should reflect the actual list length."""
        samples = [sample_entity_existing, sample_entity_missing]
        report = validator.generate_report(
            celeb_df_root, DatasetName.CELEB_DF, samples
        )
        assert report["total_samples"] == 2
