"""
DeepGuard — datasets/splitter.py

Stratified, subject-aware dataset splitter implementing IDatasetSplitter.

Supports:
  - Simple random stratified split (preserves class balance)
  - Subject-aware split (same person never in both train and test)
  - Reproducible with seed
  - Configurable ratios via SplitRatios value object
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from core.domain.entities.dataset_entity import Label, SampleEntity, SplitName
from core.domain.value_objects.dataset_split import SplitRatios
from core.exceptions.dataset_exceptions import DatasetSplitError
from core.interfaces.dataset_interface import IDatasetSplitter

logger = logging.getLogger(__name__)


class StratifiedSplitter(IDatasetSplitter):
    """Stratified random splitter that maintains class balance in each split.

    Splits data independently per class (real vs. fake) then merges,
    so each split has approximately the same class ratio as the full set.

    This is the default splitter when no subject information is available.
    """

    def split(
        self,
        samples: list[SampleEntity],
        ratios: SplitRatios,
        seed: int = 42,
    ) -> dict[SplitName, list[SampleEntity]]:
        """Partition samples into train/val/test with stratification.

        Args:
            samples: All samples to partition.
            ratios:  Target split ratios.
            seed:    Random seed for reproducibility.

        Returns:
            Dictionary mapping SplitName → list of SampleEntity.

        Raises:
            DatasetSplitError: If the sample count is too small to split.
        """
        if len(samples) < 3:
            raise DatasetSplitError(
                "Need at least 3 samples to create train/val/test splits.",
                total_samples=len(samples),
            )

        rng = random.Random(seed)

        # Separate by class
        class_groups: dict[Label, list[SampleEntity]] = defaultdict(list)
        for s in samples:
            class_groups[s.label].append(s)

        result: dict[SplitName, list[SampleEntity]] = {
            SplitName.TRAIN: [],
            SplitName.VAL: [],
            SplitName.TEST: [],
        }

        for label, group in class_groups.items():
            shuffled = list(group)
            rng.shuffle(shuffled)

            try:
                n_train, n_val, n_test = ratios.compute_sizes(len(shuffled))
            except ValueError as exc:
                raise DatasetSplitError(str(exc), total_samples=len(shuffled)) from exc

            result[SplitName.TRAIN].extend(shuffled[:n_train])
            result[SplitName.VAL].extend(shuffled[n_train : n_train + n_val])
            result[SplitName.TEST].extend(shuffled[n_train + n_val :])

            logger.debug(
                "Class %s → train=%d val=%d test=%d",
                label.name,
                n_train,
                n_val,
                n_test,
            )

        # Shuffle within each split to mix classes
        for split_samples in result.values():
            rng.shuffle(split_samples)

        self._log_split_summary(result)
        return result


class SubjectAwareSplitter(IDatasetSplitter):
    """Subject-aware splitter that prevents identity leakage between splits.

    Groups samples by ``subject_id`` and assigns whole identity groups
    to each split. This prevents the model from learning identity-specific
    features rather than manipulation artifacts.

    Falls back to StratifiedSplitter for samples with empty subject_id.
    """

    def split(
        self,
        samples: list[SampleEntity],
        ratios: SplitRatios,
        seed: int = 42,
    ) -> dict[SplitName, list[SampleEntity]]:
        """Partition samples while keeping all samples of each identity together.

        Args:
            samples: All samples with subject_id populated.
            ratios:  Target split ratios.
            seed:    Random seed for reproducibility.

        Returns:
            Dictionary mapping SplitName → list of SampleEntity.

        Raises:
            DatasetSplitError: If not enough subjects to populate all splits.
        """
        if len(samples) < 3:
            raise DatasetSplitError(
                "Need at least 3 samples to split.",
                total_samples=len(samples),
            )

        # Separate samples with and without subject_id
        with_subject: list[SampleEntity] = []
        without_subject: list[SampleEntity] = []
        for s in samples:
            if s.subject_id:
                with_subject.append(s)
            else:
                without_subject.append(s)

        result: dict[SplitName, list[SampleEntity]] = {
            SplitName.TRAIN: [],
            SplitName.VAL: [],
            SplitName.TEST: [],
        }

        if with_subject:
            subject_split = self._split_by_subject(with_subject, ratios, seed)
            for split_name, split_list in subject_split.items():
                result[split_name].extend(split_list)

        if without_subject:
            logger.debug(
                "%d samples have no subject_id; using StratifiedSplitter.",
                len(without_subject),
            )
            fallback = StratifiedSplitter()
            fallback_result = fallback.split(without_subject, ratios, seed)
            for split_name, split_list in fallback_result.items():
                result[split_name].extend(split_list)

        self._log_split_summary(result)
        return result

    def _split_by_subject(
        self,
        samples: list[SampleEntity],
        ratios: SplitRatios,
        seed: int,
    ) -> dict[SplitName, list[SampleEntity]]:
        """Assign subjects to splits then collect their samples.

        Args:
            samples: Samples that all have subject_id set.
            ratios:  Split ratios.
            seed:    RNG seed.

        Returns:
            Split result dictionary.
        """
        rng = random.Random(seed)

        # Group by subject_id (preserving real/fake per subject)
        subject_groups: dict[str, list[SampleEntity]] = defaultdict(list)
        for s in samples:
            subject_groups[s.subject_id].append(s)

        all_subjects = sorted(subject_groups.keys())
        rng.shuffle(all_subjects)

        try:
            n_train, n_val, n_test = ratios.compute_sizes(len(all_subjects))
        except ValueError as exc:
            raise DatasetSplitError(str(exc), total_samples=len(samples)) from exc

        train_subjects = set(all_subjects[:n_train])
        val_subjects = set(all_subjects[n_train : n_train + n_val])
        test_subjects = set(all_subjects[n_train + n_val :])

        result: dict[SplitName, list[SampleEntity]] = {
            SplitName.TRAIN: [],
            SplitName.VAL: [],
            SplitName.TEST: [],
        }

        for subj_id, subj_samples in subject_groups.items():
            if subj_id in train_subjects:
                result[SplitName.TRAIN].extend(subj_samples)
            elif subj_id in val_subjects:
                result[SplitName.VAL].extend(subj_samples)
            elif subj_id in test_subjects:
                result[SplitName.TEST].extend(subj_samples)

        return result


def _log_split_summary(result: dict[SplitName, list[SampleEntity]]) -> None:
    """Log a concise per-split sample count summary.

    Args:
        result: Completed split dictionary.
    """
    for split_name, split_samples in result.items():
        real = sum(1 for s in split_samples if s.label == Label.REAL)
        fake = sum(1 for s in split_samples if s.label == Label.FAKE)
        logger.info(
            "Split %-5s | total=%4d  real=%4d  fake=%4d",
            split_name,
            len(split_samples),
            real,
            fake,
        )


# Monkey-patch the log helper onto the splitter classes for DRY reuse
StratifiedSplitter._log_split_summary = staticmethod(_log_split_summary)  # type: ignore[attr-defined]
SubjectAwareSplitter._log_split_summary = staticmethod(_log_split_summary)  # type: ignore[attr-defined]
