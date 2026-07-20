"""
DeepGuard — datasets/statistics.py

Dataset statistics computation implementing IDatasetStatistics.

Computes:
  - Per-class sample counts and ratios
  - Per-split breakdowns
  - Manipulation type distribution (FF++)
  - Compression level distribution (FF++)
  - File size statistics
  - Image dimension statistics (sampled)
  - Class balance metrics
  - Exports to JSON report
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from core.domain.entities.dataset_entity import Label, ManipulationType, SampleEntity, SplitName
from core.interfaces.dataset_interface import IDatasetStatistics

logger = logging.getLogger(__name__)


class DatasetStatistics(IDatasetStatistics):
    """Comprehensive dataset statistics computer.

    Args:
        sample_size_for_dimensions: Number of files to sample for dimension
                                    stats (reading all would be slow).
    """

    def __init__(self, sample_size_for_dimensions: int = 100) -> None:
        self._dim_sample_size = sample_size_for_dimensions

    # ------------------------------------------------------------------
    # IDatasetStatistics interface
    # ------------------------------------------------------------------

    def compute(self, samples: list[SampleEntity]) -> dict[str, Any]:
        """Compute full dataset statistics.

        Args:
            samples: All dataset samples to analyse.

        Returns:
            Nested dictionary of computed statistics.
        """
        if not samples:
            logger.warning("[Statistics] No samples to compute statistics for.")
            return {"total_samples": 0, "error": "No samples provided."}

        logger.info("[Statistics] Computing statistics for %d samples.", len(samples))

        stats: dict[str, Any] = {
            "total_samples": len(samples),
            "class_distribution": self._class_distribution(samples),
            "split_distribution": self._split_distribution(samples),
            "manipulation_distribution": self._manipulation_distribution(samples),
            "compression_distribution": self._compression_distribution(samples),
            "media_type_distribution": self._media_type_distribution(samples),
            "file_size_stats": self._file_size_statistics(samples),
            "class_balance": self._class_balance_metrics(samples),
        }

        logger.info("[Statistics] Statistics computation complete.")
        return stats

    # ------------------------------------------------------------------
    # Individual statistics methods
    # ------------------------------------------------------------------

    def _class_distribution(self, samples: list[SampleEntity]) -> dict[str, Any]:
        """Compute per-class sample counts.

        Args:
            samples: All samples.

        Returns:
            Dictionary with 'real', 'fake', 'total', 'real_pct', 'fake_pct'.
        """
        n_real = sum(1 for s in samples if s.label == Label.REAL)
        n_fake = sum(1 for s in samples if s.label == Label.FAKE)
        total = len(samples)
        return {
            "real": n_real,
            "fake": n_fake,
            "total": total,
            "real_pct": round(100.0 * n_real / max(total, 1), 2),
            "fake_pct": round(100.0 * n_fake / max(total, 1), 2),
        }

    def _split_distribution(
        self, samples: list[SampleEntity]
    ) -> dict[str, dict[str, int]]:
        """Compute per-split, per-class counts.

        Args:
            samples: All samples.

        Returns:
            Nested dict: {split_name: {real, fake, total}}.
        """
        split_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"real": 0, "fake": 0, "total": 0}
        )
        for s in samples:
            split_name = str(s.split)
            split_counts[split_name]["total"] += 1
            if s.label == Label.REAL:
                split_counts[split_name]["real"] += 1
            else:
                split_counts[split_name]["fake"] += 1

        return dict(split_counts)

    def _manipulation_distribution(
        self, samples: list[SampleEntity]
    ) -> dict[str, int]:
        """Count samples per manipulation type.

        Args:
            samples: All samples.

        Returns:
            Dictionary mapping manipulation type name → count.
        """
        counter = Counter(str(s.manipulation) for s in samples)
        return dict(counter.most_common())

    def _compression_distribution(
        self, samples: list[SampleEntity]
    ) -> dict[str, int]:
        """Count samples per compression level.

        Args:
            samples: All samples.

        Returns:
            Dictionary mapping compression level → count.
        """
        counter = Counter(str(s.compression) for s in samples)
        return dict(counter.most_common())

    def _media_type_distribution(
        self, samples: list[SampleEntity]
    ) -> dict[str, int]:
        """Count samples per media type.

        Args:
            samples: All samples.

        Returns:
            Dictionary mapping media type → count.
        """
        counter = Counter(str(s.media_type) for s in samples)
        return dict(counter.most_common())

    def _file_size_statistics(
        self, samples: list[SampleEntity]
    ) -> dict[str, Any]:
        """Compute file size statistics across all existing files.

        Args:
            samples: All samples.

        Returns:
            Dictionary with min, max, mean, median, std (bytes) and total_gb.
        """
        sizes: list[int] = []
        missing = 0

        for sample in samples:
            if sample.path.exists():
                sizes.append(sample.path.stat().st_size)
            else:
                missing += 1

        if not sizes:
            return {
                "count": 0,
                "missing_files": missing,
                "min_bytes": 0,
                "max_bytes": 0,
                "mean_bytes": 0,
                "median_bytes": 0,
                "std_bytes": 0,
                "total_gb": 0.0,
            }

        return {
            "count": len(sizes),
            "missing_files": missing,
            "min_bytes": min(sizes),
            "max_bytes": max(sizes),
            "mean_bytes": round(statistics.mean(sizes)),
            "median_bytes": round(statistics.median(sizes)),
            "std_bytes": round(statistics.stdev(sizes)) if len(sizes) > 1 else 0,
            "total_gb": round(sum(sizes) / (1024**3), 4),
        }

    def _class_balance_metrics(
        self, samples: list[SampleEntity]
    ) -> dict[str, Any]:
        """Compute class imbalance metrics.

        Args:
            samples: All samples.

        Returns:
            Dictionary with balance ratio, imbalance ratio, is_balanced flag.
        """
        n_real = sum(1 for s in samples if s.label == Label.REAL)
        n_fake = sum(1 for s in samples if s.label == Label.FAKE)
        total = max(len(samples), 1)

        balance = n_real / total
        minority = min(n_real, n_fake)
        majority = max(n_real, n_fake)
        imbalance_ratio = majority / max(minority, 1)

        return {
            "real_fraction": round(balance, 4),
            "fake_fraction": round(1.0 - balance, 4),
            "imbalance_ratio": round(imbalance_ratio, 2),
            "is_balanced": 0.4 <= balance <= 0.6,
            "minority_class": "real" if n_real <= n_fake else "fake",
            "minority_count": minority,
            "majority_count": majority,
            "recommended_strategy": (
                "none"
                if imbalance_ratio < 1.5
                else "oversample_minority"
                if imbalance_ratio < 5
                else "class_weights_or_focal_loss"
            ),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_report(
        self,
        stats: dict[str, Any],
        output_path: Path,
    ) -> None:
        """Export computed statistics to a JSON file.

        Args:
            stats:       Statistics dictionary from compute().
            output_path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info("[Statistics] Report saved to '%s'.", output_path)

    def format_summary(self, stats: dict[str, Any]) -> str:
        """Format statistics as a human-readable multi-line string.

        Args:
            stats: Statistics dictionary from compute().

        Returns:
            Formatted string suitable for logging or CLI output.
        """
        lines = [
            "═" * 60,
            "  DATASET STATISTICS SUMMARY",
            "═" * 60,
            f"  Total samples  : {stats.get('total_samples', 0):,}",
        ]

        cd = stats.get("class_distribution", {})
        if cd:
            lines.append(
                f"  Real samples   : {cd.get('real', 0):,} ({cd.get('real_pct', 0):.1f}%)"
            )
            lines.append(
                f"  Fake samples   : {cd.get('fake', 0):,} ({cd.get('fake_pct', 0):.1f}%)"
            )

        sd = stats.get("split_distribution", {})
        if sd:
            lines.append("  " + "─" * 56)
            lines.append("  Split distribution:")
            for split_name, counts in sd.items():
                lines.append(
                    f"    {split_name:6s}: {counts.get('total', 0):6,} "
                    f"(real={counts.get('real', 0):,} fake={counts.get('fake', 0):,})"
                )

        fs = stats.get("file_size_stats", {})
        if fs and fs.get("count", 0) > 0:
            lines.append("  " + "─" * 56)
            lines.append(
                f"  Total dataset  : {fs.get('total_gb', 0):.2f} GB"
            )
            lines.append(
                f"  File size mean : {fs.get('mean_bytes', 0) / 1024:.1f} KB"
            )

        cb = stats.get("class_balance", {})
        if cb:
            lines.append("  " + "─" * 56)
            lines.append(f"  Balanced       : {cb.get('is_balanced', False)}")
            lines.append(
                f"  Imbalance ratio: {cb.get('imbalance_ratio', 1.0):.2f}x"
            )
            lines.append(
                f"  Recommendation : {cb.get('recommended_strategy', 'none')}"
            )

        lines.append("═" * 60)
        return "\n".join(lines)
