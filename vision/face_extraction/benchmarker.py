"""
DeepGuard — vision/face_extraction/benchmarker.py

Detector benchmarking utilities.

Runs a battery of inference calls and produces a ``BenchmarkReport`` with:
  - Per-run latency (ms)
  - Percentile statistics (p50, p95, p99)
  - Throughput (images/second, faces/second)
  - Memory delta (RSS increase during benchmark)
  - Per-backend comparisons when multiple detectors are supplied
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from vision.face_extraction.base import DetectionConfig, IDetector

logger = logging.getLogger(__name__)


@dataclass
class BackendStats:
    """Benchmark statistics for one detector backend.

    Attributes:
        backend_name:   Identifier string of the backend.
        device:         Compute device used.
        n_runs:         Number of inference calls measured.
        latencies_ms:   All per-run latency values.
        total_faces:    Total detected faces across all runs.
        memory_delta_mb: Memory increase in MB during the benchmark.
    """

    backend_name: str
    device: str
    n_runs: int
    latencies_ms: list[float] = field(default_factory=list)
    total_faces: int = 0
    memory_delta_mb: float = 0.0

    @property
    def mean_ms(self) -> float:
        """Mean latency in ms."""
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def median_ms(self) -> float:
        """Median (P50) latency in ms."""
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        """95th-percentile latency in ms."""
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = max(0, int(len(s) * 0.95) - 1)
        return s[idx]

    @property
    def p99_ms(self) -> float:
        """99th-percentile latency in ms."""
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = max(0, int(len(s) * 0.99) - 1)
        return s[idx]

    @property
    def throughput_img_per_sec(self) -> float:
        """Images processed per second."""
        total_time_s = sum(self.latencies_ms) / 1000.0
        return self.n_runs / max(total_time_s, 1e-9)

    @property
    def throughput_faces_per_sec(self) -> float:
        """Faces detected per second."""
        total_time_s = sum(self.latencies_ms) / 1000.0
        return self.total_faces / max(total_time_s, 1e-9)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "backend_name": self.backend_name,
            "device": self.device,
            "n_runs": self.n_runs,
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "throughput_img_per_sec": round(self.throughput_img_per_sec, 2),
            "throughput_faces_per_sec": round(self.throughput_faces_per_sec, 2),
            "total_faces": self.total_faces,
            "memory_delta_mb": round(self.memory_delta_mb, 2),
        }

    def summary(self) -> str:
        """Format a one-line human-readable summary.

        Returns:
            Formatted summary string.
        """
        return (
            f"[{self.backend_name} @ {self.device}] "
            f"mean={self.mean_ms:.1f}ms p99={self.p99_ms:.1f}ms "
            f"throughput={self.throughput_img_per_sec:.1f}img/s "
            f"faces={self.total_faces}"
        )


@dataclass
class BenchmarkReport:
    """Complete benchmark report for one or more detectors.

    Attributes:
        results:   List of BackendStats, one per benchmarked detector.
        n_images:  Number of test images used.
        n_warmup:  Warmup runs discarded before measurement.
    """

    results: list[BackendStats]
    n_images: int
    n_warmup: int

    def best_by_latency(self) -> BackendStats | None:
        """Return the backend with the lowest mean latency.

        Returns:
            BackendStats of the fastest backend, or None if empty.
        """
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.mean_ms)

    def best_by_throughput(self) -> BackendStats | None:
        """Return the backend with the highest throughput.

        Returns:
            BackendStats of the highest-throughput backend, or None if empty.
        """
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.throughput_img_per_sec)

    def print_report(self) -> None:
        """Print a formatted benchmark report to stdout."""
        sep = "─" * 70
        print(f"\n{'BENCHMARK REPORT':^70}")
        print(sep)
        print(f"  Images per run: {self.n_images}  |  Warmup runs: {self.n_warmup}")
        print(sep)
        for stat in self.results:
            print(f"  {stat.summary()}")
        print(sep)
        if best := self.best_by_latency():
            print(f"  🏆 Fastest: {best.backend_name} ({best.mean_ms:.1f} ms mean)")
        print()

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "n_images": self.n_images,
            "n_warmup": self.n_warmup,
            "results": [r.to_dict() for r in self.results],
        }


class Benchmarker:
    """Runs latency and throughput benchmarks for face detector backends.

    Args:
        config: Detection config used for all benchmarks.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self._config = config or DetectionConfig.default()

    def run(
        self,
        images: Sequence[np.ndarray],
        detectors: Sequence[IDetector],
        n_warmup: int = 3,
        n_runs: int = 20,
    ) -> BenchmarkReport:
        """Benchmark one or more detectors on a set of images.

        Each detector runs n_warmup + n_runs inference calls per image.
        Warmup results are discarded. Final stats aggregate all n_runs × n_images
        individual timing measurements.

        Args:
            images:    List of RGB uint8 test images.
            detectors: List of detector instances to benchmark.
            n_warmup:  Number of warmup passes to discard.
            n_runs:    Number of measured passes per detector per image.

        Returns:
            BenchmarkReport with per-backend statistics.
        """
        if not images:
            raise ValueError("images must not be empty")
        if not detectors:
            raise ValueError("detectors must not be empty")

        results: list[BackendStats] = []

        for detector in detectors:
            if not detector.is_available:
                logger.warning(
                    "Skipping unavailable detector: %s", detector.backend_name
                )
                continue

            logger.info(
                "Benchmarking '%s' (%d warmup, %d runs, %d images).",
                detector.backend_name,
                n_warmup,
                n_runs,
                len(images),
            )

            stats = BackendStats(
                backend_name=detector.backend_name,
                device=detector.device,
                n_runs=n_runs * len(images),
            )

            # Warmup
            for _ in range(n_warmup):
                for img in images:
                    try:
                        detector.detect(img, self._config)
                    except Exception:
                        pass

            # Measure memory before
            mem_before = self._get_rss_mb()

            # Timed runs
            for _ in range(n_runs):
                for img in images:
                    t0 = time.perf_counter()
                    try:
                        detections = detector.detect(img, self._config)
                        stats.total_faces += len(detections)
                    except Exception as exc:
                        logger.debug(
                            "Benchmark: %s error on image: %s",
                            detector.backend_name,
                            exc,
                        )
                        detections = []
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    stats.latencies_ms.append(elapsed_ms)

            # Measure memory after
            mem_after = self._get_rss_mb()
            stats.memory_delta_mb = max(0.0, mem_after - mem_before)

            logger.info("  %s", stats.summary())
            results.append(stats)

        return BenchmarkReport(
            results=results,
            n_images=len(images),
            n_warmup=n_warmup,
        )

    def run_batch_benchmark(
        self,
        images: Sequence[np.ndarray],
        detector: IDetector,
        batch_sizes: Sequence[int] = (1, 4, 8, 16),
        n_runs: int = 10,
    ) -> dict[int, BackendStats]:
        """Benchmark batch inference at multiple batch sizes.

        Args:
            images:      Pool of images to sample batches from.
            detector:    Detector to benchmark.
            batch_sizes: List of batch sizes to evaluate.
            n_runs:      Number of timed runs per batch size.

        Returns:
            Dict mapping batch_size → BackendStats.
        """
        results: dict[int, BackendStats] = {}

        for bs in batch_sizes:
            batch = list(images[:bs]) + [images[0]] * max(0, bs - len(images))
            stats = BackendStats(
                backend_name=detector.backend_name,
                device=detector.device,
                n_runs=n_runs,
            )
            for _ in range(n_runs):
                t0 = time.perf_counter()
                try:
                    per_image = detector.detect_batch(batch, self._config)
                    for dets in per_image:
                        stats.total_faces += len(dets)
                except Exception as exc:
                    logger.debug("Batch benchmark error (bs=%d): %s", bs, exc)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                stats.latencies_ms.append(elapsed_ms)

            logger.info(
                "  batch_size=%d %s", bs, stats.summary()
            )
            results[bs] = stats

        return results

    @staticmethod
    def _get_rss_mb() -> float:
        """Return current process RSS memory in megabytes.

        Returns:
            RSS in MB, or 0.0 if psutil is not installed.
        """
        try:
            import os
            import psutil

            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        except ImportError:
            return 0.0
