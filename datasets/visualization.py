"""
DeepGuard — datasets/visualization.py

Dataset visualization utilities for exploratory data analysis (EDA),
sample inspection, class distribution plots, and augmentation previews.

All plotting uses Matplotlib with non-interactive 'Agg' backend so it
works in headless server environments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from core.domain.entities.dataset_entity import Label, SampleEntity

logger = logging.getLogger(__name__)

# Consistent colour palette
_REAL_COLOR = "#2ECC71"   # Green
_FAKE_COLOR = "#E74C3C"   # Red
_ACCENT_COLOR = "#3498DB" # Blue


def plot_class_distribution(
    samples: list[SampleEntity],
    title: str = "Class Distribution",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (8, 5),
) -> Figure:
    """Plot a bar chart of real vs. fake sample counts.

    Args:
        samples:     Dataset samples to analyse.
        title:       Plot title string.
        output_path: If provided, saves the figure to this path.
        figsize:     Figure dimensions (width, height) in inches.

    Returns:
        Matplotlib Figure object.
    """
    n_real = sum(1 for s in samples if s.label == Label.REAL)
    n_fake = sum(1 for s in samples if s.label == Label.FAKE)
    total = max(len(samples), 1)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        ["Real", "Fake"],
        [n_real, n_fake],
        color=[_REAL_COLOR, _FAKE_COLOR],
        edgecolor="white",
        linewidth=1.5,
        width=0.5,
    )

    for bar, count in zip(bars, [n_real, n_fake]):
        pct = 100.0 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylim(0, max(n_real, n_fake) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_split_distribution(
    samples: list[SampleEntity],
    title: str = "Split Distribution",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> Figure:
    """Plot grouped bar chart of real/fake counts per split.

    Args:
        samples:     Dataset samples to analyse.
        title:       Plot title.
        output_path: Save path (optional).
        figsize:     Figure dimensions.

    Returns:
        Matplotlib Figure object.
    """
    from collections import defaultdict

    split_data: dict[str, dict[str, int]] = defaultdict(lambda: {"real": 0, "fake": 0})
    for s in samples:
        split_name = str(s.split)
        if s.label == Label.REAL:
            split_data[split_name]["real"] += 1
        else:
            split_data[split_name]["fake"] += 1

    splits = sorted(split_data.keys())
    reals = [split_data[s]["real"] for s in splits]
    fakes = [split_data[s]["fake"] for s in splits]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars_real = ax.bar(x - width / 2, reals, width, label="Real", color=_REAL_COLOR)
    bars_fake = ax.bar(x + width / 2, fakes, width, label="Fake", color=_FAKE_COLOR)

    for bars in [bars_real, bars_fake]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1,
                f"{h:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits], fontsize=11)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_sample_grid(
    samples: list[SampleEntity],
    n_rows: int = 3,
    n_cols: int = 4,
    image_size: int = 224,
    title: str = "Sample Grid",
    output_path: Path | None = None,
) -> Figure:
    """Plot a grid of sample images with their labels.

    Randomly selects up to n_rows × n_cols images from the sample list.

    Args:
        samples:     Dataset samples to display.
        n_rows:      Number of grid rows.
        n_cols:      Number of grid columns.
        image_size:  Size to resize each image to (square).
        title:       Plot title.
        output_path: Save path (optional).

    Returns:
        Matplotlib Figure object.
    """
    import random

    n_images = n_rows * n_cols
    selected = random.sample(samples, min(n_images, len(samples)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.8))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= len(selected):
            continue

        sample = selected[i]
        img = _load_thumbnail(sample.path, image_size)
        if img is not None:
            ax.imshow(img)

        label_str = "REAL" if sample.label == Label.REAL else "FAKE"
        color = _REAL_COLOR if sample.label == Label.REAL else _FAKE_COLOR
        ax.set_title(label_str, color=color, fontsize=9, fontweight="bold", pad=3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_augmentation_comparison(
    image: np.ndarray,
    augmented_images: list[np.ndarray],
    titles: list[str] | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Show an original image alongside its augmented versions.

    Args:
        image:            Original RGB image (H, W, 3) uint8.
        augmented_images: List of augmented RGB images.
        titles:           Labels for augmented images (auto-generated if None).
        output_path:      Save path (optional).

    Returns:
        Matplotlib Figure object.
    """
    n = 1 + len(augmented_images)
    titles = titles or [f"Aug {i + 1}" for i in range(len(augmented_images))]
    all_titles = ["Original"] + titles
    all_images = [image] + augmented_images

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, img, t in zip(axes, all_images, all_titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Augmentation Preview", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_manipulation_distribution(
    samples: list[SampleEntity],
    title: str = "Manipulation Type Distribution",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> Figure:
    """Plot horizontal bar chart of fake samples per manipulation type.

    Args:
        samples:     Dataset samples to analyse.
        title:       Plot title.
        output_path: Save path (optional).
        figsize:     Figure dimensions.

    Returns:
        Matplotlib Figure object.
    """
    from collections import Counter

    fake_samples = [s for s in samples if s.label == Label.FAKE]
    if not fake_samples:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No fake samples", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    counts = Counter(str(s.manipulation) for s in fake_samples)
    labels_list = list(counts.keys())
    values = [counts[l] for l in labels_list]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(labels_list)))  # type: ignore[attr-defined]
    bars = ax.barh(labels_list, values, color=colors, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=9,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sample Count", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_thumbnail(path: Path, size: int) -> np.ndarray | None:
    """Load and resize an image file for display.

    Args:
        path: File path to load.
        size: Target square size in pixels.

    Returns:
        RGB numpy array or None on failure.
    """
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if suffix in video_exts:
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def _maybe_save(fig: Figure, output_path: Path | None) -> None:
    """Save the figure to disk if output_path is specified.

    Args:
        fig:         Matplotlib figure to save.
        output_path: Destination path or None.
    """
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    logger.info("[Visualization] Saved plot to '%s'.", output_path)
    plt.close(fig)
