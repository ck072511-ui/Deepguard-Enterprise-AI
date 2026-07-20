"""
DeepGuard — core/domain/value_objects/dataset_split.py

Value objects representing dataset partition configurations.
Immutable, self-validating, equality-by-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class SplitRatios:
    """Immutable value object encapsulating train/val/test split ratios.

    Ratios must be positive floats that sum to exactly 1.0 (within ±1e-6).

    Args:
        train: Fraction of data for training (e.g. 0.7).
        val:   Fraction of data for validation (e.g. 0.15).
        test:  Fraction of data for testing (e.g. 0.15).

    Raises:
        ValueError: If any ratio is non-positive or if they don't sum to 1.

    Example:
        >>> ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
        >>> ratios.train
        0.7
    """

    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        """Validate split ratios on construction."""
        for name, ratio in [("train", self.train), ("val", self.val), ("test", self.test)]:
            if ratio <= 0.0:
                raise ValueError(
                    f"Split ratio '{name}' must be positive, got {ratio}"
                )
            if ratio >= 1.0:
                raise ValueError(
                    f"Split ratio '{name}' must be less than 1.0, got {ratio}"
                )

        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.6f} "
                f"(train={self.train}, val={self.val}, test={self.test})"
            )

    @classmethod
    def from_train_val(cls, train: float, val: float) -> "SplitRatios":
        """Construct ratios from train and val, deriving test as remainder.

        Args:
            train: Training fraction.
            val:   Validation fraction.

        Returns:
            SplitRatios with test = 1.0 - train - val.
        """
        test = round(1.0 - train - val, 10)
        return cls(train=train, val=val, test=test)

    @classmethod
    def default(cls) -> "SplitRatios":
        """Return the standard 80/10/10 split.

        Returns:
            SplitRatios(train=0.8, val=0.1, test=0.1).
        """
        return cls(train=0.8, val=0.1, test=0.1)

    def to_cumulative(self) -> tuple[float, float]:
        """Return cumulative breakpoints (train_end, val_end) for indexing.

        Returns:
            Tuple of (train_cumulative, train+val_cumulative).

        Example:
            >>> SplitRatios(0.7, 0.15, 0.15).to_cumulative()
            (0.7, 0.85)
        """
        return (self.train, self.train + self.val)

    def compute_sizes(self, total: int) -> tuple[int, int, int]:
        """Compute integer sample counts for each split from a total.

        Remainder samples are added to the training set to ensure
        all samples are used.

        Args:
            total: Total number of samples to partition.

        Returns:
            Tuple of (n_train, n_val, n_test) integer counts.

        Example:
            >>> SplitRatios(0.8, 0.1, 0.1).compute_sizes(100)
            (80, 10, 10)
        """
        n_val = max(1, round(total * self.val))
        n_test = max(1, round(total * self.test))
        n_train = total - n_val - n_test
        if n_train <= 0:
            raise ValueError(
                f"Not enough samples ({total}) to create all splits with "
                f"ratios train={self.train}, val={self.val}, test={self.test}. "
                f"At least {n_val + n_test + 1} samples required."
            )
        return (n_train, n_val, n_test)

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """Iterate over (name, ratio) pairs."""
        yield ("train", self.train)
        yield ("val", self.val)
        yield ("test", self.test)

    def __str__(self) -> str:
        return f"SplitRatios(train={self.train:.0%}, val={self.val:.0%}, test={self.test:.0%})"
