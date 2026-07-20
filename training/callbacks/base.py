"""
DeepGuard — training/callbacks/base.py

Base Callback module defining lifecyle hooks for training execution.
"""

from typing import Any


class Callback:
    """Abstract base class for all training callbacks.

    Enables insertion of custom functionality at specific hooks throughout
    the trainer's lifecycle.
    """

    def on_train_begin(self, trainer: Any) -> None:
        """Called immediately before the training loop starts."""

    def on_train_end(self, trainer: Any) -> None:
        """Called immediately after the training loop terminates."""

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each training epoch."""

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each training epoch.

        Args:
            trainer: The active trainer instance.
            epoch:   Active zero-indexed epoch number.
            metrics: Computed metrics dictionary for the completed epoch.
        """

    def on_train_batch_begin(self, trainer: Any, batch_idx: int) -> None:
        """Called immediately before a training batch is processed."""

    def on_train_batch_end(
        self, trainer: Any, batch_idx: int, logs: dict[str, float]
    ) -> None:
        """Called immediately after a training batch completes.

        Args:
            trainer:   The active trainer instance.
            batch_idx: Active zero-indexed batch number.
            logs:      Batch-level computed metrics (e.g. running loss).
        """
