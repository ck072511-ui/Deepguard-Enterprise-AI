"""
DeepGuard — training/callbacks/early_stopping.py

Early stopping callback module to prevent overfitting during model training.
"""

import logging
from typing import Any
from training.callbacks.base import Callback

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(Callback):
    """Monitors a metric and terminates training if improvement stops.

    Args:
        monitor:    Metric key name to monitor (default: 'val/auc_roc').
        patience:   Number of epochs with no improvement to wait before stopping.
        min_delta:  Minimum change in target metric to qualify as an improvement.
        mode:       'min' | 'max' (default: 'max' for AUC-ROC).
    """

    def __init__(
        self,
        monitor: str = "val/auc_roc",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()

        self.wait = 0
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        val = metrics.get(self.monitor)
        if val is None:
            logger.warning(
                "Early stopping monitor key '%s' not found in epoch metrics. "
                "Available keys: %s",
                self.monitor,
                list(metrics.keys()),
            )
            return

        if self.mode == "min":
            improved = val < (self.best - self.min_delta)
        else:
            improved = val > (self.best + self.min_delta)

        if improved:
            self.best = val
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                logger.info(
                    "Early stopping triggered at epoch %d. "
                    "Best '%s' was %.4f",
                    epoch + 1,
                    self.monitor,
                    self.best,
                )
