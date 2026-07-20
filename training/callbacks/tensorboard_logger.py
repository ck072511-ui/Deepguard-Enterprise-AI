"""
DeepGuard — training/callbacks/tensorboard_logger.py

TensorBoard logging callback module.
"""

import logging
from typing import Any
from torch.utils.tensorboard import SummaryWriter
from training.callbacks.base import Callback

logger = logging.getLogger(__name__)


class TensorBoardCallback(Callback):
    """Callback for logging training metrics to TensorBoard event files.

    Args:
        log_dir: Directory path for saving TensorBoard event logs.
    """

    def __init__(self, log_dir: str = "logs/tensorboard") -> None:
        self.log_dir = log_dir
        self._writer: SummaryWriter | None = None

    def on_train_begin(self, trainer: Any) -> None:
        try:
            self._writer = SummaryWriter(log_dir=self.log_dir)
            logger.info("TensorBoard logging initialized at '%s'.", self.log_dir)
        except Exception as exc:
            logger.warning("Failed to initialize TensorBoard SummaryWriter: %s", exc)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        if self._writer is None:
            return
        try:
            for k, v in metrics.items():
                # Replace slashes with standard classification tags for cleaner plots grouping
                tag = k.replace("/", "_")
                self._writer.add_scalar(tag, v, global_step=epoch + 1)
        except Exception as exc:
            logger.warning("Failed to write metrics to TensorBoard: %s", exc)

    def on_train_end(self, trainer: Any) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                logger.info("TensorBoard SummaryWriter closed.")
            except Exception as exc:
                logger.warning("Failed to close TensorBoard SummaryWriter: %s", exc)
            finally:
                self._writer = None
