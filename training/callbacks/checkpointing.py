"""
DeepGuard — training/callbacks/checkpointing.py

Model checkpointing callback module to save weights during training.
"""

import logging
from pathlib import Path
from typing import Any
import torch
from training.callbacks.base import Callback

logger = logging.getLogger(__name__)


class ModelCheckpointCallback(Callback):
    """Automatically saves model checkpoints and tracks top-k best models.

    Args:
        output_dir:        Directory to save weights files.
        monitor:           Metric key name to monitor (default: 'val/auc_roc').
        mode:              'min' | 'max' (default: 'max').
        save_top_k:        Number of top-performing checkpoints to keep on disk.
        filename_template: Formatting template for checkpoints file naming.
    """

    def __init__(
        self,
        output_dir: str | Path,
        monitor: str = "val/auc_roc",
        mode: str = "max",
        save_top_k: int = 3,
        filename_template: str = "epoch={epoch:03d}-auc={val_auc_roc:.4f}",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.mode = mode.lower()
        self.save_top_k = save_top_k
        self.filename_template = filename_template

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_k_checkpoints: list[tuple[float, Path]] = []
        self.best_val = float("inf") if self.mode == "min" else float("-inf")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        val = metrics.get(self.monitor)
        if val is None:
            logger.warning(
                "Checkpoint monitor metric '%s' not present. Skipping checkpointer.",
                self.monitor,
            )
            return

        # Prepare formatting arguments (replace slashes in keys for safety)
        format_dict = {k.replace("/", "_"): v for k, v in metrics.items()}
        format_dict["epoch"] = epoch + 1

        cleaned_template = self.filename_template.replace("/", "_")
        try:
            filename = cleaned_template.format(**format_dict) + ".pt"
        except Exception:
            filename = f"epoch={epoch + 1:03d}.pt"

        filepath = self.output_dir / filename

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": (
                trainer.optimizer.state_dict() if trainer.optimizer else None
            ),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict() if trainer.scheduler else None
            ),
            "metrics": metrics,
        }

        # Check if it's the absolute best model
        if self.mode == "min":
            is_best = val < self.best_val
        else:
            is_best = val > self.best_val

        if is_best:
            self.best_val = val
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                "New best model checkpoint saved to '%s' | %s=%.4f",
                best_path.name,
                self.monitor,
                val,
            )

        # Save current epoch checkpoint
        torch.save(checkpoint, filepath)
        logger.info("Saved epoch checkpoint: '%s'", filepath.name)

        # Track top-k
        self.top_k_checkpoints.append((val, filepath))
        reverse = self.mode == "max"
        self.top_k_checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Remove oldest/worst if count exceeds save_top_k
        if len(self.top_k_checkpoints) > self.save_top_k:
            _, worst_path = self.top_k_checkpoints.pop()
            if worst_path.exists():
                try:
                    worst_path.unlink()
                    logger.debug("Removed low-ranking checkpoint: '%s'", worst_path.name)
                except Exception as exc:
                    logger.debug("Failed to delete checkpoint file '%s': %s", worst_path, exc)
