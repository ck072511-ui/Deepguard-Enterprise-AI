"""
DeepGuard — training/trainer.py

Core training and validation loop manager with mixed precision support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from training.callbacks.base import Callback
from training.evaluators.metrics import MetricsEvaluator

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer coordinating ViT model training, evaluation, and logging.

    Handles mixed precision training, gradient clipping, schedulers, device
    mapping, and execution callbacks hooks.

    Args:
        model:              Vision Transformer classifier model (ViTClassifier).
        loss_fn:            Loss module.
        optimizer:          Wrapped optimizer.
        scheduler:          Optional learning rate scheduler.
        callbacks:          List of Callback instances.
        device:             Compute target ('cpu', 'cuda', or 'auto').
        use_amp:            Enable Automatic Mixed Precision (default: True).
        gradient_clip_norm: Maximum gradient norm value for clipping (default: 1.0).
        output_dir:         Output directory for checkpoint storage (default: './weights').
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        callbacks: list[Callback] | None = None,
        device: str = "auto",
        use_amp: bool = True,
        gradient_clip_norm: float | None = 1.0,
        output_dir: str | Path = "./weights",
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_clip_norm = gradient_clip_norm
        self.output_dir = Path(output_dir)

        # Resolve device
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        if hasattr(self.loss_fn, "weight") and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(self.device)

        # Scaler for Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.stop_training = False
        self.global_step = 0

        logger.info(
            "Trainer initialized | device=%s amp=%s clip_norm=%s output_dir=%s",
            self.device,
            self.use_amp,
            gradient_clip_norm,
            output_dir,
        )

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        epochs: int = 10,
        limit_batches: int | None = None,
    ) -> dict[str, float]:
        """Run the complete training and validation cycle.

        Args:
            train_loader:  Data loader for training split.
            val_loader:    Data loader for validation split.
            epochs:        Number of training epochs.
            limit_batches: Limit number of batches per epoch (for quick dry-runs).

        Returns:
            Dictionary of metrics calculated during the final epoch.
        """
        # Training begin hook
        for callback in self.callbacks:
            callback.on_train_begin(self)

        final_metrics: dict[str, float] = {}

        for epoch in range(epochs):
            if self.stop_training:
                logger.info("Training termination requested by early stopping.")
                break

            logger.info("Epoch %d/%d starting...", epoch + 1, epochs)

            # Epoch begin hook
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            # Train epoch
            train_metrics = self._train_epoch(train_loader, epoch, limit_batches)

            # Validate epoch
            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self._evaluate(val_loader, limit_batches)

            # Merge epoch metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Update final metrics reference
            final_metrics = epoch_metrics

            # Log summary
            log_str = f"Epoch {epoch + 1:03d} summary | " + " | ".join(
                f"{k}: {v:.4f}" for k, v in epoch_metrics.items()
            )
            logger.info(log_str)

            # Epoch end hook
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, epoch_metrics)

        # Training end hook
        for callback in self.callbacks:
            callback.on_train_end(self)

        return final_metrics

    def _train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        epoch: int,
        limit_batches: int | None = None,
    ) -> dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        running_loss = 0.0
        processed_batches = 0

        pbar = tqdm(
            desc=f"Epoch {epoch + 1:03d} [Train]",
            total=limit_batches or len(loader),
            leave=False,
        )

        for batch_idx, (images, targets) in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Batch begin hook
            for callback in self.callbacks:
                callback.on_train_batch_begin(self, batch_idx)

            self.optimizer.zero_grad()

            # Forward pass with optional autocast
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)

            # Backward pass and optimizer step
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                self.optimizer.step()

            # Step scheduler if present (step-based learning rate decay)
            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item()
            processed_batches += 1
            self.global_step += 1

            batch_logs = {"loss": loss.item()}
            if self.scheduler is not None:
                try:
                    batch_logs["lr"] = self.scheduler.get_last_lr()[0]
                except Exception:
                    pass

            # Batch end hook
            for callback in self.callbacks:
                callback.on_train_batch_end(self, batch_idx, batch_logs)

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        avg_loss = running_loss / max(1, processed_batches)
        metrics = {"train/loss": avg_loss}
        if self.scheduler is not None:
            try:
                metrics["train/lr"] = self.scheduler.get_last_lr()[0]
            except Exception:
                pass
        return metrics

    @torch.no_grad()
    def _evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        limit_batches: int | None = None,
    ) -> dict[str, float]:
        """Run validation evaluation over validation loader."""
        self.model.eval()
        running_loss = 0.0
        processed_batches = 0

        all_targets: list[int] = []
        all_probs: list[float] = []

        pbar = tqdm(
            desc="Evaluating",
            total=limit_batches or len(loader),
            leave=False,
        )

        for batch_idx, (images, targets) in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)

            running_loss += loss.item()
            processed_batches += 1

            # Softmax probabilities for the FAKE positive class (1)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            pbar.update(1)

        pbar.close()

        avg_loss = running_loss / max(1, processed_batches)

        # Compute full validation metrics using MetricsEvaluator
        eval_metrics = MetricsEvaluator.evaluate(all_targets, all_probs)

        return {
            "val/loss": avg_loss,
            "val/accuracy": eval_metrics["accuracy"],
            "val/precision": eval_metrics["precision"],
            "val/recall": eval_metrics["recall"],
            "val/f1_score": eval_metrics["f1_score"],
            "val/auc_roc": eval_metrics["auc_roc"],
        }

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Map device string option to concrete torch.device."""
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
