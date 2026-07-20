"""
DeepGuard — training/schedulers/cosine_warmup.py

Learning rate scheduler combining linear warmup with cosine annealing.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithWarmupLR(_LRScheduler):
    """Cosine Annealing learning rate scheduler with linear warmup steps.

    Transitions from a minimum learning rate to the configured base learning
    rate linearly over warmup steps, then decays back to the minimum learning
    rate following a cosine curve over the remaining steps.

    Args:
        optimizer:    Wrapped optimizer.
        warmup_steps: Number of iterations for linear learning rate warmup.
        total_steps:  Total training iterations.
        min_lr:       Minimum target learning rate (default: 1e-6).
        last_epoch:   The index of the last epoch/step (default: -1).
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute the current step's learning rate for all parameter groups."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / float(self.warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * alpha
                for base_lr in self.base_lrs
            ]

        if step >= self.total_steps:
            # Clamped to min_lr after training completes
            return [self.min_lr for _ in self.base_lrs]

        # Cosine annealing
        progress = (step - self.warmup_steps) / float(
            self.total_steps - self.warmup_steps
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * coeff
            for base_lr in self.base_lrs
        ]
