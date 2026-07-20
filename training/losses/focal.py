"""
DeepGuard — training/losses/focal.py

Focal Loss implementation for handling imbalanced datasets in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for classification tasks.

    Reduces the loss scaling factor for easy-to-classify examples, focusing the
    model's attention on hard examples.

    Formula:
        FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

    Args:
        alpha:     Balancing factor for positive/negative classes (default: 0.25).
        gamma:     Focusing parameter for modulation of easy/hard examples (default: 2.0).
        reduction: 'mean' | 'sum' | 'none'.
        weight:    Per-class manual scale weight tensor.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            logits:  Raw output predictions (batch_size, num_classes) float32.
            targets: Target labels (batch_size,) int64.

        Returns:
            Calculated loss tensor.
        """
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Gather log probabilities and probabilities for target classes
        log_p_target = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        p_target = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Calculate modular focal loss
        loss = -((1.0 - p_target) ** self.gamma) * log_p_target

        # Apply class-specific alpha factor
        if self.alpha >= 0:
            alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
            loss = loss * alpha_t

        # Apply manual class weights if present
        if self.weight is not None:
            w_t = self.weight.to(logits.device).gather(dim=-1, index=targets)
            loss = loss * w_t

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
