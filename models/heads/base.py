"""
DeepGuard — models/heads/base.py

Abstract base module for all Vision Transformer classification heads.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseClassifierHead(nn.Module, ABC):
    """Abstract base class for all classifier heads in DeepGuard.

    All custom heads must inherit from this module and implement
    the forward pass.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_features) or sequence
               tensor (batch_size, num_tokens, num_features).

        Returns:
            Output logits of shape (batch_size, num_classes).
        """
