"""
DeepGuard — models/heads/mlp.py

Multi-Layer Perceptron (MLP) classification head module for Vision Transformers.
"""

import torch
import torch.nn as nn
from models.heads.base import BaseClassifierHead


class MLPClassifierHead(BaseClassifierHead):
    """Multi-Layer Perceptron (MLP) classification head.

    Consists of a linear projection, layer normalization, non-linear activation,
    dropout, and final projection layer.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        activation: str = "gelu",
    ) -> None:
        super().__init__(in_features, num_classes)

        if activation.lower() == "gelu":
            act_fn: type[nn.Module] = nn.GELU
        else:
            act_fn = nn.ReLU

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is sequence tokens (batch_size, num_tokens, in_features),
        # extract the CLS token at index 0.
        if x.ndim == 3:
            x = x[:, 0]
        return self.net(x)
