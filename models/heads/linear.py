"""
DeepGuard — models/heads/linear.py

Linear classifier head module for Vision Transformers.
"""

import torch
import torch.nn as nn
from models.heads.base import BaseClassifierHead


class LinearClassifierHead(BaseClassifierHead):
    """Standard linear classification head.

    Applies optional dropout followed by a single linear projection layer.
    Handles both pooled features of shape (B, C) and token sequence features
    of shape (B, N, C) by using the first token (CLS) as input.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__(in_features, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is sequence tokens (batch_size, num_tokens, in_features),
        # extract the CLS token at index 0.
        if x.ndim == 3:
            x = x[:, 0]
        x = self.dropout(x)
        return self.fc(x)
