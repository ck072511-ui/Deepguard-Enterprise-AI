"""
DeepGuard — models/heads/attention_pool.py

Multi-head attention pooling classifier head module for Vision Transformers.
"""

import torch
import torch.nn as nn
from models.heads.base import BaseClassifierHead


class AttentionPoolingHead(BaseClassifierHead):
    """Attention pooling classification head.

    Uses a learnable query vector to pool feature embeddings from all sequence
    tokens via multi-head cross-attention. This can capture richer spatial
    context than relying solely on the CLS token.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(in_features, num_classes)

        self.query = nn.Parameter(torch.zeros(1, 1, in_features))
        self.mha = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features, num_classes)

        # Initialize the query vector
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is already pooled (B, C), unsqueeze to (B, 1, C)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)

        # Multi-head attention query pooling
        # pooled shape: (batch_size, 1, in_features)
        pooled, _ = self.mha(query, x, x)
        pooled = pooled.squeeze(1)

        return self.fc(pooled)
