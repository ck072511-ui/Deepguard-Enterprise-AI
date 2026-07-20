"""
DeepGuard — models/backbones/lora.py

Custom Low-Rank Adaptation (LoRA) module and model injection utilities.
Allows parameter-efficient fine-tuning of Vision Transformer layers.
"""

import math
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear layer with a low-rank adapter."""

    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # Low-rank matrices A and B
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Reset parameters like standard LoRA initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize LoRA adapter weights (A is Kaiming uniform, B is zero)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original projection
        out = self.original_linear(x)

        # Low-rank path: (x @ A.T) @ B.T
        # Handles any batch or sequence dimensions
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()

        return out + self.scaling * lora_out


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0) -> int:
    """Scan and replace target Linear layers in a model's self-attention with LoRA layers.

    Args:
        model: PyTorch module (e.g. ViT backbone).
        r: Low-rank dimension.
        alpha: LoRA scaling factor.

    Returns:
        Number of layers wrapped with LoRA adapters.
    """
    injected_count = 0

    # Locate the blocks / layers in timm models
    for name, module in model.named_modules():
        # Look for the Multi-head Attention (MHA) module layers
        # In timm ViTs, they are typically named block.attn.qkv
        # We can also wrap block.attn.proj if desired
        if name.endswith("attn"):
            if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear) and not isinstance(module.qkv, LoRALinear):
                module.qkv = LoRALinear(module.qkv, r=r, alpha=alpha)
                injected_count += 1
            if hasattr(module, "proj") and isinstance(module.proj, nn.Linear) and not isinstance(module.proj, LoRALinear):
                module.proj = LoRALinear(module.proj, r=r, alpha=alpha)
                injected_count += 1

    logger.info("Injected LoRA adapters (rank=%d, alpha=%.1f) into %d attention layers.", r, alpha, injected_count)
    return injected_count
