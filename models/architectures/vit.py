"""
DeepGuard — models/architectures/vit.py

Vision Transformer classification model wrapping timm backbones.
"""

import logging
import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class ViTClassifier(nn.Module):
    """Vision Transformer classifier wrapping timm feature extractors.

    Supports custom classification heads and flexible fine-tuning freeze
    strategies to allow parameter-efficient training.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 2,
        head_type: str = "linear",
        dropout: float = 0.0,
        hidden_dim: int = 512,
        num_heads: int = 8,
        fine_tuning_strategy: str = "full",
        frozen_layers: list[int] | None = None,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
    ) -> None:
        super().__init__()

        # Initialize backbone with num_classes=0 to act as a feature extractor
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
        )

        self.num_features = self.backbone.num_features
        self.head_type = head_type.lower()
        self.fine_tuning_strategy = fine_tuning_strategy
        self.frozen_layers = frozen_layers or []
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        # Instantiate classification head
        if self.head_type == "linear":
            from models.heads.linear import LinearClassifierHead
            self.head: nn.Module = LinearClassifierHead(
                self.num_features, num_classes, dropout=dropout
            )
        elif self.head_type == "mlp":
            from models.heads.mlp import MLPClassifierHead
            self.head = MLPClassifierHead(
                self.num_features, num_classes, hidden_dim=hidden_dim, dropout=dropout
            )
        elif self.head_type == "attention_pool":
            from models.heads.attention_pool import AttentionPoolingHead
            self.head = AttentionPoolingHead(
                self.num_features, num_classes, num_heads=num_heads, dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

        # Apply layers freezing strategy
        self._apply_fine_tuning_strategy()

    def _apply_fine_tuning_strategy(self) -> None:
        """Freeze model weights based on selected fine-tuning strategy."""
        strategy = self.fine_tuning_strategy.lower()

        # By default, make everything trainable
        for param in self.parameters():
            param.requires_grad = True

        if strategy == "head_only":
            # Freeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Fine-tuning strategy: head_only. Backbone frozen.")

        elif strategy == "lora":
            # Freeze entire backbone first
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Inject LoRA parameters
            from models.backbones.lora import inject_lora
            inject_lora(self.backbone, r=self.lora_r, alpha=self.lora_alpha)
            # Ensure the injected parameters are trainable
            logger.info("Fine-tuning strategy: lora. Backbone frozen except injected LoRA weights.")


        elif strategy == "partial":
            # Freeze bottom blocks, keep top blocks trainable
            if hasattr(self.backbone, "blocks"):
                num_blocks = len(self.backbone.blocks)
                # If specific frozen layer indices are not provided, freeze the first 75%
                if self.frozen_layers:
                    freeze_indices = set(self.frozen_layers)
                else:
                    freeze_up_to = int(num_blocks * 0.75)
                    freeze_indices = set(range(freeze_up_to))

                # Freeze block parameters
                for idx, block in enumerate(self.backbone.blocks):
                    if idx in freeze_indices:
                        for param in block.parameters():
                            param.requires_grad = False
                    else:
                        for param in block.parameters():
                            param.requires_grad = True

                # Freeze raw embeddings
                if hasattr(self.backbone, "patch_embed"):
                    for param in self.backbone.patch_embed.parameters():
                        param.requires_grad = False
                if hasattr(self.backbone, "pos_embed"):
                    self.backbone.pos_embed.requires_grad = False
                if hasattr(self.backbone, "cls_token"):
                    self.backbone.cls_token.requires_grad = False

                logger.info(
                    "Fine-tuning strategy: partial. Frozen blocks: %s",
                    sorted(list(freeze_indices)),
                )
            else:
                logger.warning(
                    "Backbone does not have 'blocks' attribute. "
                    "Cannot apply partial freezing; using full fine-tuning."
                )

        elif strategy == "full":
            logger.info("Fine-tuning strategy: full. All weights trainable.")
        else:
            logger.warning(
                "Unknown fine-tuning strategy '%s'; keeping full fine-tuning.",
                self.fine_tuning_strategy,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward_features returns sequence embeddings: (batch_size, num_tokens, num_features)
        features = self.backbone.forward_features(x)
        return self.head(features)
