"""
DeepGuard — models/factory.py

Model factory for configuring and constructing ViTClassifier models.
"""

from typing import Any
from models.architectures.vit import ViTClassifier


from models.config import FullModelConfig


class ModelFactory:
    """Factory for building models from configuration dictionaries or objects."""

    @staticmethod
    def create_model(config: dict[str, Any] | FullModelConfig) -> ViTClassifier:
        """Create a ViTClassifier instance based on configuration settings.

        Args:
            config: Full configurations dictionary or FullModelConfig object.

        Returns:
            Configured ViTClassifier instance.
        """
        if isinstance(config, FullModelConfig):
            cfg_dict = config.model_dump()
        else:
            cfg_dict = config

        model_cfg = cfg_dict.get("model", {})
        head_cfg = model_cfg.get("head", {})
        ft_cfg = model_cfg.get("fine_tuning", {})

        return ViTClassifier(
            model_name=model_cfg.get("name", "vit_base_patch16_224"),
            pretrained=model_cfg.get("pretrained", True),
            num_classes=model_cfg.get("num_classes", 2),
            head_type=head_cfg.get("type", "linear"),
            dropout=head_cfg.get("dropout", 0.0),
            hidden_dim=head_cfg.get("hidden_dim", 512),
            num_heads=model_cfg.get("num_heads", 12),
            fine_tuning_strategy=ft_cfg.get("strategy", "full"),
            frozen_layers=ft_cfg.get("frozen_layers", []),
            lora_r=ft_cfg.get("lora_r", 8),
            lora_alpha=ft_cfg.get("lora_alpha", 16.0),
        )


