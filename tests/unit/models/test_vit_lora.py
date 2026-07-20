"""
DeepGuard — tests/unit/models/test_vit_lora.py

Unit tests for LoRA adapters, model configuration, and ViTClassifier strategy integration.
"""

import pytest
import torch
import torch.nn as nn
from models.config import FullModelConfig
from models.factory import ModelFactory
from models.architectures.vit import ViTClassifier
from models.backbones.lora import LoRALinear, inject_lora


def test_model_config_parsing() -> None:
    """Test loading and validating config using FullModelConfig."""
    raw_config = {
        "model": {
            "name": "vit_tiny_patch16_224",
            "pretrained": False,
            "num_classes": 2,
            "head": {
                "type": "mlp",
                "hidden_dim": 256,
                "dropout": 0.2,
                "activation": "gelu"
            },
            "fine_tuning": {
                "strategy": "lora",
                "lora_r": 4,
                "lora_alpha": 8.0
            }
        }
    }

    config = FullModelConfig(**raw_config)
    assert config.model.name == "vit_tiny_patch16_224"
    assert config.model.pretrained is False
    assert config.model.head.type == "mlp"
    assert config.model.head.hidden_dim == 256
    assert config.model.fine_tuning.strategy == "lora"
    assert config.model.fine_tuning.lora_r == 4
    assert config.model.fine_tuning.lora_alpha == 8.0


def test_lora_linear_layer() -> None:
    """Test custom LoRALinear layer parameter properties and forward pass."""
    in_features, out_features = 64, 128
    r, alpha = 4, 8.0
    original = nn.Linear(in_features, out_features)

    # Check requires_grad is originally True
    assert original.weight.requires_grad is True

    # Wrap in LoRALinear
    lora_layer = LoRALinear(original, r=r, alpha=alpha)

    # Verify original weights are frozen
    assert lora_layer.original_linear.weight.requires_grad is False
    if lora_layer.original_linear.bias is not None:
        assert lora_layer.original_linear.bias.requires_grad is False

    # Verify LoRA adapter parameters are trainable and match dimensions
    assert lora_layer.lora_A.requires_grad is True
    assert lora_layer.lora_B.requires_grad is True
    assert lora_layer.lora_A.shape == (r, in_features)
    assert lora_layer.lora_B.shape == (out_features, r)

    # Run forward pass on 3D sequence tensor (batch, tokens, features)
    x = torch.randn(2, 10, in_features)
    out = lora_layer(x)
    assert out.shape == (2, 10, out_features)


def test_lora_injection_in_backbone() -> None:
    """Test recursive LoRA adapter injection into self-attention projections."""
    # Instantiate a mini ViTClassifier model using the factory
    raw_config = {
        "model": {
            "name": "vit_tiny_patch16_224",
            "pretrained": False,
            "num_classes": 2,
            "head": {"type": "linear"},
            "fine_tuning": {
                "strategy": "lora",
                "lora_r": 4,
                "lora_alpha": 8.0
            }
        }
    }
    model = ModelFactory.create_model(raw_config)

    # Verify that the self-attention projections have been wrapped with LoRALinear
    qkv_wrapped = False
    proj_wrapped = False
    for name, module in model.backbone.named_modules():
        if name.endswith("attn.qkv"):
            assert isinstance(module, LoRALinear)
            qkv_wrapped = True
        if name.endswith("attn.proj"):
            assert isinstance(module, LoRALinear)
            proj_wrapped = True

    assert qkv_wrapped
    assert proj_wrapped

    # Run forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 2)


def test_vit_classifier_lora_freezing() -> None:
    """Verify that only LoRA parameters and classification head parameters are trainable."""
    model = ViTClassifier(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        num_classes=2,
        head_type="linear",
        fine_tuning_strategy="lora",
        lora_r=4,
        lora_alpha=8.0
    )

    # Head parameters should be trainable
    for param in model.head.parameters():
        assert param.requires_grad is True

    # Check backbone parameters
    for name, param in model.backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False
