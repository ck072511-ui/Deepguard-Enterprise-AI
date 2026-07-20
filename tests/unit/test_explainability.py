"""
DeepGuard — tests/unit/test_explainability.py

Unit tests for the explainability engine.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from utils.explainability import ExplainabilityEngine


class MockViTBlock(nn.Module):
    """Mock ViT block to test forward hooks during GradCAM computation."""
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(768, 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockViTBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MockViTBlock()])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Returns sequence feature embeddings shape (B, N, C)
        # N tokens = 1 + 14 * 14 = 197 tokens
        batch_size = x.shape[0]
        # Make tokens of dimension 768
        return torch.ones(batch_size, 197, 768)


class MockViTClassifier(nn.Module):
    """Mock classifier wrapping backbone features and linear projection head."""
    def __init__(self) -> None:
        super().__init__()
        self.backbone = MockViTBackbone()
        self.head = nn.Linear(768, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        # Standard CLS token pooling
        x_cls = features[:, 0]
        return self.head(x_cls)


def test_to_base64_jpeg() -> None:
    # 1. Create a dummy image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    b64 = ExplainabilityEngine.to_base64_jpeg(img)
    assert b64.startswith("data:image/jpeg;base64,")
    
    # 2. Test invalid inputs do not raise unhandled exception
    assert ExplainabilityEngine.to_base64_jpeg(None) == ""


def test_generate_synthetic_map() -> None:
    face_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Check shape
    map_res = ExplainabilityEngine.generate_synthetic_map(face_img, is_fake=True, map_type="heatmap")
    assert map_res.shape == (224, 224, 3)

    map_res_gradcam = ExplainabilityEngine.generate_synthetic_map(face_img, is_fake=False, map_type="gradcam")
    assert map_res_gradcam.shape == (224, 224, 3)


def test_explainability_fallback() -> None:
    face_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # A generic model with no blocks should use fallback
    dummy_model = nn.Linear(10, 2)
    res = ExplainabilityEngine.generate(dummy_model, face_img, label=1, confidence=0.88)
    
    assert "real_probability" in res
    assert "fake_probability" in res
    assert "explanation" in res
    assert "heatmap_b64" in res
    assert "attention_b64" in res
    assert "gradcam_b64" in res
    assert res["fake_probability"] == 0.88
    assert res["real_probability"] == pytest.approx(0.12)


def test_explainability_vit_hooks() -> None:
    face_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    model = MockViTClassifier()

    # Verify GradCAM hook works successfully on dummy ViT structure
    res = ExplainabilityEngine.generate(model, face_img, label=1, confidence=0.92)

    assert "heatmap_b64" in res
    assert "attention_b64" in res
    assert "gradcam_b64" in res
    assert res["fake_probability"] == 0.92
