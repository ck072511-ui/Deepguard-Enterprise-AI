"""
DeepGuard — Models package.

Contains Vision Transformer model architecture definitions,
custom classification heads, and backbone configurations.

Packages:
    models.architectures — Complete ViT model classes
    models.backbones     — Pre-trained backbone wrappers (timm integration)
    models.heads         — Classification head modules
"""

from models.architectures.vit import ViTClassifier
from models.config import FullModelConfig
from models.factory import ModelFactory

__all__ = ["ViTClassifier", "FullModelConfig", "ModelFactory"]

