"""
DeepGuard — models/config.py

Pydantic configuration models for parsing and validating model configurations.
"""

from typing import Any
from pydantic import BaseModel, Field


class HeadConfig(BaseModel):
    """Configuration for the classification head."""

    type: str = Field("linear", description="Type of classification head: linear, mlp, attention_pool")
    hidden_dim: int = Field(512, description="Hidden dimensions for MLP head")
    dropout: float = Field(0.0, description="Dropout rate in the head")
    activation: str = Field("gelu", description="Activation function for intermediate layers")


class FineTuningConfig(BaseModel):
    """Configuration for model fine-tuning strategy."""

    strategy: str = Field("full", description="Strategy: full, partial, head_only, lora")
    frozen_layers: list[int] = Field(default_factory=list, description="Specific layers to freeze")
    layer_decay: float = Field(0.75, description="Layer-wise learning rate decay")
    lora_r: int = Field(8, description="LoRA rank")
    lora_alpha: float = Field(16.0, description="LoRA alpha scaling factor")



class NormalizationConfig(BaseModel):
    """Configuration for image normalization."""

    mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])


class ModelArchConfig(BaseModel):
    """Configuration for the Vision Transformer model architecture."""

    name: str = Field("vit_tiny_patch16_224", description="timm ViT model name")
    pretrained: bool = Field(True, description="Whether to load pre-trained weights")
    num_classes: int = Field(2, description="Number of output target classes")
    input_size: int = Field(224, description="Input image height and width")
    patch_size: int = Field(16, description="ViT patch size")
    embed_dim: int = Field(768, description="Embedding dimensions")
    depth: int = Field(12, description="Transformer encoder blocks depth")
    num_heads: int = Field(12, description="Number of attention heads")
    mlp_ratio: float = Field(4.0, description="Expansion ratio in MLP layers")
    drop_rate: float = Field(0.0, description="Dropout rate")
    attn_drop_rate: float = Field(0.0, description="Attention dropout rate")
    drop_path_rate: float = Field(0.0, description="Stochastic depth rate")

    head: HeadConfig = Field(default_factory=HeadConfig)
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)


class InferenceConfig(BaseModel):
    """Configuration for running model inference."""

    device: str = Field("auto")
    precision: str = Field("fp32")
    batch_size: int = Field(8)
    confidence_threshold: float = Field(0.5)
    use_onnx: bool = Field(False)
    use_torch_compile: bool = Field(False)
    tta_enabled: bool = Field(False)
    tta_flips: bool = Field(True)
    face_aggregation: str = Field("mean")


class ONNXConfig(BaseModel):
    """Configuration for ONNX export operations."""

    opset_version: int = Field(17)
    dynamic_axes: dict[str, Any] = Field(
        default_factory=lambda: {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    optimize: bool = Field(True)
    quantize: bool = Field(False)
    providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])


class FullModelConfig(BaseModel):
    """Container for the complete model configuration layout."""

    model: ModelArchConfig = Field(default_factory=ModelArchConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    onnx: ONNXConfig = Field(default_factory=ONNXConfig)
