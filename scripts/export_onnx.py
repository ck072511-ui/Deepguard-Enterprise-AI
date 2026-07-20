"""
DeepGuard — scripts/export_onnx.py

Export PyTorch ViT model checkpoint to ONNX format.
"""

import argparse
import logging
from pathlib import Path
import torch
import yaml
from models.factory import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch ViT model checkpoint to ONNX format")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/best_model.pt",
        help="Path to the PyTorch checkpoint file (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/model.onnx",
        help="Output path for the exported ONNX model weights",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "model_config.yaml"
    with open(config_path) as f:
        model_config = yaml.safe_load(f)

    logger.info("Initializing ViT model architecture from config...")
    model = ModelFactory.create_model(model_config)
    model.eval()

    # Load weights if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        logger.info("Loading PyTorch model weights from checkpoint: %s", args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Weights loaded successfully.")
    else:
        logger.warning(
            "Checkpoint '%s' not found. Exporting with randomly initialized weights.",
            args.checkpoint,
        )

    # Prepare export arguments
    onnx_cfg = model_config.get("onnx", {})
    opset_version = onnx_cfg.get("opset_version", 17)

    # Standard input size: (batch_size, channels, height, width) -> (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting model to ONNX format (opset_version=%d)...", opset_version)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    logger.info("Successfully exported ONNX weights to '%s'", output_path)


if __name__ == "__main__":
    main()
