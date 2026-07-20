"""
DeepGuard — scripts/benchmark.py

Benchmark latency and throughput comparison between PyTorch and ONNX Runtime.
"""

import time
import numpy as np
import torch
import yaml
from pathlib import Path
import onnxruntime as ort
from models.factory import ModelFactory

def run_benchmark() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "model_config.yaml"
    with open(config_path) as f:
        model_config = yaml.safe_load(f)

    # ── PyTorch Benchmarking ──────────────────────────────────────────────────
    print("Preparing PyTorch model...")
    pytorch_model = ModelFactory.create_model(model_config)
    pytorch_model.eval()

    # Load weights if available
    ckpt_path = project_root / "weights" / "best_model.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            pytorch_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            pytorch_model.load_state_dict(checkpoint)
        print("PyTorch model loaded with weights.")

    dummy_input_np = np.random.randn(8, 3, 224, 224).astype(np.float32)
    dummy_input_torch = torch.from_numpy(dummy_input_np)

    print("Warming up PyTorch model...")
    for _ in range(5):
        with torch.no_grad():
            _ = pytorch_model(dummy_input_torch)

    print("Benchmarking PyTorch model (20 runs, batch_size=8)...")
    torch_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = pytorch_model(dummy_input_torch)
        torch_times.append(time.perf_counter() - t0)

    avg_torch_time = np.mean(torch_times) * 1000  # ms
    std_torch_time = np.std(torch_times) * 1000  # ms
    print(f"PyTorch Average Latency: {avg_torch_time:.2f} ms ± {std_torch_time:.2f} ms (per batch of 8)")

    # ── ONNX Runtime Benchmarking ─────────────────────────────────────────────
    onnx_path = project_root / "weights" / "model.onnx"
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}. Please export it first.")
        return

    print("\nPreparing ONNX Runtime session...")
    ort_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )
    input_name = ort_session.get_inputs()[0].name

    print("Warming up ONNX Runtime model...")
    for _ in range(5):
        _ = ort_session.run(None, {input_name: dummy_input_np})

    print("Benchmarking ONNX Runtime (20 runs, batch_size=8)...")
    ort_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {input_name: dummy_input_np})
        ort_times.append(time.perf_counter() - t0)

    avg_ort_time = np.mean(ort_times) * 1000  # ms
    std_ort_time = np.std(ort_times) * 1000  # ms
    print(f"ONNX Runtime Average Latency: {avg_ort_time:.2f} ms ± {std_ort_time:.2f} ms (per batch of 8)")

    # Speedup ratio
    speedup = avg_torch_time / avg_ort_time
    print(f"\n[INFO] ONNX Runtime is {speedup:.2f}x faster than PyTorch on CPU!")


if __name__ == "__main__":
    run_benchmark()
