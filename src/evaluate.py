"""
Evaluation and Benchmarking Script for NoctVision.
Evaluates model mAP, precision, recall, and edge inference latency (PyTorch vs ONNX vs TFLite).
"""

import time
import argparse
from pathlib import Path
from ultralytics import YOLO


def evaluate_pytorch(weights_path: str, data_config: str, imgsz: int = 640):
    """Evaluates PyTorch model performance on test set."""
    print(f"\n--- Evaluating PyTorch Model ({weights_path}) ---")
    model = YOLO(weights_path)
    metrics = model.val(data=data_config, imgsz=imgsz, split="test", project="results", name="eval_pytorch", exist_ok=True)
    print(f"mAP50-95: {metrics.box.map:.4f} | mAP50: {metrics.box.map50:.4f}")
    return metrics


def benchmark_latency(model_path: str, runs: int = 100, imgsz: int = 640):
    """Benchmarks inference latency for edge deployment models."""
    print(f"\n--- Benchmarking Latency for {model_path} ({runs} runs) ---")
    model = YOLO(model_path, task="detect")
    
    # Warmup
    print("Warming up model...")
    import numpy as np
    dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        _ = model(dummy_img, verbose=False)
        
    start_time = time.time()
    for _ in range(runs):
        _ = model(dummy_img, verbose=False)
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / runs) * 1000
    fps = 1000 / avg_latency_ms
    print(f"Average Latency: {avg_latency_ms:.2f} ms | FPS: {fps:.1f}")
    return avg_latency_ms, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NoctVision Models")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to model weights")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--benchmark", action="store_true", help="Run edge latency benchmarks")
    args = parser.parse_args()

    if Path(args.weights).exists():
        evaluate_pytorch(args.weights, args.config)
        if args.benchmark:
            benchmark_latency(args.weights)
    else:
        print(f"Weights file '{args.weights}' not found. Train a model or place valid weights in models/")
