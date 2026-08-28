"""
Model Export Pipeline for NoctVision.
Exports trained PyTorch YOLOv8n weights (.pt) to ONNX (.onnx) and TFLite (.tflite) for edge deployment.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_model(weights_path: str, output_dir: str = "exports", imgsz: int = 640, int8: bool = False):
    """
    Exports PyTorch YOLOv8n model to ONNX and TFLite formats.
    """
    weights = Path(weights_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not weights.exists():
        print(f"Warning: Weights file '{weights_path}' not found. Please provide a valid trained model path.")
        return

    print(f"Loading PyTorch model from: {weights}")
    model = YOLO(str(weights))

    # 1. Export to ONNX
    print("\n[1/2] Exporting model to ONNX format...")
    onnx_file = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True
    )
    print(f"ONNX export completed: {onnx_file}")

    # 2. Export to TFLite
    print("\n[2/2] Exporting model to TFLite format...")
    tflite_file = model.export(
        format="tflite",
        imgsz=imgsz,
        int8=int8
    )
    print(f"TFLite export completed: {tflite_file}")

    print(f"\nAll exported models are available in the exports directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export NoctVision YOLOv8n Model to ONNX and TFLite")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to trained PyTorch weights (.pt)")
    parser.add_argument("--output-dir", type=str, default="exports", help="Export directory for ONNX & TFLite")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization for TFLite edge deployment")
    args = parser.parse_args()

    export_model(args.weights, args.output_dir, args.imgsz, args.int8)
