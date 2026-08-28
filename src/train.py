"""
Training Script for NoctVision (YOLOv8n).
Trains YOLOv8n on the processed visible-camera LLVIP dataset using custom Albumentations augmentations.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_noctvision(config_path: str, weights: str = "yolov8n.pt", epochs: int = 100, batch: int = 16, imgsz: int = 640):
    """
    Launches YOLOv8n training pipeline.
    """
    print(f"--- NoctVision Training Pipeline ---")
    print(f"Base Model: {weights}")
    print(f"Config: {config_path}")
    print(f"Epochs: {epochs} | Batch Size: {batch} | Img Size: {imgsz}")

    model = YOLO(weights)
    
    results = model.train(
        data=config_path,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project="models/runs",
        name="noctvision_yolov8n",
        exist_ok=True,
        save=True,
        plots=True
    )
    
    print(f"Training completed successfully. Best weights saved to models/runs/noctvision_yolov8n/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n on NoctVision Dataset")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to dataset YAML config")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    train_noctvision(args.config, args.weights, args.epochs, args.batch, args.imgsz)
