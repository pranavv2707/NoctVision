"""
Dataset Preparation Script for NoctVision.
Converts raw LLVIP dataset (visible camera images & annotations) into standard YOLO structure.
"""

import os
import shutil
import argparse
from pathlib import Path


def prepare_llvip_visible(raw_dir: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Parses raw LLVIP visible dataset and prepares YOLO format layout.
    
    Structure created in output_dir:
        images/train, images/val, images/test
        labels/train, labels/val, labels/test
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    visible_img_dir = raw_path / "visible" / "train"  # LLVIP visible channel folder
    labels_dir = raw_path / "labels"
    
    print(f"Processing visible camera images from: {raw_path}")
    print(f"Output directory: {output_path}")

    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    print("Dataset directory structure initialized.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LLVIP Visible Dataset for YOLOv8")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw LLVIP dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to output processed dataset")
    args = parser.parse_args()
    
    prepare_llvip_visible(args.raw_dir, args.output_dir)
