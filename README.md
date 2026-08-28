# NoctVision: Low-Light Pedestrian Detection with YOLOv8n

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://docs.ultralytics.com/)
[![Albumentations](https://img.shields.io/badge/Augmentations-Albumentations-orange.svg)](https://albumentations.ai/)
[![Export](https://img.shields.io/badge/Export-ONNX%20%7C%20TFLite-purple.svg)](https://onnx.ai/)

**NoctVision** is a light-weight ML vision pipeline designed for low-light pedestrian detection using **YOLOv8n**. It targets night-time and challenging environmental conditions trained exclusively on visible-spectrum camera images from the **LLVIP** dataset. Custom Albumentations augmentations (low-light simulation, synthetic fog, rain, and sensor noise) are applied to boost robust detection performance before exporting optimized models to **ONNX** and **TFLite** formats for edge deployment.

---

## 📁 Project Structure

```
NoctVision/
├── data/
│   ├── raw/                  # Raw LLVIP dataset (visible images & annotations)
│   └── processed/            # Processed YOLO-format dataset split (images & labels)
│       ├── images/           # train / val / test splits
│       └── labels/           # train / val / test YOLO labels
├── configs/
│   ├── data.yaml             # YOLOv8 dataset configuration
│   └── train.yaml            # Hyperparameters and training configuration
├── src/
│   ├── __init__.py
│   ├── augmentations.py      # Custom Albumentations pipeline (rain, fog, low-light)
│   ├── dataset.py            # LLVIP dataset conversion & structure builder
│   ├── train.py              # YOLOv8n training pipeline
│   ├── export.py             # Model exporter (PyTorch -> ONNX -> TFLite)
│   └── evaluate.py           # Evaluation & latency benchmarking script
├── models/                   # Trained PyTorch model checkpoints (.pt)
├── exports/                  # Exported models for edge deployment (.onnx, .tflite)
├── results/                  # Validation metrics, confusion matrices, and latency reports
├── requirements.txt          # Python project dependencies
├── .gitignore                # Git exclusions
└── README.md                 # Project documentation
```

---

## ⚡ Quick Start

### 1. Installation

Clone the repository and set up your Python environment:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup (LLVIP Visible Spectrum)

1. Download the [LLVIP Dataset](https://github.com/bupt-ai-cz/LLVIP).
2. Place the visible camera images and annotations in `data/raw/`.
3. Run the processing script to convert and split data into standard YOLO format:

```bash
python src/dataset.py --raw-dir data/raw --output-dir data/processed
```

---

## 🌧️ Custom Albumentations Pipeline

NoctVision implements custom image augmentation routines tailored for nighttime and low-visibility conditions in `src/augmentations.py`:
- **Low-Light / Gamma adjustment**: Simulates extreme dark scenes and varying exposure levels.
- **Environmental conditions**: Synthetic rain drops, low-visibility fog, and motion blur.
- **Sensor Noise**: Low-light ISO noise and Gaussian sensor noise modeling.

Run a quick test of the pipeline:
```bash
python src/augmentations.py
```

---

## 🚀 Training YOLOv8n

Train YOLOv8n on the visible camera dataset using `configs/data.yaml`:

```bash
python src/train.py --config configs/data.yaml --epochs 100 --batch 16 --imgsz 640
```

Trained weights will automatically be saved under `models/runs/noctvision_yolov8n/weights/best.pt`.

---

## 📦 Model Export (ONNX & TFLite)

Export the trained PyTorch checkpoint (`.pt`) to ONNX (`.onnx`) and TFLite (`.tflite`) for deployment on edge platforms (e.g. Raspberry Pi, Jetson Nano, Android edge devices):

```bash
# Standard Export (FP32/FP16)
python src/export.py --weights models/best.pt --output-dir exports

# INT8 Quantized TFLite Export
python src/export.py --weights models/best.pt --output-dir exports --int8
```

Output files produced in `exports/`:
- `exports/best.onnx`
- `exports/best_saved_model/best_float32.tflite` (or INT8 quantized)

---

## 📊 Evaluation & Edge Benchmarking

Evaluate accuracy metrics (mAP@0.5, mAP@0.5:0.95) and benchmark edge latency:

```bash
python src/evaluate.py --weights models/best.pt --config configs/data.yaml --benchmark
```

Metric outputs and plots will be logged under `results/`.

---

## 📜 License

Distributed under the MIT License.
