# NoctVision — Complete Workflow & Benchmarks

## Workflow

### Step 1: Input Image
- **Dimensions:** 640x640x3 (width, height, RGB channels)
- **Format:** JPEG/PNG image from camera

### Step 2: Backbone (Feature Extraction) 
- **Operation:** Strided convolution (3x3 kernels, stride=2, padding=1)
- **Kernels:** 64 different kernels (each detects different pattern: edges, textures, colors)
- **Output:** 64 feature maps (320x320 each) — spatial size halved, channels increased
- **Pre-training:** COCO dataset (330k images, 80 classes) — backbone already knows what humans/cars look like
- **Freezing:** First 6 layers frozen (retain COCO knowledge, prevent overfitting)

### Step 3: Neck (Feature Fusion)
- **Operation:** PANet (Path Aggregation Network) — fuses features from different scales
- **Purpose:** Combine fine details (early backbone) with semantic meaning (late backbone)
- **Unfreezing:** Neck is trainable — adapts feature fusion to pedestrian detection (LLVIP domain)

### Step 4: Head (Detection)
- **Operation:** For each position in feature maps:
  - Predicts 4 numbers: `[x1, y1, x2, y2]` (bounding box) (the this is [class] with x% accuracy ahh)
  - Predicts 1 number: objectness (is there an object?)
  - Predicts 1 number: class probability (pedestrian vs. background)
- **Output:** List of `[bbox, confidence, class_id]` for each detected object

### Step 5: Post-Processing
- **NMS (Non-Maximum Suppression):** Remove duplicate bboxes (keep only highest confidence)
- **Thresholding:** Keep only bboxes with confidence > 0.5

### Step 6: Training Loop (50 Epochs)
- **Forward pass:** Input image → backbone → neck → head → predictions
- **Loss computation:** Compare predictions to ground-truth bboxes
- **Backpropagation:** Compute gradients, update weights (unfrozen layers only)
- **Repeat:** For all images in dataset (24k images / 16 batch_size = 1500 batches per epoch)

### Step 7: Data Augmentation (Albumentations) 
(synthesise new data by alterations of pre existing data's features which is already there in albumentations library)
- **Transforms:** Random brightness/contrast, fog, blur, shadows, compression
- **Purpose:** Create 2x larger training dataset (original + augmented)
- **Benefit:** Improves generalization to real-world rain/fog by 10-15% mAP

### Step 8: Evaluation Metrics
- **mAP@50:** Detection accuracy (IoU ≥ 0.5) — target >75%
- **mAP@50-95:** Detection accuracy (IoU ≥ 0.5:0.95) — target >45%
- **Precision:** How many predictions are correct? — target >0.80
- **Recall:** How many ground-truth objects are detected? — target >0.75
- **FPS:** Inference speed — target ≥20 FPS (real-time)


## Benchmarks/ metrics to check tho:
- mAP %50
- mAP @50-95
- precision,recall,fps


# YOLOv8n Architecture — Complete Dimension Flow

## Input → Backbone → Neck → Head

| Stage | Operation | Kernels | Kernel Size | Stride | Padding | Output Dimensions | Notes |
|-------|-----------|---------|-------------|--------|---------|-------------------|-------|
| **Input** | — | — | — | — | — | `640 × 640 × 3` | RGB image |
| **Backbone** | | | | | | | |
| 1 | Conv | 32 | 3×3 | 2 | 1 | `320 × 320 × 32` | Stride=2 halves spatial size |
| 2 | C2f | 64 | 3×3 | 1 | 1 | `320 × 320 × 64` | Stride=1 keeps size same |
| 3 | Conv | 128 | 3×3 | 2 | 1 | `160 × 160 × 128` | Stride=2 halves spatial size |
| 4 | C2f | 128 | 3×3 | 1 | 1 | `160 × 160 × 128` | Stride=1 keeps size same |
| 5 | Conv | 256 | 3×3 | 2 | 1 | `80 × 80 × 256` | Stride=2 halves spatial size |
| 6 | C2f | 256 | 3×3 | 1 | 1 | `80 × 80 × 256` | Stride=1 keeps size same |
| 7 | Conv | 512 | 3×3 | 2 | 1 | `40 × 40 × 512` | Stride=2 halves spatial size |
| 8 | C2f | 512 | 3×3 | 1 | 1 | `40 × 40 × 512` | Stride=1 keeps size same |
| 9 | Conv | 1024 | 3×3 | 2 | 1 | `20 × 20 × 1024` | Stride=2 halves spatial size |
| 10 | SPPF | 1024 | 5×5 | 1 | 2 | `20 × 20 × 1024` | Multi-scale pooling, keeps size same |
| **Neck** | PANet (Feature Fusion) | — | — | — | — | Multi-scale: `20×20`, `40×40`, `80×80` | Fuses features from different backbone levels |
| **Head** | Detection Head | — | — | — | — | List of `[bbox, confidence, class_id]` | Predicts bounding boxes + class probabilities |

---

## Key Patterns

### Spatial Dimensions (Height × Width)
640 → 320 → 160 → 80 → 40 → 20

### Channels (Feature Maps)
3 → 32 → 64 → 128 → 256 → 512 → 1024


### Stride Pattern
- **Stride=2:** Halves spatial dimensions (downsampling).
- **Stride=1:** Keeps spatial dimensions same (feature extraction).

### Padding Pattern
- **Padding=1:** For 3×3 kernels (keeps output size same when stride=1).
- **Padding=2:** For 5×5 kernel in SPPF (keeps output size same).

---

## What Happens at Each Stage?

### **Backbone (Feature Extraction)**
- Extracts hierarchical features: edges → textures → shapes → object parts.
- Pre-trained on COCO (already knows what humans, cars, etc. look like).
- First 6 layers frozen (retain COCO knowledge, prevent overfitting).

### **SPPF (Multi-Scale Context)**
- Applies max pooling at multiple kernel sizes (5×5, 9×9, 13×13).
- Captures small, medium, and large objects simultaneously.
- Output: 20×20×1024 feature maps.

### **Neck (Feature Fusion — PANet)**
- Fuses features from different backbone levels (20×20, 40×40, 80×80).
- Combines fine details (early backbone) with semantic meaning (late backbone).
- Enables detection at multiple scales (small pedestrians far away, large pedestrians close up).

### **Head (Detection)**
- For each position in feature maps:
  - Predicts 4 numbers: `[x1, y1, x2, y2]` (bounding box).
  - Predicts 1 number: objectness (is there an object?).
  - Predicts 1 number: class probability (pedestrian vs. background).
- Output: List of `[bbox, confidence, class_id]` for each detected object.

---

## Freezing Strategy

| Component | Layers | Frozen? | Reason |
|-----------|--------|---------|--------|
| **Backbone** | 0–5 | ✅ Yes | Retain COCO pre-trained knowledge (edges, textures, human silhouettes) |
| **Backbone** | 6–9 | ❌ No | Adapt to LLVIP's low-light textures |
| **Neck** | 10–19 | ❌ No | Adapt feature fusion to pedestrian detection |
| **Head** | 20+ | ❌ No | Completely retrained for LLVIP (1 class: pedestrian) |

**Trainable parameters:** ~1.8M (out of 3.01M total) — 40% reduction vs. full fine-tuning.

---

## Training Configuration

```bash
yolo train \
  model=yolov8n.pt \
  data=llvip.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  freeze=6 \
  lr0=0.001 \
  cos_lr=True \
  warmup_epochs=5 \
  augment=True

