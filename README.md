# HairAPP — Hair Type Classifier

An ML-powered hair type classifier that identifies hair types (Type 1–4: Straight, Wavy, Curly, Kinky) from photos or a live webcam feed.

## Features

- **Transfer Learning** — EfficientNet-B0 fine-tuned for hair type classification
- **Training Pipeline** — Full training loop with early stopping, cosine LR scheduling, and checkpoint saving
- **Evaluation** — Per-class precision/recall/F1 reports and confusion matrix
- **Image Prediction** — Classify a single image or a batch of images from the command line
- **Live Camera** — Real-time hair type detection via webcam with a visual overlay

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Organize your images in a folder structure:

```
data/
  Straight/
  Wavy/
  Curly/
  Kinky/
```

## Usage

### Train
```bash
python train.py --data_dir ./data --epochs 25 --batch_size 32
```

### Evaluate
```bash
python evaluate.py --data_dir ./data --checkpoint checkpoints/best_model.pth
```

### Predict on an image
```bash
python predict.py --image path/to/photo.jpg
```

### Live camera detection
```bash
python live_camera.py
```
**Controls:** `R` = scan, `S` = screenshot, `Q` = quit

## Architecture

- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Head:** Dropout → Linear(1280→256) → ReLU → Dropout → Linear(256→4)
- **Classes:** Straight, Wavy, Curly, Kinky (Type 1–4)
