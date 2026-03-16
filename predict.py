"""
Inference script — predict hair type from a single image or a directory of images.

Usage:
    python predict.py --image photo.jpg
    python predict.py --image ./test_images/          # batch mode (directory)
    python predict.py --image photo.jpg --checkpoint checkpoints/best_model.pth
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from dataset import get_val_transforms, CLASS_NAMES, TYPE_LABELS
from model import create_model


# ── Hair type info database ──────────────────────────────────────────────────

HAIR_TYPE_INFO = {
    0: {
        "type": "Type 1 — Straight",
        "description": "Hair lies flat from root to tip with no natural curl pattern.",
        "characteristics": [
            "Naturally shiny due to oil distribution",
            "Tends to get oily quickly",
            "Difficult to hold a curl",
            "Ranges from fine (1A) to coarse (1C)",
        ],
        "care_tips": [
            "Use lightweight, volumizing products",
            "Avoid heavy conditioners that weigh hair down",
            "Dry shampoo helps manage oiliness",
        ],
    },
    1: {
        "type": "Type 2 — Wavy",
        "description": 'Hair forms an "S" shape, lying somewhere between straight and curly.',
        "characteristics": [
            "S-shaped waves that can be loose or defined",
            "Prone to frizz, especially at higher subtypes",
            "2A: fine, loose waves | 2B: medium, more defined | 2C: thick, almost curly",
        ],
        "care_tips": [
            "Use a sulfate-free shampoo to reduce frizz",
            "Scrunch with a microfiber towel instead of rubbing",
            "Light hold mousse enhances wave definition",
        ],
    },
    2: {
        "type": "Type 3 — Curly",
        "description": "Well-defined, springy curls that form loops or corkscrew shapes.",
        "characteristics": [
            "Clearly defined curl pattern",
            "3A: loose spiral curls | 3B: tight ringlets | 3C: corkscrew curls",
            "Volume and bounce, but prone to dryness",
        ],
        "care_tips": [
            "Deep condition regularly to combat dryness",
            "Detangle with a wide-tooth comb on wet hair",
            "Apply leave-in conditioner and curl cream",
            "Avoid brushing dry hair to prevent breakage and frizz",
        ],
    },
    3: {
        "type": "Type 4 — Coily / Kinky",
        "description": "Tight coils, Z-shaped patterns, or dense, spring-like curls with high shrinkage.",
        "characteristics": [
            "Very tight curl pattern, often appears shorter than actual length",
            "4A: defined S-pattern coils | 4B: Z-shaped bends | 4C: very tight, less defined coils",
            "Most fragile hair type — prone to dryness and breakage",
            "High shrinkage (up to 75%)",
        ],
        "care_tips": [
            "Moisturize frequently with water-based products",
            "Use the LOC/LCO method (Liquid, Oil, Cream)",
            "Protective styles help retain length",
            "Detangle gently with fingers or wide-tooth comb",
            "Sleep on a satin/silk pillowcase",
        ],
    },
}


# ── Prediction ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load the trained model from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names", CLASS_NAMES)

    model = create_model(num_classes=len(class_names), pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model, image_path: str, device: torch.device) -> dict:
    """
    Predict the hair type for a single image.

    Returns a dict with: predicted_class, type_label, confidence, all_probs, info
    """
    transform = get_val_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1).squeeze()

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "type_label": TYPE_LABELS[pred_idx],
        "confidence": confidence,
        "all_probabilities": {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))},
        "info": HAIR_TYPE_INFO[pred_idx],
    }


def print_result(result: dict, image_path: str):
    """Pretty-print a single prediction."""
    print(f"\n{'─'*60}")
    print(f"  📷  {image_path}")
    print(f"{'─'*60}")
    print(f"  🏷️  Prediction : {result['type_label']}")
    print(f"  📊  Confidence : {result['confidence']:.1%}")
    print()
    print(f"  Probabilities:")
    for cls, prob in result["all_probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cls:>10s}  {prob:.1%}  {bar}")
    print()

    info = result["info"]
    print(f"  💡 {info['description']}")
    print()
    print("  Characteristics:")
    for c in info["characteristics"]:
        print(f"    • {c}")
    print()
    print("  Care Tips:")
    for t in info["care_tips"]:
        print(f"    • {t}")
    print(f"{'─'*60}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict hair type from an image")
    parser.add_argument("--image", type=str, required=True, help="Path to a single image or a directory of images")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth", help="Model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Single image or directory
    path = Path(args.image)

    if path.is_dir():
        image_files = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not image_files:
            print(f"  No images found in {path}")
            return
        print(f"\n  Found {len(image_files)} images in {path}\n")
        for img_path in image_files:
            result = predict_image(model, str(img_path), device)
            print_result(result, str(img_path))
    elif path.is_file():
        result = predict_image(model, str(path), device)
        print_result(result, str(path))
    else:
        print(f"  ❌ Path not found: {path}")


if __name__ == "__main__":
    main()
