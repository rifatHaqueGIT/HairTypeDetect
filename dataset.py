"""
Hair Type Dataset — loads images from a folder structure and applies augmentations.

Expected directory layout:
    data/
        Straight/
            img001.jpg
            ...
        Wavy/
            img001.jpg
            ...
        Curly/
            img001.jpg
            ...
        Kinky/
            img001.jpg
            ...
"""

import os
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Mapping from folder name → hair-type label (0-indexed for CrossEntropyLoss)
CLASS_NAMES = ["Straight", "Wavy", "Curly", "Kinky"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Case-insensitive lookup: maps lowercase folder name → canonical class name
_FOLDER_TO_CLASS = {name.lower(): name for name in CLASS_NAMES}

# Folders to explicitly ignore (hairstyles, not hair *types*)
_IGNORED_FOLDERS = {"dreadlocks", "dreads", "braids", "locs"}

# Human-readable type numbers (for display only)
TYPE_LABELS = {0: "Type 1 (Straight)", 1: "Type 2 (Wavy)", 2: "Type 3 (Curly)", 3: "Type 4 (Coily/Kinky)"}

# ── Transforms ───────────────────────────────────────────────────────────────

IMG_SIZE = 224  # EfficientNet-B0 input size

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """Augmentations applied during training."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # slight over-size for random crop
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Deterministic transforms applied during validation / inference."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset ──────────────────────────────────────────────────────────────────

class HairTypeDataset(Dataset):
    """
    PyTorch dataset that reads hair images from a folder-per-class layout.

    Folder matching is **case-insensitive** — ``curly/``, ``Curly/``, and
    ``CURLY/`` all map to the "Curly" class.  Folders whose names appear in
    ``_IGNORED_FOLDERS`` (e.g. ``dreadlocks``) are skipped automatically.

    Parameters
    ----------
    root_dir : str | Path
        Path to the top-level data directory (e.g. ``./data``).
    transform : torchvision.transforms.Compose, optional
        Transform pipeline applied to each image.
    """

    def __init__(self, root_dir: str | Path, transform: Optional[transforms.Compose] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform or get_val_transforms()

        self.samples: list[Tuple[Path, int]] = []

        # Scan all sub-folders; match to class names case-insensitively
        for folder in sorted(self.root_dir.iterdir()):
            if not folder.is_dir():
                continue

            folder_lower = folder.name.lower()

            # Skip ignored folders (hairstyles, not hair types)
            if folder_lower in _IGNORED_FOLDERS:
                print(f"  ⏭  Skipping '{folder.name}/' (hairstyle, not a hair type)")
                continue

            # Match folder name to a known class
            canonical = _FOLDER_TO_CLASS.get(folder_lower)
            if canonical is None:
                print(f"  ⚠  Unknown folder '{folder.name}/' — skipping")
                continue

            label = CLASS_TO_IDX[canonical]
            count = 0
            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((img_path, label))
                    count += 1

            print(f"  ✓  {canonical:>10s}: {count} images  ← {folder.name}/")

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found under {self.root_dir}. "
                f"Expected sub-folders: {CLASS_NAMES}"
            )

        print(f"\n  Total: {len(self.samples)} images across {len(set(l for _, l in self.samples))} classes")

    # ─ Standard Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    # ─ Utilities ─────────────────────────────────────────────────────────────

    def class_distribution(self) -> dict[str, int]:
        """Return a dict of class_name → count."""
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts
