"""
Training script for the Hair Type Classifier.

Usage:
    python train.py --data_dir ./data --epochs 25 --batch_size 32 --lr 0.001
    python train.py --data_dir ./data --epochs 10 --unfreeze   # full fine-tuning
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import HairTypeDataset, get_train_transforms, get_val_transforms, CLASS_NAMES
from model import create_model, unfreeze_backbone, count_parameters


# ── CLI Arguments ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train the Hair Type Classifier")
    p.add_argument("--data_dir", type=str, default="./data", help="Path to dataset root")
    p.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--val_split", type=float, default=0.2, help="Fraction of data for validation")
    p.add_argument("--unfreeze", action="store_true", help="Unfreeze backbone from start")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Where to save model")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


# ── Training Logic ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_curves(history: dict, save_path: str):
    """Save training / validation loss & accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(history["val_acc"], label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Curves saved to {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Seed everything
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Hair Type Classifier — Training")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {args.batch_size}")
    print(f"  LR     : {args.lr}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────

    print("Loading dataset …")
    full_dataset = HairTypeDataset(args.data_dir, transform=None)  # transforms applied later
    print(f"  Class distribution: {full_dataset.class_distribution()}\n")

    # Train / val split
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val

    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Wrap subsets with transforms
    train_dataset = _TransformSubset(train_subset, get_train_transforms())
    val_dataset = _TransformSubset(val_subset, get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Train: {n_train} samples  |  Val: {n_val} samples\n")

    # ── Model ────────────────────────────────────────────────────────────────

    freeze = not args.unfreeze
    model = create_model(num_classes=len(CLASS_NAMES), pretrained=True, freeze_backbone=freeze)
    model = model.to(device)

    info = count_parameters(model)
    print(f"  Model: EfficientNet-B0  —  {info['trainable']:,} trainable / {info['total']:,} total params")
    print(f"  Backbone {'frozen' if freeze else 'unfrozen'}\n")

    # ── Optimizer / Scheduler / Loss ─────────────────────────────────────────

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ────────────────────────────────────────────────────────

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = scheduler.get_last_lr()[0]
        print(f"  Loss: {train_loss:.4f} / {val_loss:.4f}  |  "
              f"Acc: {train_acc:.2%} / {val_acc:.2%}  |  LR: {lr_now:.6f}")

        # ── Checkpoint best model ────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
            }, ckpt_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ⏹ Early stopping triggered (patience={args.patience})")
                break

        print()

    # ── Save curves & history ────────────────────────────────────────────────

    plot_curves(history, os.path.join(args.checkpoint_dir, "training_curves.png"))

    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print(f"{'='*60}\n")


# ── Helper: apply transforms to a Subset ─────────────────────────────────────

class _TransformSubset:
    """Wraps a torch Subset to apply a specific transform pipeline."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    main()
