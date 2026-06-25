"""
Evaluation script — loads a saved checkpoint and reports metrics on the dataset.

Usage:
    python evaluate.py --data_dir ./data --checkpoint checkpoints/best_model.pth
"""

import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import mlflow

from dataset import HairTypeDataset, get_val_transforms, CLASS_NAMES
from model import create_model
from mlflow_config import setup_tracking, get_or_create_experiment, log_evaluation_artifacts


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the Hair Type Classifier")
    p.add_argument("--data_dir", type=str, default="./data", help="Path to dataset root")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth", help="Model checkpoint")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--output_dir", type=str, default="./checkpoints", help="Where to save reports")
    p.add_argument("--run_id", type=str, default=None, help="MLflow run ID to log evaluation under (optional)")
    return p.parse_args()


@torch.no_grad()
def gather_predictions(model, loader, device):
    """Run inference on the full loader and return (all_labels, all_preds)."""
    model.eval()
    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── MLflow setup ─────────────────────────────────────────────────────────

    setup_tracking()
    get_or_create_experiment()

    # ── Load checkpoint ──────────────────────────────────────────────────────

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    class_names = ckpt.get("class_names", CLASS_NAMES)
    num_classes = len(class_names)

    model = create_model(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    print(f"  Loaded from epoch {ckpt.get('epoch', '?')} — "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}, val_acc={ckpt.get('val_acc', '?'):.2%}\n")

    # ── Dataset ──────────────────────────────────────────────────────────────

    dataset = HairTypeDataset(args.data_dir, transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Predictions ──────────────────────────────────────────────────────────

    y_true, y_pred = gather_predictions(model, loader, device)

    # ── Classification Report ────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                        digits=4, output_dict=True)
    print(report)

    # Save text report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  📝 Report saved to {report_path}")

    # ── Confusion Matrix ─────────────────────────────────────────────────────

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Hair Type Classifier — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved to {cm_path}")

    # ── Log to MLflow ────────────────────────────────────────────────────────

    # If a training run_id was provided, log under that run; otherwise start a new one
    run_kwargs = {}
    if args.run_id:
        run_kwargs["run_id"] = args.run_id
        print(f"\n  📊 Logging evaluation to existing MLflow run: {args.run_id}")
    else:
        run_kwargs["run_name"] = "evaluation"
        print(f"\n  📊 Logging evaluation to new MLflow run")

    with mlflow.start_run(**run_kwargs):
        # Log overall metrics
        overall_acc = report_dict["accuracy"]
        mlflow.log_metric("eval_accuracy", overall_acc)
        mlflow.log_metric("eval_macro_f1", report_dict["macro avg"]["f1-score"])
        mlflow.log_metric("eval_weighted_f1", report_dict["weighted avg"]["f1-score"])

        # Log per-class metrics
        for cls_name in class_names:
            if cls_name in report_dict:
                for metric_name in ["precision", "recall", "f1-score"]:
                    key = f"eval_{cls_name.lower()}_{metric_name.replace('-', '_')}"
                    mlflow.log_metric(key, report_dict[cls_name][metric_name])

        # Log evaluation params
        mlflow.log_params({
            "eval_checkpoint": args.checkpoint,
            "eval_dataset_size": len(dataset),
        })

        # Tag as evaluation run
        mlflow.set_tag("run_type", "evaluation")

        # Log artifacts
        log_evaluation_artifacts(args.output_dir)

        print(f"  ✓ Evaluation metrics logged to MLflow")
        print(f"  Overall accuracy: {overall_acc:.2%}\n")


if __name__ == "__main__":
    main()
