"""
MLflow configuration — handles local or Databricks-hosted experiment tracking.

By default, experiments are stored locally in ./mlruns/.
Set environment variables to connect to a Databricks MLflow server:

    MLFLOW_TRACKING_URI=databricks
    DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
    DATABRICKS_TOKEN=dapi...

Usage:
    from mlflow_config import setup_tracking, get_or_create_experiment
    setup_tracking()
    experiment_id = get_or_create_experiment("HairTypeClassifier")
"""

import os
from pathlib import Path

import mlflow


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_EXPERIMENT_NAME = "HairTypeClassifier"
LOCAL_TRACKING_DIR = "./mlruns"


def setup_tracking(tracking_uri: str | None = None) -> str:
    """
    Configure the MLflow tracking URI.

    Priority:
        1. Explicit ``tracking_uri`` argument
        2. ``MLFLOW_TRACKING_URI`` environment variable
        3. Local file store at ``./mlruns/``

    Returns
    -------
    str
        The resolved tracking URI that MLflow is now using.
    """
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if tracking_uri is None:
        # Fall back to local file-based tracking
        local_path = Path(LOCAL_TRACKING_DIR).resolve()
        local_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = str(local_path.as_uri())

    mlflow.set_tracking_uri(tracking_uri)

    is_remote = "databricks" in tracking_uri or tracking_uri.startswith("http")
    mode = "remote" if is_remote else "local"
    print(f"  📊 MLflow tracking: {mode} ({tracking_uri})")

    return tracking_uri


def get_or_create_experiment(name: str = DEFAULT_EXPERIMENT_NAME) -> str:
    """
    Get or create an MLflow experiment by name.

    Returns
    -------
    str
        The experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(name)
        print(f"  📁 Created new experiment: {name} (id={experiment_id})")

    mlflow.set_experiment(name)
    return experiment_id


def log_dataset_info(class_distribution: dict, total_samples: int, val_split: float):
    """Log dataset metadata as MLflow tags and params."""
    mlflow.log_param("dataset_total_samples", total_samples)
    mlflow.log_param("dataset_val_split", val_split)

    for cls_name, count in class_distribution.items():
        mlflow.log_param(f"dataset_class_{cls_name}", count)


def log_model_info(model_name: str, trainable_params: int, total_params: int,
                   freeze_backbone: bool):
    """Log model architecture metadata."""
    mlflow.log_param("model_backbone", model_name)
    mlflow.log_param("model_trainable_params", trainable_params)
    mlflow.log_param("model_total_params", total_params)
    mlflow.log_param("model_freeze_backbone", freeze_backbone)


def log_epoch_metrics(epoch: int, train_loss: float, val_loss: float,
                      train_acc: float, val_acc: float, lr: float):
    """Log training metrics for a single epoch."""
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "learning_rate": lr,
    }, step=epoch)


def log_training_artifacts(checkpoint_dir: str):
    """Log training output files (curves, history, best model) as artifacts."""
    artifact_files = ["training_curves.png", "history.json", "best_model.pth"]
    for fname in artifact_files:
        fpath = os.path.join(checkpoint_dir, fname)
        if os.path.exists(fpath):
            mlflow.log_artifact(fpath)
            print(f"    📎 Logged artifact: {fname}")


def log_evaluation_artifacts(output_dir: str):
    """Log evaluation output files as artifacts."""
    artifact_files = ["confusion_matrix.png", "classification_report.txt"]
    for fname in artifact_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            mlflow.log_artifact(fpath)
            print(f"    📎 Logged artifact: {fname}")


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uri = setup_tracking()
    exp_id = get_or_create_experiment()
    print(f"  Experiment ID: {exp_id}")
    print(f"  Tracking URI:  {uri}")
    print("  ✓ MLflow config OK")
