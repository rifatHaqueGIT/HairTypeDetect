# HairAPP — Hair Type Classifier

An ML-powered hair type classifier that identifies hair types (Type 1–4: Straight, Wavy, Curly, Kinky) from photos or a live webcam feed. Includes MLflow experiment tracking, a user data contribution pipeline, and Databricks integration for cloud dataset management.

## Features

- **Transfer Learning** — EfficientNet-B0 fine-tuned for hair type classification
- **Training Pipeline** — Full training loop with early stopping, cosine LR scheduling, and checkpoint saving
- **MLflow Experiment Tracking** — All training runs are tracked with params, metrics, and artifacts
- **Evaluation** — Per-class precision/recall/F1 reports and confusion matrix
- **Image Prediction** — Classify a single image or a batch of images from the command line
- **Live Camera** — Real-time hair type detection via webcam with a visual overlay
- **User Data Contribution** — Submit labeled hair images to grow the collective dataset
- **Databricks Sync** — Upload/download the dataset to Databricks DBFS for cloud-based training

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `databricks-sdk` is only required if you plan to sync with Databricks. Everything else works without it.

### 2. Organize your dataset

Place your images in a folder structure under `data/`:

```
data/
  Straight/
  Wavy/
  Curly/
  Kinky/
```

### 3. (Optional) Configure Databricks

Copy the environment template and fill in your credentials:

```bash
cp .env.example .env
```

Then edit `.env` with your Databricks workspace details:

```env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi_your_token_here
MLFLOW_TRACKING_URI=databricks
```

> **Where to get these values:**
> - **DATABRICKS_HOST** — Your workspace URL from the Databricks console
> - **DATABRICKS_TOKEN** — Generate one at: *Databricks → User Settings → Developer → Access Tokens → Generate New Token*
> - **MLFLOW_TRACKING_URI** — Set to `databricks` to use Databricks-hosted MLflow, or leave it out to use local tracking

To load these into your shell before running commands:

**PowerShell:**
```powershell
Get-Content .env | ForEach-Object { if ($_ -match '^([^#]\S+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
```

**Bash / macOS:**
```bash
export $(grep -v '^#' .env | xargs)
```

---

## Usage

### Train

```bash
python train.py --data_dir ./data --epochs 25 --batch_size 32
python train.py --data_dir ./data --epochs 10 --unfreeze   # full fine-tuning
```

Training automatically logs to MLflow. After training, you'll see output like:

```
  🔬 MLflow run: a1b2c3d4e5f6...
  ...
  Training complete!  Best val loss: 0.3412
```

### View experiment tracking (MLflow UI)

```bash
mlflow ui
```

Then open **http://localhost:5000** in your browser. You'll see all your training runs with:
- Hyperparameters (epochs, LR, batch size, etc.)
- Per-epoch metrics (loss, accuracy, learning rate)
- Artifacts (training curves, model checkpoint, history)

### Evaluate

```bash
python evaluate.py --data_dir ./data --checkpoint checkpoints/best_model.pth
```

To log evaluation results to the same MLflow run as training, pass the run ID:

```bash
python evaluate.py --data_dir ./data --run_id <mlflow-run-id>
```

### Predict on an image

```bash
python predict.py --image path/to/photo.jpg
python predict.py --image ./test_images/           # batch mode (directory)
```

### Live camera detection

```bash
python live_camera.py
```

**Controls:** `R` = scan, `S` = screenshot, `Q` = quit

---

## Contributing Data

Users can submit labeled hair images to grow the collective dataset.

### Submit images

```bash
python contribute.py --image photo.jpg --label Curly
python contribute.py --image ./my_photos/ --label Wavy
python contribute.py --image photo.jpg --label Straight --contributor "YourName"
```

Images are staged in `contributions/pending/` and tracked in `contributions/manifest.csv`.

### Review submissions

```bash
python review_contributions.py              # interactive review (accept/reject/relabel)
python review_contributions.py --auto-accept # accept all pending
python review_contributions.py --stats       # show contribution stats
```

### Merge into dataset

```bash
python review_contributions.py --merge       # copy accepted images into data/
```

After merging, retrain the model to incorporate the new data.

---

## Databricks Sync

Upload and download the dataset to/from Databricks DBFS. Requires `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables (see [Setup](#3-optional-configure-databricks)).

### Upload dataset

```bash
python databricks_sync.py upload --source ./data
python databricks_sync.py upload --source ./data --dest /FileStore/custom-path/
python databricks_sync.py upload --source ./data --full    # force full re-upload
```

Uploads are **incremental by default** — only new or changed files are uploaded.

### Upload contributions

```bash
python databricks_sync.py upload-contributions
```

### Download dataset

```bash
python databricks_sync.py download --dest ./data
```

### Check sync status

```bash
python databricks_sync.py status
```

---

## Architecture

- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Head:** Dropout → Linear(1280→256) → ReLU → Dropout → Linear(256→4)
- **Classes:** Straight, Wavy, Curly, Kinky (Type 1–4)
- **Tracking:** MLflow (local `./mlruns/` or Databricks-hosted)

## Project Structure

```
HairAPP/
├── model.py                 # EfficientNet-B0 with custom head
├── dataset.py               # Dataset loader & augmentations
├── train.py                 # Training loop with MLflow tracking
├── evaluate.py              # Evaluation with MLflow logging
├── predict.py               # Single-image & batch prediction
├── live_camera.py           # Real-time webcam detection
├── mlflow_config.py         # MLflow setup (local or Databricks)
├── contribute.py            # User data submission CLI
├── review_contributions.py  # Admin review & merge workflow
├── databricks_sync.py       # Dataset sync with Databricks DBFS
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
├── data/                    # Training images (per-class folders)
├── checkpoints/             # Saved models & training artifacts
├── contributions/           # User-submitted images
│   ├── pending/             # Awaiting review
│   ├── accepted/            # Reviewed & approved
│   ├── rejected/            # Rejected submissions
│   └── manifest.csv         # Submission tracking
├── mlruns/                  # Local MLflow experiment data
└── screenshots/             # Saved camera screenshots
```
