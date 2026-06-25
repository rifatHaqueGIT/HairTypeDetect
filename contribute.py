"""
User Data Contribution — submit labeled hair images to grow the collective dataset.

Usage:
    python contribute.py --image photo.jpg --label Curly
    python contribute.py --image ./my_photos/ --label Wavy
    python contribute.py --image photo.jpg --label Straight --contributor "Jane"
"""

import argparse
import csv
import hashlib
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image


# ── Constants ────────────────────────────────────────────────────────────────

VALID_LABELS = ["Straight", "Wavy", "Curly", "Kinky"]
_LABEL_LOOKUP = {name.lower(): name for name in VALID_LABELS}

CONTRIBUTIONS_DIR = "./contributions"
PENDING_DIR = os.path.join(CONTRIBUTIONS_DIR, "pending")
ACCEPTED_DIR = os.path.join(CONTRIBUTIONS_DIR, "accepted")
REJECTED_DIR = os.path.join(CONTRIBUTIONS_DIR, "rejected")
MANIFEST_PATH = os.path.join(CONTRIBUTIONS_DIR, "manifest.csv")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_RESOLUTION = 64  # minimum width/height in pixels

MANIFEST_FIELDS = [
    "id", "original_filename", "stored_filename", "label",
    "contributor", "timestamp", "image_hash", "width", "height",
    "status", "review_timestamp", "merged_timestamp",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_label(label: str) -> Optional[str]:
    """Case-insensitive label lookup. Returns canonical name or None."""
    return _LABEL_LOOKUP.get(label.strip().lower())


def _file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file (for deduplication)."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # first 16 hex chars is enough for dedup


def _validate_image(filepath: str) -> tuple[bool, str]:
    """
    Validate that the file is a readable image meeting minimum quality.

    Returns (is_valid, reason).
    """
    try:
        img = Image.open(filepath)
        img.verify()  # verify it's a valid image
    except Exception as e:
        return False, f"Not a valid image: {e}"

    # Re-open to get dimensions (verify closes the file)
    img = Image.open(filepath)
    w, h = img.size

    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        return False, f"Too small ({w}×{h}), minimum is {MIN_RESOLUTION}×{MIN_RESOLUTION}"

    return True, f"{w}×{h}"


def _ensure_dirs():
    """Create the contributions directory structure."""
    for label in VALID_LABELS:
        os.makedirs(os.path.join(PENDING_DIR, label), exist_ok=True)
        os.makedirs(os.path.join(ACCEPTED_DIR, label), exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)


def _init_manifest():
    """Create the manifest CSV if it doesn't exist."""
    if not os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()


def _append_manifest(row: dict):
    """Append a row to the manifest CSV."""
    with open(MANIFEST_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writerow(row)


def _load_existing_hashes() -> set:
    """Load all known image hashes from the manifest to detect duplicates."""
    hashes = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("image_hash"):
                    hashes.add(row["image_hash"])
    return hashes


# ── Main submission logic ────────────────────────────────────────────────────

def submit_image(image_path: str, label: str, contributor: str = "anonymous") -> dict:
    """
    Submit a single image with a label.

    Returns a dict with submission details, or raises ValueError on failure.
    """
    # Validate label
    canonical_label = _normalize_label(label)
    if canonical_label is None:
        raise ValueError(
            f"Invalid label '{label}'. Must be one of: {', '.join(VALID_LABELS)}"
        )

    # Validate the image file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check extension
    ext = Path(image_path).suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Use: {', '.join(VALID_EXTENSIONS)}")

    # Validate image quality
    is_valid, info = _validate_image(image_path)
    if not is_valid:
        raise ValueError(f"Image rejected: {info}")

    # Check for duplicates
    img_hash = _file_hash(image_path)
    existing_hashes = _load_existing_hashes()
    if img_hash in existing_hashes:
        raise ValueError(f"Duplicate image detected (hash={img_hash}). Skipping.")

    # Get image dimensions
    img = Image.open(image_path)
    width, height = img.size

    # Generate unique filename
    submission_id = uuid.uuid4().hex[:12]
    stored_filename = f"{submission_id}{ext}"
    dest_path = os.path.join(PENDING_DIR, canonical_label, stored_filename)

    # Copy the image
    shutil.copy2(image_path, dest_path)

    # Record in manifest
    row = {
        "id": submission_id,
        "original_filename": os.path.basename(image_path),
        "stored_filename": stored_filename,
        "label": canonical_label,
        "contributor": contributor,
        "timestamp": datetime.now().isoformat(),
        "image_hash": img_hash,
        "width": width,
        "height": height,
        "status": "pending",
        "review_timestamp": "",
        "merged_timestamp": "",
    }
    _append_manifest(row)

    return row


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Submit hair images to grow the dataset")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to a single image or a directory of images")
    parser.add_argument("--label", type=str, required=True,
                        help=f"Hair type label: {', '.join(VALID_LABELS)}")
    parser.add_argument("--contributor", type=str, default="anonymous",
                        help="Your name or identifier (optional)")
    args = parser.parse_args()

    _ensure_dirs()
    _init_manifest()

    path = Path(args.image)
    submitted = 0
    skipped = 0
    errors = []

    print(f"\n{'='*60}")
    print(f"  Hair Type Dataset — Contribute Images")
    print(f"  Label: {args.label}  |  Contributor: {args.contributor}")
    print(f"{'='*60}\n")

    # Collect image files
    if path.is_dir():
        image_files = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        )
        if not image_files:
            print(f"  ❌ No images found in {path}")
            return
        print(f"  Found {len(image_files)} images in {path}\n")
    elif path.is_file():
        image_files = [path]
    else:
        print(f"  ❌ Path not found: {path}")
        return

    # Submit each image
    for img_path in image_files:
        try:
            result = submit_image(str(img_path), args.label, args.contributor)
            print(f"  ✓ {img_path.name} → {result['label']}/{result['stored_filename']}")
            submitted += 1
        except (ValueError, FileNotFoundError) as e:
            print(f"  ⏭ {img_path.name} — {e}")
            skipped += 1
            errors.append(str(e))

    # Summary
    print(f"\n{'─'*60}")
    print(f"  Submitted: {submitted}  |  Skipped: {skipped}")
    print(f"  Status: pending review")
    print(f"  Manifest: {MANIFEST_PATH}")
    print(f"{'─'*60}\n")

    if submitted > 0:
        print(f"  💡 Run 'python review_contributions.py' to review and merge.\n")


if __name__ == "__main__":
    main()
