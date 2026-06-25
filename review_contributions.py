"""
Review & merge user-contributed images into the main training dataset.

Usage:
    python review_contributions.py                   # interactive review
    python review_contributions.py --auto-accept     # accept all pending
    python review_contributions.py --merge           # merge accepted → data/
    python review_contributions.py --stats            # show contribution stats
"""

import argparse
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image


# ── Constants (shared with contribute.py) ────────────────────────────────────

VALID_LABELS = ["Straight", "Wavy", "Curly", "Kinky"]

CONTRIBUTIONS_DIR = "./contributions"
PENDING_DIR = os.path.join(CONTRIBUTIONS_DIR, "pending")
ACCEPTED_DIR = os.path.join(CONTRIBUTIONS_DIR, "accepted")
REJECTED_DIR = os.path.join(CONTRIBUTIONS_DIR, "rejected")
MANIFEST_PATH = os.path.join(CONTRIBUTIONS_DIR, "manifest.csv")

DATA_DIR = "./data"

MANIFEST_FIELDS = [
    "id", "original_filename", "stored_filename", "label",
    "contributor", "timestamp", "image_hash", "width", "height",
    "status", "review_timestamp", "merged_timestamp",
]


# ── Manifest helpers ─────────────────────────────────────────────────────────

def _load_manifest() -> list[dict]:
    """Load all rows from the manifest."""
    rows = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def _save_manifest(rows: list[dict]):
    """Overwrite the manifest with updated rows."""
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _get_pending(rows: list[dict]) -> list[dict]:
    """Filter rows to only pending submissions."""
    return [r for r in rows if r.get("status") == "pending"]


def _get_accepted(rows: list[dict]) -> list[dict]:
    """Filter rows to accepted (not yet merged) submissions."""
    return [r for r in rows if r.get("status") == "accepted"]


# ── Review actions ───────────────────────────────────────────────────────────

def accept_submission(row: dict) -> dict:
    """Move a submission from pending to accepted."""
    src = os.path.join(PENDING_DIR, row["label"], row["stored_filename"])
    dest_dir = os.path.join(ACCEPTED_DIR, row["label"])
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, row["stored_filename"])

    if os.path.exists(src):
        shutil.move(src, dest)

    row["status"] = "accepted"
    row["review_timestamp"] = datetime.now().isoformat()
    return row


def reject_submission(row: dict) -> dict:
    """Move a submission from pending to rejected."""
    src = os.path.join(PENDING_DIR, row["label"], row["stored_filename"])
    os.makedirs(REJECTED_DIR, exist_ok=True)
    dest = os.path.join(REJECTED_DIR, row["stored_filename"])

    if os.path.exists(src):
        shutil.move(src, dest)

    row["status"] = "rejected"
    row["review_timestamp"] = datetime.now().isoformat()
    return row


def relabel_submission(row: dict, new_label: str) -> dict:
    """Change the label and accept the submission."""
    # Move from old pending location
    src = os.path.join(PENDING_DIR, row["label"], row["stored_filename"])

    # Update label
    row["label"] = new_label

    # Move to accepted with new label
    dest_dir = os.path.join(ACCEPTED_DIR, new_label)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, row["stored_filename"])

    if os.path.exists(src):
        shutil.move(src, dest)

    row["status"] = "accepted"
    row["review_timestamp"] = datetime.now().isoformat()
    return row


# ── Interactive review ───────────────────────────────────────────────────────

def interactive_review(rows: list[dict]):
    """Review each pending submission interactively."""
    pending = _get_pending(rows)
    if not pending:
        print("  ✓ No pending submissions to review.\n")
        return rows

    print(f"\n  {len(pending)} pending submissions to review.\n")
    print("  Commands: [a]ccept  [r]eject  [1-4] relabel (1=Straight, 2=Wavy, 3=Curly, 4=Kinky)  [s]kip  [q]uit\n")

    reviewed = 0
    for i, row in enumerate(pending):
        img_path = os.path.join(PENDING_DIR, row["label"], row["stored_filename"])

        # Show image info
        info = f"  [{i+1}/{len(pending)}] {row['original_filename']}"
        info += f"  |  Label: {row['label']}  |  {row['width']}×{row['height']}"
        info += f"  |  By: {row['contributor']}  |  {row['timestamp'][:10]}"
        print(info)

        # Try to show the image (optional — works if display is available)
        try:
            img = Image.open(img_path)
            img.show()
        except Exception:
            print(f"    (image at: {img_path})")

        # Get user decision
        while True:
            choice = input("    → ").strip().lower()

            if choice == "a":
                accept_submission(row)
                print("    ✓ Accepted")
                reviewed += 1
                break
            elif choice == "r":
                reject_submission(row)
                print("    ✗ Rejected")
                reviewed += 1
                break
            elif choice in ("1", "2", "3", "4"):
                new_label = VALID_LABELS[int(choice) - 1]
                relabel_submission(row, new_label)
                print(f"    ✓ Relabeled → {new_label} and accepted")
                reviewed += 1
                break
            elif choice == "s":
                print("    ⏭ Skipped")
                break
            elif choice == "q":
                print(f"\n  Reviewed {reviewed} submissions.\n")
                return rows
            else:
                print("    Invalid. Use: a/r/1/2/3/4/s/q")

    print(f"\n  Reviewed {reviewed} submissions.\n")
    return rows


# ── Auto-accept ──────────────────────────────────────────────────────────────

def auto_accept_all(rows: list[dict]) -> int:
    """Accept all pending submissions without review."""
    pending = _get_pending(rows)
    count = 0
    for row in pending:
        accept_submission(row)
        count += 1
    return count


# ── Merge accepted → data/ ──────────────────────────────────────────────────

def merge_to_dataset(rows: list[dict]) -> dict:
    """
    Copy all accepted (not yet merged) contributions into the main data/ folder.

    Returns a dict of label → count merged.
    """
    accepted = _get_accepted(rows)
    if not accepted:
        print("  ✓ No accepted submissions to merge.\n")
        return {}

    merged_counts = {label: 0 for label in VALID_LABELS}

    for row in accepted:
        label = row["label"]
        src = os.path.join(ACCEPTED_DIR, label, row["stored_filename"])

        if not os.path.exists(src):
            print(f"  ⚠ Missing file: {src}")
            continue

        dest_dir = os.path.join(DATA_DIR, label)
        os.makedirs(dest_dir, exist_ok=True)

        # Use a contribution-prefixed filename to avoid collisions with original data
        dest_filename = f"contrib_{row['stored_filename']}"
        dest = os.path.join(dest_dir, dest_filename)

        shutil.copy2(src, dest)

        row["status"] = "merged"
        row["merged_timestamp"] = datetime.now().isoformat()
        merged_counts[label] += 1

    total = sum(merged_counts.values())
    print(f"  ✓ Merged {total} images into {DATA_DIR}/")
    for label, count in merged_counts.items():
        if count > 0:
            print(f"    {label}: +{count}")

    return merged_counts


# ── Stats ────────────────────────────────────────────────────────────────────

def print_stats(rows: list[dict]):
    """Print contribution statistics."""
    status_counts = {}
    label_counts = {}
    contributor_counts = {}

    for row in rows:
        status = row.get("status", "unknown")
        label = row.get("label", "unknown")
        contributor = row.get("contributor", "unknown")

        status_counts[status] = status_counts.get(status, 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1
        contributor_counts[contributor] = contributor_counts.get(contributor, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Contribution Statistics")
    print(f"{'='*60}")
    print(f"  Total submissions: {len(rows)}\n")

    print("  By status:")
    for status, count in sorted(status_counts.items()):
        print(f"    {status:>10s}: {count}")

    print("\n  By label:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:>10s}: {count}")

    print("\n  By contributor:")
    for contributor, count in sorted(contributor_counts.items()):
        print(f"    {contributor:>15s}: {count}")

    print(f"{'='*60}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Review and merge contributed hair images")
    parser.add_argument("--auto-accept", action="store_true",
                        help="Accept all pending submissions without review")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all accepted images into the main data/ folder")
    parser.add_argument("--stats", action="store_true",
                        help="Show contribution statistics")
    args = parser.parse_args()

    if not os.path.exists(MANIFEST_PATH):
        print("  No contributions found. Run 'python contribute.py' first.\n")
        return

    rows = _load_manifest()

    if args.stats:
        print_stats(rows)
        return

    if args.auto_accept:
        count = auto_accept_all(rows)
        print(f"\n  ✓ Auto-accepted {count} pending submissions.\n")
        _save_manifest(rows)

    if args.merge:
        merge_to_dataset(rows)
        _save_manifest(rows)
        print()
        return

    # Default: interactive review
    if not args.auto_accept and not args.merge:
        rows = interactive_review(rows)
        _save_manifest(rows)


if __name__ == "__main__":
    main()
