"""
Databricks Sync — upload / download the hair type dataset to/from Databricks.

Supports DBFS (Databricks File System) and Unity Catalog Volumes.
Uses incremental sync (only uploads new/changed files) via a local state file.

Usage:
    python databricks_sync.py upload --source ./data
    python databricks_sync.py download --source /FileStore/hair-type-dataset/ --dest ./data
    python databricks_sync.py upload-contributions --source ./contributions/accepted/
    python databricks_sync.py status

Environment variables:
    DATABRICKS_HOST   — e.g. https://your-workspace.cloud.databricks.com
    DATABRICKS_TOKEN  — personal access token (dapi...)
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.files import FileIO
    HAS_DATABRICKS_SDK = True
except ImportError:
    HAS_DATABRICKS_SDK = False


# ── Constants ────────────────────────────────────────────────────────────────

SYNC_STATE_FILE = ".databricks_sync.json"
DEFAULT_DBFS_PATH = "/FileStore/hair-type-dataset"
DEFAULT_CONTRIBUTIONS_DBFS_PATH = "/FileStore/hair-type-dataset/contributions"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Sync state management ───────────────────────────────────────────────────

def _load_sync_state() -> dict:
    """Load the local sync state (file hashes, last sync time)."""
    if os.path.exists(SYNC_STATE_FILE):
        with open(SYNC_STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_sync": None, "uploaded_files": {}}


def _save_sync_state(state: dict):
    """Save the sync state to disk."""
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


# ── Databricks client ───────────────────────────────────────────────────────

def _get_client() -> "WorkspaceClient":
    """Create a Databricks workspace client from environment variables."""
    if not HAS_DATABRICKS_SDK:
        print("  ❌ databricks-sdk is not installed.")
        print("     Install it with: pip install databricks-sdk")
        sys.exit(1)

    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")

    if not host or not token:
        print("  ❌ Missing environment variables:")
        if not host:
            print("     DATABRICKS_HOST — e.g. https://your-workspace.cloud.databricks.com")
        if not token:
            print("     DATABRICKS_TOKEN — your personal access token")
        print("\n  See .env.example for configuration details.\n")
        sys.exit(1)

    return WorkspaceClient(host=host, token=token)


# ── Upload ───────────────────────────────────────────────────────────────────

def _collect_files(source_dir: str) -> list[tuple[str, str]]:
    """
    Recursively collect image files from a directory.

    Returns list of (local_path, relative_path) tuples.
    """
    source = Path(source_dir)
    files = []
    for filepath in sorted(source.rglob("*")):
        if filepath.is_file() and filepath.suffix.lower() in VALID_EXTENSIONS:
            rel_path = filepath.relative_to(source).as_posix()
            files.append((str(filepath), rel_path))
    return files


def upload(source_dir: str, dbfs_dest: str, incremental: bool = True):
    """Upload a local directory to DBFS, skipping unchanged files."""
    client = _get_client()
    state = _load_sync_state()
    uploaded_files = state.get("uploaded_files", {})

    files = _collect_files(source_dir)
    if not files:
        print(f"  No image files found in {source_dir}")
        return

    print(f"\n  📤 Uploading {len(files)} files to DBFS: {dbfs_dest}")
    print(f"  Incremental: {'yes' if incremental else 'no (full re-upload)'}\n")

    uploaded = 0
    skipped = 0

    for local_path, rel_path in files:
        file_hash = _file_hash(local_path)
        remote_path = f"{dbfs_dest}/{rel_path}"

        # Skip if already uploaded with same hash
        if incremental and uploaded_files.get(rel_path) == file_hash:
            skipped += 1
            continue

        try:
            with open(local_path, "rb") as f:
                client.dbfs.put(remote_path, f, overwrite=True)
            uploaded_files[rel_path] = file_hash
            uploaded += 1
            print(f"  ✓ {rel_path}")
        except Exception as e:
            print(f"  ✗ {rel_path} — {e}")

    # Update sync state
    state["uploaded_files"] = uploaded_files
    state["last_sync"] = datetime.now().isoformat()
    state["last_sync_direction"] = "upload"
    state["dbfs_path"] = dbfs_dest
    _save_sync_state(state)

    print(f"\n  Done: {uploaded} uploaded, {skipped} skipped (unchanged)")
    print(f"  DBFS path: {dbfs_dest}\n")


# ── Download ─────────────────────────────────────────────────────────────────

def download(dbfs_source: str, local_dest: str):
    """Download files from DBFS to a local directory."""
    client = _get_client()

    print(f"\n  📥 Downloading from DBFS: {dbfs_source}")
    print(f"  Destination: {local_dest}\n")

    downloaded = 0

    try:
        files = list(client.dbfs.list(dbfs_source))
    except Exception as e:
        print(f"  ❌ Could not list DBFS path: {e}")
        return

    for file_info in files:
        if file_info.is_dir:
            # Recurse into subdirectories
            subdir_name = file_info.path.split("/")[-1]
            local_subdir = os.path.join(local_dest, subdir_name)
            download(file_info.path, local_subdir)
        else:
            filename = file_info.path.split("/")[-1]
            ext = Path(filename).suffix.lower()
            if ext not in VALID_EXTENSIONS:
                continue

            os.makedirs(local_dest, exist_ok=True)
            local_path = os.path.join(local_dest, filename)

            try:
                response = client.dbfs.read(file_info.path)
                with open(local_path, "wb") as f:
                    f.write(response.data)
                downloaded += 1
                print(f"  ✓ {filename}")
            except Exception as e:
                print(f"  ✗ {filename} — {e}")

    if downloaded > 0:
        print(f"\n  Downloaded {downloaded} files to {local_dest}\n")


# ── Status ───────────────────────────────────────────────────────────────────

def show_status():
    """Show sync status and DBFS contents if reachable."""
    state = _load_sync_state()

    print(f"\n{'='*60}")
    print(f"  Databricks Sync Status")
    print(f"{'='*60}")

    if state.get("last_sync"):
        print(f"  Last sync:      {state['last_sync']}")
        print(f"  Direction:      {state.get('last_sync_direction', 'unknown')}")
        print(f"  DBFS path:      {state.get('dbfs_path', 'unknown')}")
        print(f"  Tracked files:  {len(state.get('uploaded_files', {}))}")
    else:
        print("  No sync history found.")

    # Check environment configuration
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    print(f"\n  DATABRICKS_HOST:  {'✓ set' if host else '✗ not set'}")
    print(f"  DATABRICKS_TOKEN: {'✓ set' if token else '✗ not set'}")

    # Try to list DBFS contents
    if host and token and HAS_DATABRICKS_SDK:
        try:
            client = _get_client()
            dbfs_path = state.get("dbfs_path", DEFAULT_DBFS_PATH)
            files = list(client.dbfs.list(dbfs_path))

            print(f"\n  DBFS contents ({dbfs_path}):")
            dir_counts = {}
            file_count = 0
            for f in files:
                if f.is_dir:
                    dir_name = f.path.split("/")[-1]
                    try:
                        sub_files = list(client.dbfs.list(f.path))
                        dir_counts[dir_name] = len(sub_files)
                    except Exception:
                        dir_counts[dir_name] = "?"
                else:
                    file_count += 1

            for dir_name, count in sorted(dir_counts.items()):
                print(f"    📁 {dir_name}/  ({count} files)")
            if file_count > 0:
                print(f"    📄 {file_count} files at root")

        except Exception as e:
            print(f"\n  ⚠ Could not connect to DBFS: {e}")
    elif not HAS_DATABRICKS_SDK:
        print("\n  ⚠ databricks-sdk not installed — install with: pip install databricks-sdk")

    print(f"{'='*60}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sync hair type dataset with Databricks")
    subparsers = parser.add_subparsers(dest="command", help="Sync command")

    # Upload
    upload_parser = subparsers.add_parser("upload", help="Upload dataset to DBFS")
    upload_parser.add_argument("--source", type=str, default="./data",
                               help="Local source directory")
    upload_parser.add_argument("--dest", type=str, default=DEFAULT_DBFS_PATH,
                               help="DBFS destination path")
    upload_parser.add_argument("--full", action="store_true",
                               help="Force full re-upload (ignore sync state)")

    # Download
    download_parser = subparsers.add_parser("download", help="Download dataset from DBFS")
    download_parser.add_argument("--source", type=str, default=DEFAULT_DBFS_PATH,
                                  help="DBFS source path")
    download_parser.add_argument("--dest", type=str, default="./data",
                                  help="Local destination directory")

    # Upload contributions
    contrib_parser = subparsers.add_parser("upload-contributions",
                                            help="Upload accepted contributions to DBFS")
    contrib_parser.add_argument("--source", type=str, default="./contributions/accepted/",
                                 help="Local accepted contributions directory")
    contrib_parser.add_argument("--dest", type=str, default=DEFAULT_CONTRIBUTIONS_DBFS_PATH,
                                 help="DBFS destination path for contributions")

    # Status
    subparsers.add_parser("status", help="Show sync status")

    args = parser.parse_args()

    if args.command == "upload":
        upload(args.source, args.dest, incremental=not args.full)
    elif args.command == "download":
        download(args.source, args.dest)
    elif args.command == "upload-contributions":
        upload(args.source, args.dest)
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
