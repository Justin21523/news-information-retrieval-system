#!/usr/bin/env python3
"""
Monitor preprocessing and auto-build indexes when complete.

This script monitors the preprocessing progress and automatically
triggers index building when preprocessing is done.

Usage:
    python scripts/monitor_and_build.py

Author: CNIRS Project
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = "/home/justin/miniconda3/envs/ai_env/bin/python"
INPUT_FILE = PROJECT_ROOT / "data/preprocessed/merged_14days_preprocessed.jsonl"
TARGET_COUNT = 9808
CHECK_INTERVAL = 30  # seconds

def get_line_count(filepath: Path) -> int:
    """Get number of lines in file."""
    if not filepath.exists():
        return 0
    try:
        # Force file system sync and read directly
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    except:
        return 0

def run_indexing():
    """Run the complete indexing pipeline."""
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE! Starting index building...")
    print("=" * 60 + "\n")

    # Step 1: Build search indexes
    print("[Step 1/3] Building search indexes...")
    result = subprocess.run([
        PYTHON, "scripts/build_indexes_from_preprocessed.py",
        "--input", str(INPUT_FILE),
        "--output", "data/indexes_10k"
    ], cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print("ERROR: Index building failed!")
        return False

    # Step 2: Build BERT embeddings
    print("\n[Step 2/3] Building BERT embeddings...")
    result = subprocess.run([
        PYTHON, "scripts/build_bert_embeddings.py",
        "--input", str(INPUT_FILE),
        "--output", "data/indexes_10k/bert_embeddings.npy",
        "--model", "paraphrase-multilingual-MiniLM-L12-v2",
        "--batch-size", "64"
    ], cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print("ERROR: BERT embedding failed!")
        return False

    # Step 3: Update app configuration
    print("\n[Step 3/3] Updating app configuration...")
    app_file = PROJECT_ROOT / "app_simple.py"

    if app_file.exists():
        content = app_file.read_text()

        # Update INDEX_DIR
        content = content.replace(
            "app.config['INDEX_DIR'] = project_root / 'data' / 'indexes'",
            "app.config['INDEX_DIR'] = project_root / 'data' / 'indexes_10k'"
        )

        # Update preprocessed file path
        content = content.replace(
            "preprocessed_file = project_root / 'data' / 'preprocessed' / 'cna_mvp_preprocessed.jsonl'",
            "preprocessed_file = project_root / 'data' / 'preprocessed' / 'merged_14days_preprocessed.jsonl'"
        )

        app_file.write_text(content)
        print("App configuration updated.")

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nTo start the application:")
    print("  python app_simple.py")
    print("\nTo verify:")
    print("  curl http://localhost:5001/api/stats")

    return True

def main():
    print("=" * 60)
    print("CNIRS Preprocessing Monitor")
    print("=" * 60)
    print(f"Target file: {INPUT_FILE}")
    print(f"Target count: {TARGET_COUNT}")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print("=" * 60)
    print()

    last_count = 0
    start_time = time.time()

    while True:
        current_count = get_line_count(INPUT_FILE)
        progress = (current_count / TARGET_COUNT) * 100
        elapsed = time.time() - start_time

        # Calculate rate
        if elapsed > 0 and current_count > 0:
            rate = current_count / (elapsed / 60)  # per minute
            remaining = (TARGET_COUNT - current_count) / rate if rate > 0 else 0
            eta_str = f"ETA: {remaining:.1f} min"
        else:
            eta_str = "Calculating..."

        # Print progress
        print(f"\r[{time.strftime('%H:%M:%S')}] Progress: {current_count}/{TARGET_COUNT} ({progress:.1f}%) | {eta_str}    ", end="", flush=True)

        # Check if complete
        if current_count >= TARGET_COUNT:
            print()  # New line
            run_indexing()
            break

        # Check if stalled (no progress for 5 minutes)
        if current_count == last_count:
            stall_time = CHECK_INTERVAL
        else:
            stall_time = 0
            last_count = current_count

        if stall_time > 300:  # 5 minutes
            print(f"\nWARNING: No progress for 5 minutes. Preprocessing may have stopped.")
            break

        time.sleep(CHECK_INTERVAL)

    print("\nMonitor finished.")

if __name__ == "__main__":
    main()
