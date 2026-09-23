#!/usr/bin/env python3
"""Download the preprocessed MCD-rPPG dataset (Hugging Face Datasets format)
into a local directory so it can be mounted into the model containers.

The dataset is NOT included in this repository and must be downloaded
separately before training.

Usage:
    python download_dataset.py                # downloads to ../data/preprocessed
    python download_dataset.py /path/to/dir   # downloads to a custom directory
"""

import sys
from pathlib import Path

DEFAULT_TARGET = Path(__file__).resolve().parent.parent / 'data' / 'preprocessed'


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    target.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("The 'datasets' package is required: pip install datasets")
        return 1

    print(f"Downloading CrisChir/simple_mcd_rppg to {target} ...")
    dataset = load_dataset("CrisChir/simple_mcd_rppg")
    dataset.save_to_disk(str(target))
    print("Done. Point the container at this directory via RPPG_DATA_DIR "
          "(see DOCKER_TUTORIAL.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
