#!/usr/bin/env python
"""Extract downloaded Video-MME video chunks."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/home/mcy/projects/Std/datasets/Video-MME")
    parser.add_argument("--output-dir", default="/home/mcy/projects/Std/datasets/Video-MME/videos")
    parser.add_argument("--chunks", default="01")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk in [x.strip() for x in args.chunks.split(",") if x.strip()]:
        zip_path = dataset_dir / f"videos_chunked_{chunk}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing {zip_path}. Run prepare_std_datasets.py first.")
        print(f"Extracting {zip_path} -> {output_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)


if __name__ == "__main__":
    main()
