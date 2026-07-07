#!/usr/bin/env python
"""Download benchmark datasets used by the STD paper.

By default this script downloads annotations/metadata and, for Video-MME,
selected video chunks. Use --full only when the target disk has enough space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DATASETS = {
    "MLVU": "MLVU/MVLU",
    "Video-MME": "lmms-lab/Video-MME",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MLVU", "Video-MME", "all"], default="all")
    parser.add_argument("--output-root", default="/home/mcy/projects/Std/datasets")
    parser.add_argument("--full", action="store_true", help="Download every file in the selected dataset repo.")
    parser.add_argument(
        "--videomme-chunks",
        default="01",
        help="Comma-separated Video-MME chunk ids to download when --full is not set, e.g. 01,02,03.",
    )
    args = parser.parse_args()

    names = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    for name in names:
        repo_id = DATASETS[name]
        local_dir = root / name
        allow_patterns = None
        if not args.full:
            if name == "Video-MME":
                chunks = [x.strip() for x in args.videomme_chunks.split(",") if x.strip()]
                allow_patterns = ["README.md", "videomme/*.parquet", "subtitle.zip"]
                allow_patterns.extend([f"videos_chunked_{chunk}.zip" for chunk in chunks])
            elif name == "MLVU":
                allow_patterns = ["README.md", "MLVU/json/*.json"]

        mode = "full" if args.full else f"partial allow_patterns={allow_patterns}"
        print(f"Downloading {repo_id} -> {local_dir} ({mode})")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=allow_patterns,
        )


if __name__ == "__main__":
    main()
