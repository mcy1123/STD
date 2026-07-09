#!/usr/bin/env python
"""Check which Video-MME metadata rows have local video files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def find_video(root: Path, names):
    for name in names:
        if not name:
            continue
        value = Path(str(name))
        if value.exists():
            return value
        for ext in ("",) + VIDEO_EXTS:
            direct = root / f"{name}{ext}"
            if direct.exists():
                return direct
        for ext in VIDEO_EXTS:
            matches = list(root.rglob(f"{name}{ext}"))
            if matches:
                return matches[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", default=str(Path(__file__).resolve().parents[1] / "datasets" / "Video-MME"))
    parser.add_argument("--video-root", default=str(Path(__file__).resolve().parents[1] / "datasets" / "Video-MME" / "videos"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "results" / "videomme_asset_check.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ds = load_dataset(args.metadata_path, split=args.split)
    root = Path(args.video_root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    found = 0
    total = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            total += 1
            path = find_video(root, [row.get("videoID"), row.get("video_id")])
            record = {
                "question_id": row.get("question_id"),
                "videoID": row.get("videoID"),
                "video_id": row.get("video_id"),
                "duration": row.get("duration"),
                "found": path is not None,
                "path": str(path) if path else None,
            }
            if path:
                found += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.limit and total >= args.limit:
                break

    print(f"found={found} total={total} video_root={root} output={out}")


if __name__ == "__main__":
    main()

