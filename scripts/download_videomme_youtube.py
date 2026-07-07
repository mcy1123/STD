#!/usr/bin/env python
"""Download Video-MME videos on demand from metadata URLs."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def existing_video(output_dir: Path, video_id: str):
    for ext in VIDEO_EXTS:
        path = output_dir / f"{video_id}{ext}"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", default="/home/mcy/projects/Std/datasets/Video-MME")
    parser.add_argument("--output-dir", default="/home/mcy/projects/Std/datasets/Video-MME/videos")
    parser.add_argument("--split", default="test")
    parser.add_argument("--duration", choices=["short", "medium", "long", "all"], default="short")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.metadata_path, split=args.split)

    ydl_opts = {
        "format": "best[ext=mp4][vcodec!=none][acodec!=none]/best[ext=mp4]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
    }

    attempted = 0
    downloaded = 0
    seen = set()
    with YoutubeDL(ydl_opts) as ydl:
        for row in ds:
            if args.duration != "all" and row.get("duration") != args.duration:
                continue
            video_id = row.get("videoID")
            url = row.get("url")
            if not video_id or not url or video_id in seen:
                continue
            seen.add(video_id)
            if args.skip and attempted < args.skip:
                attempted += 1
                continue
            existing = existing_video(output_dir, video_id)
            if existing:
                print(f"exists {video_id}: {existing}")
                downloaded += 1
            else:
                print(f"downloading {video_id}: {url}")
                try:
                    ydl.download([url])
                except DownloadError as exc:
                    print(f"warning: failed to download {video_id}: {exc}")
                    attempted += 1
                    continue
                existing = existing_video(output_dir, video_id)
                if existing:
                    print(f"downloaded {video_id}: {existing}")
                    downloaded += 1
                else:
                    print(f"warning: no local video found for {video_id} after download")
            attempted += 1
            if downloaded >= args.limit:
                break
    print(f"downloaded_or_existing={downloaded} attempted={attempted} output_dir={output_dir}")


if __name__ == "__main__":
    main()
