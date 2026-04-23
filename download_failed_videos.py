#!/usr/bin/env python3
"""Download videos listed in failed_downloads.csv on a local machine."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-csv", default="failed_downloads.csv")
    parser.add_argument("--video-dir", default="testset_videos")
    parser.add_argument("--cookies", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed_csv = Path(args.failed_csv)
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    if not failed_csv.exists():
        raise FileNotFoundError(f"{failed_csv} does not exist")

    yt_dlp = shutil.which("yt-dlp")
    base_cmd = [yt_dlp] if yt_dlp else ["python3", "-m", "yt_dlp"]
    rows = pd.read_csv(failed_csv, dtype={"video_id": str}).fillna("")
    for _, row in rows.iterrows():
        video_id = str(row["video_id"])
        video_url = str(row["video_url"])
        if not video_id or not video_url:
            continue
        if list(video_dir.glob(f"{video_id}.*")):
            print(f"Already exists: {video_id}")
            continue
        cmd = [
            *base_cmd,
            "--no-playlist",
            "-o",
            str(video_dir / f"{video_id}.%(ext)s"),
            video_url,
        ]
        if args.cookies:
            insert_at = 1 if yt_dlp else 3
            cmd[insert_at:insert_at] = ["--cookies", args.cookies]
        print(f"Downloading {video_id}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Still failed: {video_id}: {exc}")


if __name__ == "__main__":
    main()
