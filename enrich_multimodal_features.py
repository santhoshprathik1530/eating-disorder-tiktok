#!/usr/bin/env python3
"""Enrich top TikTok videos with comments, transcript, OCR, vision, and LLM labels.

The script is designed to be incremental:

1. It always reads the baseline feature CSV.
2. It can optionally collect TikTok comments using TikTokApi + cookies.
3. It can optionally download videos using the yt-dlp command if installed.
4. It can optionally transcribe audio with OpenAI-compatible Whisper.
5. It can optionally sample frames and send them to OpenRouter vision/LLM.

Outputs are keyed by video_id so they can be joined to human labels later.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


COMMENT_FIELDS = [
    "video_id",
    "comment_id",
    "comment_text",
    "comment_like_count",
    "comment_created_at",
    "comment_author_username",
    "comment_author_user_id",
]

SIGNALS = [
    "restriction",
    "binge_eating",
    "rapid_weight_loss",
    "body_dissatisfaction",
    "purging_compensation",
]

HUMAN_LABEL_COLUMNS = [
    "human_restriction_label",
    "human_binge_eating_label",
    "human_rapid_weight_loss_label",
    "human_body_dissatisfaction_label",
    "human_purging_compensation_label",
    "human_notes",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def percent(done: int, total: int) -> float:
    return round((done / total) * 100, 1) if total else 0.0


def build_dashboard_html(progress: dict[str, Any]) -> str:
    stages = progress.get("stages", {})
    pipeline_order = ["comments", "download", "enrich", "llm", "write_outputs"]
    stage_cards = []
    rows = []
    for name in pipeline_order:
        stage = stages.get(name, {})
        total = int(stage.get("total", 0))
        completed = int(stage.get("completed", 0))
        running = int(stage.get("running", 0))
        errors = int(stage.get("errors", 0))
        queued = max(total - completed - running, 0)
        pct = percent(completed, total)
        status = stage.get("status", "pending")
        stage_cards.append(
            f"""
            <div class="stage {status}">
              <div class="stage-name">{name.replace('_', ' ').title()}</div>
              <div class="stage-status">{status}</div>
              <div class="bar"><span style="width:{pct}%"></span></div>
              <div class="stage-meta">{completed}/{total} done</div>
              <div class="stage-sub">running {running} | queued {queued} | errors {errors}</div>
            </div>
            """
        )
        rows.append(
            f"""
            <tr>
              <td>{name}</td>
              <td>{status}</td>
              <td>{completed}/{total}</td>
              <td>{running}</td>
              <td>{queued}</td>
              <td>{errors}</td>
              <td><div class="bar"><span style="width:{pct}%"></span></div></td>
            </tr>
            """
        )
    active_items = progress.get("active_items", {})
    active_html = "".join(
        f"<tr><td>{stage}</td><td><code>{video_id}</code></td></tr>"
        for stage, video_id in sorted(active_items.items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="10">
  <title>Enrichment Pipeline Dashboard</title>
  <style>
    :root {{
      --bg: #08111f;
      --panel: #101827;
      --text: #eef4ff;
      --muted: #93a4bb;
      --line: #253247;
      --good: #4ade80;
      --warn: #fbbf24;
      --accent: #67e8f9;
      --bad: #f87171;
    }}
    body {{
      margin: 0;
      padding: 28px;
      background:
        radial-gradient(circle at top left, rgba(103,232,249,.18), transparent 35%),
        linear-gradient(135deg, #08111f, #111827 55%, #0f172a);
      color: var(--text);
      font-family: Avenir Next, Trebuchet MS, Verdana, sans-serif;
    }}
    h1 {{ margin: 0 0 6px; font-size: 32px; }}
    h2 {{ margin: 0 0 12px; }}
    .muted {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 16px;
      margin: 22px 0;
    }}
    .card {{
      background: rgba(16,24,39,.9);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(0,0,0,.24);
    }}
    .metric {{ font-size: 34px; font-weight: 800; color: var(--good); }}
    .pipeline {{
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 12px;
      margin: 18px 0 22px;
    }}
    .stage {{
      background: rgba(16,24,39,.96);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      position: relative;
    }}
    .stage.running {{ border-color: var(--accent); box-shadow: 0 0 0 1px rgba(103,232,249,.25) inset; }}
    .stage.done {{ border-color: rgba(74,222,128,.45); }}
    .stage.skipped {{ border-color: rgba(148,163,184,.35); opacity: .8; }}
    .stage-name {{ font-size: 14px; text-transform: uppercase; letter-spacing: .08em; color: var(--accent); }}
    .stage-status {{ font-size: 26px; font-weight: 800; margin: 10px 0 12px; }}
    .stage-meta {{ margin-top: 10px; font-weight: 700; }}
    .stage-sub {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    code {{ background: rgba(255,255,255,.08); padding: 3px 7px; border-radius: 7px; }}
    .bar {{ height: 10px; min-width: 150px; background: #1e293b; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--good), var(--accent)); }}
    .two {{ display: grid; grid-template-columns: 1.4fr .8fr; gap: 16px; }}
    @media (max-width: 1100px) {{ .pipeline {{ grid-template-columns: repeat(2, 1fr); }} }}
    @media (max-width: 900px) {{ .two {{ grid-template-columns: 1fr; }} .pipeline {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Enrichment Pipeline Dashboard</h1>
  <div class="muted">Auto-refreshes every 10 seconds. Last updated: <code>{progress.get("last_updated_at", "")}</code></div>

  <div class="grid">
    <div class="card"><div class="muted">Rows</div><div class="metric">{progress.get("total_rows", 0)}</div></div>
    <div class="card"><div class="muted">Current Stage</div><div class="metric">{progress.get("current_stage", "starting")}</div></div>
    <div class="card"><div class="muted">Workers</div><div class="metric">{progress.get("workers", 0)}</div></div>
    <div class="card"><div class="muted">Status</div><div class="metric">{progress.get("status", "running")}</div></div>
  </div>

  <div class="card">
    <h2>Pipeline</h2>
    <div class="muted">Flow: comments -> downloads -> media enrichment -> llm labels -> outputs</div>
    <div class="pipeline">{''.join(stage_cards)}</div>
    <div class="muted">API spacing: <code>{progress.get("api_sleep_seconds", 0)}</code> seconds between OpenRouter requests</div>
  </div>

  <div class="two">
    <div class="card">
      <h2>Stages</h2>
      <table>
        <thead><tr><th>Stage</th><th>Status</th><th>Done</th><th>Running</th><th>Queued</th><th>Errors</th><th>Progress</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    <div class="card">
      <h2>Active Items</h2>
      <table>
        <thead><tr><th>Stage</th><th>Video ID</th></tr></thead>
        <tbody>{active_html or "<tr><td colspan='2'>No active items.</td></tr>"}</tbody>
      </table>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Output</h2>
    <div class="muted">CSV: <code>{progress.get("output_csv", "")}</code></div>
    <div class="muted">JSON: <code>{progress.get("output_json", "")}</code></div>
  </div>
</body>
</html>"""


class ProgressTracker:
    def __init__(self, args: argparse.Namespace, total_rows: int) -> None:
        self.path = Path(args.progress_json)
        self.dashboard_path = Path(args.dashboard_html)
        self.lock = threading.Lock()
        self.progress: dict[str, Any] = {
            "started_at": utc_now_iso(),
            "last_updated_at": utc_now_iso(),
            "status": "running",
            "current_stage": "starting",
            "total_rows": total_rows,
            "workers": args.workers,
            "api_sleep_seconds": args.api_sleep,
            "input_features": args.input_features,
            "output_csv": args.output_csv,
            "output_json": args.output_json,
            "active_items": {},
            "stages": {
                "comments": {"status": "skipped", "total": 0, "completed": 0, "running": 0, "errors": 0},
                "download": {"status": "pending", "total": 0, "completed": 0, "running": 0, "errors": 0},
                "enrich": {"status": "pending", "total": total_rows, "completed": 0, "running": 0, "errors": 0},
                "llm": {"status": "pending", "total": 0, "completed": 0, "running": 0, "errors": 0},
                "write_outputs": {"status": "pending", "total": 2, "completed": 0, "running": 0, "errors": 0},
            },
        }
        self.write()

    def write(self) -> None:
        self.progress["last_updated_at"] = utc_now_iso()
        self.path.write_text(json.dumps(self.progress, indent=2), encoding="utf-8")
        self.dashboard_path.write_text(build_dashboard_html(self.progress), encoding="utf-8")

    def start_stage(self, stage: str, total: int | None = None) -> None:
        with self.lock:
            self.progress["current_stage"] = stage
            item = self.progress["stages"][stage]
            item["status"] = "running"
            item["running"] = 0
            if total is not None:
                item["total"] = total
            self.write()

    def finish_stage(self, stage: str) -> None:
        with self.lock:
            self.progress["stages"][stage]["status"] = "done"
            self.progress["stages"][stage]["running"] = 0
            self.progress["active_items"].pop(stage, None)
            self.write()

    def skip_stage(self, stage: str) -> None:
        with self.lock:
            self.progress["stages"][stage]["status"] = "skipped"
            self.progress["stages"][stage]["running"] = 0
            self.progress["active_items"].pop(stage, None)
            self.write()

    def increment(self, stage: str, amount: int = 1, error: bool = False) -> None:
        with self.lock:
            item = self.progress["stages"][stage]
            item["completed"] += amount
            if error:
                item["errors"] += 1
            self.write()

    def active(self, stage: str, video_id: str | None) -> None:
        with self.lock:
            if video_id:
                self.progress["active_items"][stage] = video_id
                self.progress["stages"][stage]["running"] = 1
            else:
                self.progress["active_items"].pop(stage, None)
                self.progress["stages"][stage]["running"] = 0
            self.write()

    def done(self) -> None:
        with self.lock:
            self.progress["status"] = "done"
            self.progress["current_stage"] = "complete"
            self.progress["finished_at"] = utc_now_iso()
            self.progress["active_items"] = {}
            for stage in self.progress["stages"].values():
                stage["running"] = 0
            self.write()


class ApiRateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(float(min_interval_seconds), 0.0)
        self.lock = threading.Lock()
        self.next_allowed_at = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait_for = max(0.0, self.next_allowed_at - now)
            self.next_allowed_at = max(self.next_allowed_at, now) + self.min_interval_seconds
        if wait_for > 0:
            time.sleep(wait_for)


def load_cookie_map(cookie_file: Path) -> dict[str, str]:
    raw = json.loads(cookie_file.read_text(encoding="utf-8"))
    cookies = raw if isinstance(raw, list) else raw["cookies"]
    cookie_map = {}
    for cookie in cookies:
        domain = str(cookie.get("domain", ""))
        if "tiktok.com" not in domain:
            continue
        cookie_map[str(cookie["name"])] = str(cookie["value"])
    if not cookie_map:
        raise ValueError(f"No TikTok cookies found in {cookie_file}")
    return cookie_map


async def collect_comments_for_videos(
    videos: pd.DataFrame,
    cookie_file: Path,
    output_csv: Path,
    comments_per_video: int,
    browser: str,
    headless: bool,
    min_sleep: float,
    max_sleep: float,
    tracker: ProgressTracker,
) -> None:
    from TikTokApi import TikTokApi
    from TikTokApi.exceptions import InvalidResponseException

    cookie_map = load_cookie_map(cookie_file)
    rows_to_skip = set()
    if output_csv.exists():
        previous = pd.read_csv(output_csv, dtype={"video_id": str})
        rows_to_skip = set(previous["video_id"].dropna().astype(str))

    tracker.start_stage("comments", len(videos))
    async with TikTokApi() as api:
        await api.create_sessions(
            cookies=[cookie_map],
            num_sessions=1,
            headless=headless,
            sleep_after=5,
            browser=browser,
        )
        for _, row in videos.iterrows():
            video_id = str(row["video_id"])
            if video_id in rows_to_skip:
                tracker.increment("comments")
                continue
            tracker.active("comments", video_id)
            pending = []
            failed = False
            try:
                async for comment in api.video(id=video_id).comments(count=comments_per_video):
                    data = getattr(comment, "as_dict", {}) or {}
                    user = data.get("user", {}) or {}
                    pending.append(
                        {
                            "video_id": video_id,
                            "comment_id": data.get("cid"),
                            "comment_text": normalize_text(data.get("text")),
                            "comment_like_count": data.get("digg_count"),
                            "comment_created_at": data.get("create_time"),
                            "comment_author_username": user.get("unique_id"),
                            "comment_author_user_id": user.get("uid"),
                        }
                    )
                    await asyncio.sleep(min_sleep)
            except InvalidResponseException:
                failed = True
            except Exception as exc:
                print(f"Comment collection failed for {video_id}: {exc}")
                failed = True
            append_rows(output_csv, COMMENT_FIELDS, pending)
            tracker.increment("comments", error=failed)
            await asyncio.sleep(max_sleep)
    tracker.finish_stage("comments")


def aggregate_comments(comment_csv: Path, max_chars: int = 6000) -> pd.DataFrame:
    if not comment_csv.exists():
        return pd.DataFrame(columns=["video_id", "comments_text", "comments_collected"])
    comments = pd.read_csv(comment_csv, dtype={"video_id": str}).fillna("")
    if comments.empty:
        return pd.DataFrame(columns=["video_id", "comments_text", "comments_collected"])
    comments["comment_text"] = comments["comment_text"].map(normalize_text)
    comments = comments[comments["comment_text"] != ""]
    grouped = []
    for video_id, group in comments.groupby("video_id"):
        sorted_group = group.sort_values("comment_like_count", ascending=False)
        texts = []
        total = 0
        for text in sorted_group["comment_text"].tolist():
            if total + len(text) + 2 > max_chars:
                break
            texts.append(text)
            total += len(text) + 2
        grouped.append(
            {
                "video_id": str(video_id),
                "comments_text": "\n".join(texts),
                "comments_collected": int(len(group)),
            }
        )
    return pd.DataFrame(grouped)


def write_yt_dlp_cookies(cookie_file: Path, output_file: Path) -> Path | None:
    if not cookie_file.exists():
        return None
    raw = json.loads(cookie_file.read_text(encoding="utf-8"))
    cookies = raw if isinstance(raw, list) else raw.get("cookies", [])
    lines = ["# Netscape HTTP Cookie File"]
    for cookie in cookies:
        domain = str(cookie.get("domain", ""))
        if "tiktok.com" not in domain:
            continue
        include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        path = str(cookie.get("path") or "/")
        secure = "TRUE" if cookie.get("secure") else "FALSE"
        expires = str(int(cookie.get("expirationDate") or cookie.get("expires") or 0))
        name = str(cookie.get("name", ""))
        value = str(cookie.get("value", ""))
        if name:
            lines.append(
                "\t".join([domain, include_subdomains, path, secure, expires, name, value])
            )
    if len(lines) == 1:
        return None
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_file


def resolve_yt_dlp_cookie_file(args: argparse.Namespace) -> Path | None:
    if args.yt_dlp_cookies:
        cookie_path = Path(args.yt_dlp_cookies)
        return cookie_path if cookie_path.exists() else None
    return write_yt_dlp_cookies(Path(args.cookie_file), Path(args.yt_dlp_cookie_output))


def download_video(
    row: dict[str, Any],
    video_dir: Path,
    yt_dlp_cookie_file: Path | None,
) -> dict[str, Any]:
    yt_dlp_cmd = shutil.which("yt-dlp")
    if yt_dlp_cmd is None:
        try:
            import yt_dlp  # noqa: F401

            yt_dlp_cmd = None
        except Exception:
            print("yt-dlp is not installed; skipping video download.")
            return {
                "video_id": str(row.get("video_id", "")),
                "video_url": str(row.get("video_url", "")),
                "status": "failed",
                "error": "yt-dlp is not installed",
            }
    video_id = str(row["video_id"])
    output_template = str(video_dir / f"{video_id}.%(ext)s")
    existing = list(video_dir.glob(f"{video_id}.*"))
    if existing:
        return {
            "video_id": video_id,
            "video_url": str(row.get("video_url", "")),
            "status": "exists",
            "error": "",
        }
    if yt_dlp_cmd:
        cmd = [
            yt_dlp_cmd,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "-o",
            output_template,
            str(row["video_url"]),
        ]
    else:
        cmd = [
            "python3",
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "-o",
            output_template,
            str(row["video_url"]),
        ]
    if yt_dlp_cookie_file:
        insert_at = 1 if yt_dlp_cmd else 3
        cmd[insert_at:insert_at] = ["--cookies", str(yt_dlp_cookie_file)]
    try:
        subprocess.run(cmd, check=True)
        return {
            "video_id": video_id,
            "video_url": str(row.get("video_url", "")),
            "status": "downloaded",
            "error": "",
        }
    except subprocess.CalledProcessError as exc:
        print(f"Download failed for {video_id}: {exc}")
        return {
            "video_id": video_id,
            "video_url": str(row.get("video_url", "")),
            "status": "failed",
            "error": str(exc),
        }


def download_videos(
    videos: pd.DataFrame,
    video_dir: Path,
    workers: int,
    yt_dlp_cookie_file: Path | None,
    failed_downloads_csv: Path,
    tracker: ProgressTracker,
) -> None:
    ensure_dir(video_dir)
    rows = [row.to_dict() for _, row in videos.iterrows()]
    failed_rows = []
    tracker.start_stage("download", len(rows))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_video_id = {
            executor.submit(download_video, row, video_dir, yt_dlp_cookie_file): str(row["video_id"])
            for row in rows
        }
        for future in as_completed(future_to_video_id):
            video_id = future_to_video_id[future]
            tracker.active("download", video_id)
            try:
                result = future.result()
                failed = result.get("status") == "failed"
                if failed:
                    failed_rows.append(result)
                tracker.increment("download", error=failed)
            except Exception as exc:
                print(f"Download failed for {video_id}: {exc}")
                failed_rows.append(
                    {
                        "video_id": video_id,
                        "video_url": "",
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                tracker.increment("download", error=True)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_downloads_csv, index=False)
        print(f"Wrote failed downloads to {failed_downloads_csv}")
    elif failed_downloads_csv.exists():
        failed_downloads_csv.unlink()
    tracker.finish_stage("download")


def find_video_file(video_dir: Path, video_id: str) -> Path | None:
    for suffix in [".mp4", ".webm", ".mkv", ".mov"]:
        path = video_dir / f"{video_id}{suffix}"
        if path.exists():
            return path
    matches = list(video_dir.glob(f"{video_id}.*"))
    return matches[0] if matches else None


def ffmpeg_executable() -> str | None:
    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd:
        return ffmpeg_cmd
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_media_info(video_path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    duration = round(total_frames / fps, 3) if fps else 0.0
    info = {
        "video_duration_seconds": duration,
        "video_fps": round(fps, 3) if fps else 0.0,
        "video_frame_total": total_frames,
        "has_audio": False,
    }

    ffmpeg_cmd = ffmpeg_executable()
    if ffmpeg_cmd:
        proc = subprocess.run(
            [ffmpeg_cmd, "-i", str(video_path), "-hide_banner"],
            capture_output=True,
            text=True,
        )
        info["has_audio"] = "Audio:" in proc.stderr
    return info


def extract_audio(video_path: Path, audio_dir: Path) -> Path | None:
    ensure_dir(audio_dir)
    ffmpeg_cmd = ffmpeg_executable()
    if ffmpeg_cmd is None:
        print("ffmpeg is not installed; skipping audio extraction.")
        return None
    audio_path = audio_dir / f"{video_path.stem}.mp3"
    if audio_path.exists():
        return audio_path
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "4",
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path
    except subprocess.CalledProcessError:
        print(f"Audio extraction failed for {video_path.name}")
        return None


def transcribe_audio_openai(audio_path: Path, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    with audio_path.open("rb") as f:
        result = client.audio.transcriptions.create(model=model, file=f)
    return normalize_text(getattr(result, "text", ""))


def transcribe_media_faster_whisper(media_path: Path, model: str, device: str) -> str:
    from faster_whisper import WhisperModel

    compute_type = "int8" if device == "cpu" else "float16"
    whisper = WhisperModel(model, device=device, compute_type=compute_type)
    segments, _ = whisper.transcribe(str(media_path), vad_filter=True)
    return normalize_text(" ".join(segment.text.strip() for segment in segments))


def transcribe_with_local_fallback(
    video_path: Path,
    audio_dir: Path,
    model: str,
    device: str,
) -> str:
    try:
        transcript = transcribe_media_faster_whisper(video_path, model, device)
        if transcript:
            return transcript
    except Exception as exc:
        print(f"Direct video transcription failed for {video_path.stem}: {exc}")

    audio_path = extract_audio(video_path, audio_dir)
    if not audio_path:
        return ""
    return transcribe_media_faster_whisper(audio_path, model, device)


def sample_frames(video_path: Path, frame_dir: Path, every_seconds: float, max_frames: int) -> list[dict[str, Any]]:
    import cv2

    ensure_dir(frame_dir)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return []

    duration = total_frames / fps
    timestamps = []
    current = 0.0
    while current <= duration and (max_frames <= 0 or len(timestamps) < max_frames):
        timestamps.append(current)
        current += every_seconds

    output_frames = []
    for idx, seconds in enumerate(timestamps):
        output = frame_dir / f"{video_path.stem}_frame_{idx:03d}_{int(seconds):04d}s.jpg"
        if output.exists():
            output_frames.append({"path": output, "timestamp_seconds": round(seconds, 3)})
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(output), frame)
        output_frames.append({"path": output, "timestamp_seconds": round(seconds, 3)})
    cap.release()
    return output_frames


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def call_openrouter(messages: list[dict[str, Any]], model: str, temperature: float = 0.0) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Eating Disorder TikTok Feature Extraction",
        },
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def summarize_frames_openrouter(
    frame_paths: list[Path],
    caption: str,
    model: str,
    max_images: int,
    rate_limiter: ApiRateLimiter | None = None,
) -> dict[str, str]:
    if not frame_paths:
        return {"visual_frame_summary": "", "onscreen_text_ocr": ""}
    if rate_limiter:
        rate_limiter.wait()
    selected = frame_paths if max_images <= 0 else frame_paths[:max_images]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Analyze all {len(selected)} sampled TikTok video frames together for "
                "healthcare research on "
                "disordered-eating content signals. Do not diagnose the creator. "
                "Return compact JSON with keys visual_frame_summary and onscreen_text_ocr. "
                "Mention visible calorie totals, body checking, scale images, before/after "
                "claims, food quantity, or compensatory exercise if present. "
                "If the same overlay text appears repeatedly, deduplicate it. "
                f"Caption context: {caption[:1500]}"
            ),
        }
    ]
    for path in selected:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    raw = call_openrouter([{"role": "user", "content": content}], model=model)
    try:
        parsed = json.loads(extract_json(raw))
        return {
            "visual_frame_summary": normalize_text(parsed.get("visual_frame_summary")),
            "onscreen_text_ocr": normalize_text(parsed.get("onscreen_text_ocr")),
        }
    except Exception:
        return {"visual_frame_summary": normalize_text(raw), "onscreen_text_ocr": ""}


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def llm_signal_prediction(
    row: pd.Series,
    model: str,
    rate_limiter: ApiRateLimiter | None = None,
) -> dict[str, Any]:
    if rate_limiter:
        rate_limiter.wait()
    prompt = {
        "task": (
            "Classify whether this TikTok contains content signals related to "
            "disordered eating. This is content analysis only, not a diagnosis."
        ),
        "signals": SIGNALS,
        "output_schema": {
            signal: {
                "label": "yes/no/uncertain",
                "confidence": "0.0-1.0",
                "evidence": "short quote or observation",
            }
            for signal in SIGNALS
        }
        | {
            "recovery_or_educational_context": "yes/no/uncertain",
            "overall_notes": "brief explanation",
        },
        "video": {
            "video_id": row.get("video_id", ""),
            "caption": row.get("caption", ""),
            "hashtags": row.get("hashtags", ""),
            "metadata": {
                "views": row.get("view_count", ""),
                "likes": row.get("like_count", ""),
                "comments": row.get("comment_count", ""),
                "shares": row.get("share_count", ""),
            },
            "comments_text": row.get("comments_text", ""),
            "audio_transcript": row.get("audio_transcript", ""),
            "onscreen_text_ocr": row.get("onscreen_text_ocr", ""),
            "visual_frame_summary": row.get("visual_frame_summary", ""),
        },
    }
    raw = call_openrouter(
        [
            {
                "role": "system",
                "content": (
                    "You are a careful healthcare content-analysis assistant. "
                    "Use only evidence in the provided data. Avoid diagnosing people."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        model=model,
    )
    try:
        parsed = json.loads(extract_json(raw))
    except Exception:
        parsed = {"raw_response": raw}
    return {"llm_signal_prediction_json": json.dumps(parsed, ensure_ascii=False)}


def text_word_count(value: Any) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", normalize_text(value)))


def infer_visual_flags(summary: Any, ocr_text: Any) -> dict[str, bool]:
    text = f"{normalize_text(summary)} {normalize_text(ocr_text)}".lower()
    return {
        "visual_food_quantity": any(
            term in text
            for term in [
                "large amount",
                "small portion",
                "tiny portion",
                "food quantity",
                "multiple dishes",
                "plate",
                "meal",
            ]
        ),
        "visual_body_checking_present": any(
            term in text for term in ["body checking", "body check", "mirror selfie", "body-focused"]
        ),
        "visual_scale_present": any(term in text for term in ["scale", "weigh-in", "weight display"]),
        "visual_before_after_present": any(term in text for term in ["before/after", "before and after"]),
        "visual_exercise_present": any(
            term in text for term in ["exercise", "workout", "gym", "running", "cardio"]
        ),
    }


def parse_llm_prediction(value: Any) -> dict[str, str]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}

    if "raw_response" in parsed:
        return parse_markdown_prediction_table(str(parsed["raw_response"]))

    output: dict[str, str] = {}
    for signal in SIGNALS:
        item = parsed.get(signal, {})
        if isinstance(item, dict):
            output[f"llm_{signal}_label"] = normalize_text(item.get("label"))
            output[f"llm_{signal}_confidence"] = normalize_text(item.get("confidence"))
            output[f"llm_{signal}_evidence"] = normalize_text(item.get("evidence"))
    output["llm_recovery_or_educational_context"] = normalize_text(
        parsed.get("recovery_or_educational_context")
    )
    output["llm_overall_notes"] = normalize_text(parsed.get("overall_notes"))
    return output


def parse_markdown_prediction_table(raw: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip().strip("“”\"") for cell in stripped.strip("|").split("|")]
        if len(cells) < 4 or cells[0].lower() in {"signal", "--------"}:
            continue
        signal = cells[0].lower().replace(" ", "_").replace("-", "_")
        if signal in SIGNALS:
            output[f"llm_{signal}_label"] = cells[1].lower()
            output[f"llm_{signal}_confidence"] = cells[2]
            output[f"llm_{signal}_evidence"] = cells[3]
        elif signal == "recovery_or_educational_context":
            output["llm_recovery_or_educational_context"] = cells[1].lower()

    notes_match = re.search(r"\*\*Overall Notes\*\*\s*(.*)", raw, flags=re.I | re.S)
    if notes_match:
        output["llm_overall_notes"] = normalize_text(notes_match.group(1))
    return output


def add_parsed_llm_columns(enriched: pd.DataFrame) -> pd.DataFrame:
    parsed_rows = [
        parse_llm_prediction(value)
        for value in enriched.get("llm_signal_prediction_json", pd.Series([""] * len(enriched)))
    ]
    parsed_df = pd.DataFrame(parsed_rows).fillna("")
    for signal in SIGNALS:
        for field in ["label", "confidence", "evidence"]:
            col = f"llm_{signal}_{field}"
            if col not in parsed_df.columns:
                parsed_df[col] = ""
    for col in ["llm_recovery_or_educational_context", "llm_overall_notes"]:
        if col not in parsed_df.columns:
            parsed_df[col] = ""

    existing_cols = [col for col in parsed_df.columns if col in enriched.columns]
    if existing_cols:
        enriched = enriched.drop(columns=existing_cols)
    return pd.concat([enriched.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)


def empty_media_info() -> dict[str, Any]:
    return {
        "video_duration_seconds": 0.0,
        "video_fps": 0.0,
        "video_frame_total": 0,
        "has_audio": False,
    }


def empty_frame_info(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "frame_count": 0,
        "sampled_frame_timestamps": "",
        "frame_sampling_interval_seconds": args.frame_every_seconds,
    }


def enrich_one_video(
    row: dict[str, Any],
    args: argparse.Namespace,
    rate_limiter: ApiRateLimiter | None = None,
) -> dict[str, Any]:
    video_id = str(row["video_id"])
    video_path = find_video_file(Path(args.video_dir), video_id)
    result = {
        "video_id": video_id,
        "audio_transcript": "",
        "visual_summary": {"visual_frame_summary": "", "onscreen_text_ocr": ""},
        "media_info": empty_media_info(),
        "frame_info": empty_frame_info(args),
    }
    if not video_path:
        return result

    result["media_info"] = get_media_info(video_path)

    if args.transcribe_audio:
        transcript_path = Path(args.audio_dir) / f"{video_id}.transcript.txt"
        cached_transcript = (
            transcript_path.read_text(encoding="utf-8").strip()
            if transcript_path.exists()
            else ""
        )
        if cached_transcript:
            result["audio_transcript"] = cached_transcript
        else:
            try:
                if args.transcription_backend == "faster-whisper":
                    transcript = transcribe_with_local_fallback(
                        video_path,
                        Path(args.audio_dir),
                        args.faster_whisper_model,
                        args.faster_whisper_device,
                    )
                else:
                    audio_path = extract_audio(video_path, Path(args.audio_dir))
                    transcript = (
                        transcribe_audio_openai(audio_path, args.whisper_model)
                        if audio_path
                        else ""
                    )
                transcript_path.write_text(transcript, encoding="utf-8")
                result["audio_transcript"] = transcript
            except Exception as exc:
                print(f"Transcription failed for {video_id}: {exc}")

    if args.summarize_frames:
        video_frame_dir = Path(args.frame_dir) / video_id
        sampled_frames = sample_frames(
            video_path, video_frame_dir, args.frame_every_seconds, args.max_frames
        )
        frames = [frame["path"] for frame in sampled_frames]
        timestamps = [str(frame["timestamp_seconds"]) for frame in sampled_frames]
        result["frame_info"] = {
            "frame_count": len(sampled_frames),
            "sampled_frame_timestamps": ";".join(timestamps),
            "frame_sampling_interval_seconds": args.frame_every_seconds,
        }
        interval_label = str(args.frame_every_seconds).replace(".", "p")
        image_cap_label = "all" if args.max_vision_images <= 0 else str(args.max_vision_images)
        summary_path = video_frame_dir / (
            f"vision_summary_every_{interval_label}s_images_{image_cap_label}.json"
        )
        if summary_path.exists():
            result["visual_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            try:
                summary = summarize_frames_openrouter(
                    frames,
                    str(row.get("caption", "")),
                    args.vision_model,
                    args.max_vision_images,
                    rate_limiter=rate_limiter,
                )
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                result["visual_summary"] = summary
            except Exception as exc:
                print(f"Vision summary failed for {video_id}: {exc}")

    return result


def run_enrichment_jobs(
    enriched: pd.DataFrame,
    args: argparse.Namespace,
    tracker: ProgressTracker,
) -> list[dict[str, Any]]:
    rows = [row.to_dict() for _, row in enriched.iterrows()]
    results = []
    api_rate_limiter = ApiRateLimiter(args.api_sleep)
    tracker.start_stage("enrich", len(rows))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_video_id = {
            executor.submit(enrich_one_video, row, args, api_rate_limiter): str(row["video_id"])
            for row in rows
        }
        for future in as_completed(future_to_video_id):
            video_id = future_to_video_id[future]
            tracker.active("enrich", video_id)
            try:
                results.append(future.result())
                print(f"Enriched {video_id}")
                tracker.increment("enrich")
            except Exception as exc:
                print(f"Enrichment failed for {video_id}: {exc}")
                results.append(
                    {
                        "video_id": video_id,
                        "audio_transcript": "",
                        "visual_summary": {"visual_frame_summary": "", "onscreen_text_ocr": ""},
                        "media_info": empty_media_info(),
                        "frame_info": empty_frame_info(args),
                    }
                )
                tracker.increment("enrich", error=True)
    tracker.finish_stage("enrich")
    return results


def run_llm_predictions(
    enriched: pd.DataFrame,
    args: argparse.Namespace,
    tracker: ProgressTracker,
) -> list[str]:
    rows = [row for _, row in enriched.iterrows()]
    predictions = [""] * len(rows)
    api_rate_limiter = ApiRateLimiter(args.api_sleep)
    tracker.start_stage("llm", len(rows))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(llm_signal_prediction, row, args.llm_model, api_rate_limiter): index
            for index, row in enumerate(rows)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            video_id = rows[index].get("video_id", "")
            tracker.active("llm", str(video_id))
            try:
                predictions[index] = future.result()["llm_signal_prediction_json"]
                print(f"Predicted labels for {video_id}")
                tracker.increment("llm")
            except Exception as exc:
                print(f"LLM prediction failed for {video_id}: {exc}")
                tracker.increment("llm", error=True)
    tracker.finish_stage("llm")
    return predictions


def enrich_rows(args: argparse.Namespace, tracker: ProgressTracker) -> pd.DataFrame:
    baseline = pd.read_csv(args.input_features, dtype={"video_id": str}).fillna("")
    comments = aggregate_comments(Path(args.comments_csv))
    enriched = baseline.merge(comments, on="video_id", how="left").fillna("")
    if "comments_collected" not in enriched.columns:
        enriched["comments_collected"] = 0

    ensure_dir(Path(args.video_dir))
    ensure_dir(Path(args.audio_dir))
    ensure_dir(Path(args.frame_dir))

    audio_transcripts: dict[str, str] = {}
    visual_summaries: dict[str, dict[str, str]] = {}
    media_infos: dict[str, dict[str, Any]] = {}
    frame_infos: dict[str, dict[str, Any]] = {}

    if args.download_videos:
        yt_dlp_cookie_file = resolve_yt_dlp_cookie_file(args)
        if yt_dlp_cookie_file:
            print(f"Using yt-dlp cookies from {yt_dlp_cookie_file}")
        else:
            print("No yt-dlp cookies found; restricted TikTok downloads may fail.")
        download_videos(
            enriched,
            Path(args.video_dir),
            args.workers,
            yt_dlp_cookie_file,
            Path(args.failed_downloads_csv),
            tracker,
        )
    else:
        tracker.skip_stage("download")

    for result in run_enrichment_jobs(enriched, args, tracker):
        video_id = result["video_id"]
        audio_transcripts[video_id] = result["audio_transcript"]
        visual_summaries[video_id] = result["visual_summary"]
        media_infos[video_id] = result["media_info"]
        frame_infos[video_id] = result["frame_info"]

    enriched["audio_transcript"] = enriched["video_id"].map(audio_transcripts).fillna("")
    enriched["video_duration_seconds"] = enriched["video_id"].map(
        lambda video_id: media_infos.get(str(video_id), {}).get("video_duration_seconds", 0.0)
    )
    enriched["video_fps"] = enriched["video_id"].map(
        lambda video_id: media_infos.get(str(video_id), {}).get("video_fps", 0.0)
    )
    enriched["video_frame_total"] = enriched["video_id"].map(
        lambda video_id: media_infos.get(str(video_id), {}).get("video_frame_total", 0)
    )
    enriched["has_audio"] = enriched["video_id"].map(
        lambda video_id: media_infos.get(str(video_id), {}).get("has_audio", False)
    )
    enriched["frame_count"] = enriched["video_id"].map(
        lambda video_id: frame_infos.get(str(video_id), {}).get("frame_count", 0)
    )
    enriched["sampled_frame_timestamps"] = enriched["video_id"].map(
        lambda video_id: frame_infos.get(str(video_id), {}).get("sampled_frame_timestamps", "")
    )
    enriched["frame_sampling_interval_seconds"] = enriched["video_id"].map(
        lambda video_id: frame_infos.get(str(video_id), {}).get(
            "frame_sampling_interval_seconds", args.frame_every_seconds
        )
    )
    enriched["visual_frame_summary"] = enriched["video_id"].map(
        lambda video_id: visual_summaries.get(str(video_id), {}).get("visual_frame_summary", "")
    )
    enriched["onscreen_text_ocr"] = enriched["video_id"].map(
        lambda video_id: visual_summaries.get(str(video_id), {}).get("onscreen_text_ocr", "")
    )
    enriched["transcript_word_count"] = enriched["audio_transcript"].map(text_word_count)
    enriched["ocr_word_count"] = enriched["onscreen_text_ocr"].map(text_word_count)

    visual_flags = [
        infer_visual_flags(row.get("visual_frame_summary", ""), row.get("onscreen_text_ocr", ""))
        for _, row in enriched.iterrows()
    ]
    visual_flags_df = pd.DataFrame(visual_flags)
    enriched = pd.concat([enriched.reset_index(drop=True), visual_flags_df.reset_index(drop=True)], axis=1)

    if args.predict_with_llm:
        enriched["llm_signal_prediction_json"] = run_llm_predictions(enriched, args, tracker)
    else:
        tracker.skip_stage("llm")
        enriched["llm_signal_prediction_json"] = ""

    enriched = add_parsed_llm_columns(enriched)
    for column in HUMAN_LABEL_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = ""
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-features", default="testset_baseline_features.csv")
    parser.add_argument("--output-csv", default="testset_enriched_features.csv")
    parser.add_argument("--output-json", default="testset_enriched_features.json")
    parser.add_argument("--progress-json", default="pipeline_progress.json")
    parser.add_argument("--dashboard-html", default="pipeline_dashboard.html")
    parser.add_argument("--failed-downloads-csv", default="failed_downloads.csv")
    parser.add_argument("--comments-csv", default="testset_comments.csv")
    parser.add_argument("--cookie-file", default="cookies.json")
    parser.add_argument(
        "--yt-dlp-cookies",
        default="",
        help="Optional Netscape-format cookies.txt for yt-dlp downloads.",
    )
    parser.add_argument(
        "--yt-dlp-cookie-output",
        default="yt_dlp_cookies.txt",
        help="Temporary Netscape cookies file generated from --cookie-file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel workers for downloads, transcription, frame summaries, and LLM calls.",
    )
    parser.add_argument("--comments-per-video", type=int, default=30)
    parser.add_argument("--collect-comments", action="store_true")
    parser.add_argument("--browser", default="chromium", choices=["chromium", "webkit", "firefox"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--min-sleep", type=float, default=1.2)
    parser.add_argument("--max-sleep", type=float, default=3.0)
    parser.add_argument("--download-videos", action="store_true")
    parser.add_argument("--video-dir", default="testset_videos")
    parser.add_argument("--audio-dir", default="testset_audio")
    parser.add_argument("--frame-dir", default="testset_frames")
    parser.add_argument("--transcribe-audio", action="store_true")
    parser.add_argument(
        "--transcription-backend",
        default="faster-whisper",
        choices=["faster-whisper", "openai"],
    )
    parser.add_argument("--faster-whisper-model", default="base")
    parser.add_argument("--faster-whisper-device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--whisper-model", default="whisper-1")
    parser.add_argument("--summarize-frames", action="store_true")
    parser.add_argument("--vision-model", default="nvidia/nemotron-nano-12b-v2-vl:free")
    parser.add_argument("--frame-every-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum sampled frames per video. 0 means sample the whole video.",
    )
    parser.add_argument(
        "--max-vision-images",
        type=int,
        default=0,
        help="Maximum frames sent together to the vision model. 0 means send all sampled frames.",
    )
    parser.add_argument("--predict-with-llm", action="store_true")
    parser.add_argument("--llm-model", default="openai/gpt-oss-20b:free")
    parser.add_argument("--api-sleep", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    baseline = pd.read_csv(args.input_features, dtype={"video_id": str}).fillna("")
    tracker = ProgressTracker(args, len(baseline))

    if args.collect_comments:
        asyncio.run(
            collect_comments_for_videos(
                baseline,
                Path(args.cookie_file),
                Path(args.comments_csv),
                args.comments_per_video,
                args.browser,
                args.headless,
                args.min_sleep,
                args.max_sleep,
                tracker,
            )
        )
    else:
        tracker.skip_stage("comments")

    enriched = enrich_rows(args, tracker)
    tracker.start_stage("write_outputs", 2)
    enriched.to_csv(args.output_csv, index=False)
    tracker.increment("write_outputs")
    Path(args.output_json).write_text(
        json.dumps(enriched.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tracker.increment("write_outputs")
    tracker.finish_stage("write_outputs")
    tracker.done()
    print(f"Wrote {len(enriched)} rows to {args.output_csv}")
    print(f"Wrote JSON to {args.output_json}")
    print(f"Wrote progress JSON to {args.progress_json}")
    print(f"Wrote dashboard to {args.dashboard_html}")


if __name__ == "__main__":
    main()
