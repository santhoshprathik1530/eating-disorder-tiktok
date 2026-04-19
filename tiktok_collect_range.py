import argparse
import asyncio
import csv
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from TikTokApi import TikTokApi
from TikTokApi.exceptions import InvalidResponseException


DEFAULT_HASHTAGS = [
    "whatieatinaday",
    "whatieatinday",
    "wieiad",
    "what_i_eat_in_a_day",
    "eatwithme",
    "fooddiary",
    "foodlog",
    "dailymeals",
    "mealinspo",
    "dayofmeals",
]

VIDEO_FIELDS = [
    "video_id",
    "video_url",
    "caption",
    "hashtags",
    "created_at",
    "author_username",
    "view_count",
    "like_count",
    "comment_count",
    "share_count",
    "source_hashtag_query",
]

COMMENT_FIELDS = [
    "video_id",
    "comment_id",
    "comment_text",
    "comment_like_count",
    "comment_created_at",
    "comment_author_username",
    "comment_author_user_id",
]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect TikTok hashtag videos/comments across a date range."
    )
    parser.add_argument("--cookie-file", default="cookies.json")
    parser.add_argument("--start-date", default="2025-07-01")
    parser.add_argument(
        "--end-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--hashtags",
        nargs="*",
        default=DEFAULT_HASHTAGS,
        help="Hashtags to scan without leading #.",
    )
    parser.add_argument("--video-output", default="tiktok_videos_2025_07_to_today.csv")
    parser.add_argument(
        "--comment-output", default="tiktok_comments_2025_07_to_today.csv"
    )
    parser.add_argument(
        "--summary-output", default="tiktok_collect_2025_07_to_today_summary.json"
    )
    parser.add_argument(
        "--max-videos-per-hashtag",
        type=int,
        default=1500,
        help="Per-hashtag scan limit from TikTok's ranked feed.",
    )
    parser.add_argument(
        "--comments-per-video",
        type=int,
        default=0,
        help="Set > 0 to collect comments in a second pass.",
    )
    parser.add_argument(
        "--comment-video-limit",
        type=int,
        default=0,
        help="Limit comment scraping to first N kept videos; 0 means all kept videos.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Parallel hashtag workers / TikTok browser sessions.",
    )
    parser.add_argument("--browser", default="chromium", choices=["chromium", "webkit"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--min-sleep", type=float, default=1.2)
    parser.add_argument("--max-sleep", type=float, default=3.0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Flush video rows to CSV every N accepted videos per worker.",
    )
    parser.add_argument(
        "--progress-json",
        default="tiktok_collect_progress.json",
        help="Live JSON progress file updated during the run.",
    )
    parser.add_argument(
        "--dashboard-html",
        default="tiktok_collect_dashboard.html",
        help="Simple HTML dashboard generated from the live progress file.",
    )
    return parser.parse_args()


def parse_utc_date(date_text, end_of_day=False):
    base = datetime.strptime(date_text, "%Y-%m-%d")
    if end_of_day:
        return base.replace(
            hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc
        )
    return base.replace(tzinfo=timezone.utc)


def load_cookie_map(cookie_file):
    raw = json.loads(Path(cookie_file).read_text())
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


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def parse_video_time(video_dict):
    raw = video_dict.get("createTime") or video_dict.get("create_time")
    if raw is None:
        return None
    try:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)
    except Exception:
        return None


def get_caption(video_dict):
    return normalize_text(
        video_dict.get("desc") or video_dict.get("video_description") or ""
    )


def extract_hashtags(video_dict):
    found = set()
    for item in video_dict.get("textExtra", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("hashtagName") or item.get("hashtag_name")
        if name:
            found.add(str(name).lower().lstrip("#"))
    caption = get_caption(video_dict).lower()
    for match in re.findall(r"#([a-zA-Z0-9_]+)", caption):
        found.add(match.lower())
    return sorted(found)


def build_video_url(video_dict):
    video_id = video_dict.get("id")
    author = video_dict.get("author", {}) or {}
    username = author.get("uniqueId") or author.get("unique_id") or "unknown"
    if video_id:
        return f"https://www.tiktok.com/@{username}/video/{video_id}"
    return None


def build_video_row(video_dict, source_hashtag, start_utc, end_utc):
    created_at = parse_video_time(video_dict)
    if created_at is None:
        return None, "missing_create_time"
    if created_at < start_utc or created_at > end_utc:
        return None, "out_of_date_range"

    author = video_dict.get("author", {}) or {}
    stats = video_dict.get("statsV2") or video_dict.get("stats") or {}
    row = {
        "video_id": str(video_dict.get("id") or ""),
        "video_url": build_video_url(video_dict),
        "caption": get_caption(video_dict),
        "hashtags": ", ".join(extract_hashtags(video_dict)),
        "created_at": created_at.isoformat(),
        "author_username": author.get("uniqueId") or author.get("unique_id"),
        "view_count": stats.get("playCount") or stats.get("viewCount"),
        "like_count": stats.get("diggCount") or stats.get("likeCount"),
        "comment_count": stats.get("commentCount"),
        "share_count": stats.get("shareCount"),
        "source_hashtag_query": source_hashtag,
    }
    if not row["video_id"]:
        return None, "missing_video_id"
    return row, "accepted"


async def jitter_sleep(min_sleep, max_sleep):
    await asyncio.sleep(random.uniform(min_sleep, max_sleep))


def ensure_csv(path, fieldnames):
    file_path = Path(path)
    if file_path.exists():
        return
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path, fieldnames, rows):
    if not rows:
        return
    with Path(path).open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)


def build_dashboard_html(progress):
    stats = progress.get("stats", {})
    active_workers = progress.get("active_workers", {})
    hashtags_done = progress.get("hashtags_completed", [])
    hashtags_pending = progress.get("hashtags_pending", [])
    comment_progress = progress.get("comment_progress", {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TikTok Scrape Dashboard</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --muted: #94a3b8;
      --text: #e5e7eb;
      --accent: #22c55e;
      --accent2: #38bdf8;
      --warn: #f59e0b;
      --border: #1f2937;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(135deg, #020617, #111827 45%, #0f172a);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .card {{
      background: rgba(17, 24, 39, 0.92);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.22);
    }}
    .metric {{
      font-size: 28px;
      font-weight: 700;
      color: var(--accent);
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .small {{ color: var(--muted); font-size: 13px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{ color: var(--accent2); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    code {{
      background: rgba(255,255,255,0.06);
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .two {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
    }}
    @media (max-width: 900px) {{
      .two {{ grid-template-columns: 1fr; }}
    }}
  </style>
  <meta http-equiv="refresh" content="10">
</head>
<body>
  <h1>TikTok Scrape Dashboard</h1>
  <p class="small">Auto-refreshes every 10 seconds. Last updated: <code>{progress.get("last_updated_at", "unknown")}</code></p>

  <div class="grid">
    <div class="card"><div class="label">Raw Hits</div><div class="metric">{stats.get("raw_hits", 0)}</div></div>
    <div class="card"><div class="label">Accepted Videos</div><div class="metric">{stats.get("accepted", 0)}</div></div>
    <div class="card"><div class="label">Duplicates</div><div class="metric">{stats.get("duplicate_hits", 0)}</div></div>
    <div class="card"><div class="label">Out Of Range</div><div class="metric">{stats.get("out_of_date_range", 0)}</div></div>
    <div class="card"><div class="label">Comment Rows</div><div class="metric">{comment_progress.get("comments_written", 0)}</div></div>
    <div class="card"><div class="label">Completed Hashtags</div><div class="metric">{len(hashtags_done)}</div></div>
  </div>

  <div class="two">
    <div class="card">
      <h2>Workers</h2>
      <table>
        <thead><tr><th>Worker</th><th>Status</th><th>Hashtag</th><th>Accepted</th><th>Raw</th></tr></thead>
        <tbody>
          {"".join(
              f"<tr><td>{worker}</td><td>{info.get('status','')}</td><td><code>{info.get('hashtag','')}</code></td><td>{info.get('accepted',0)}</td><td>{info.get('raw_hits',0)}</td></tr>"
              for worker, info in sorted(active_workers.items())
          ) or "<tr><td colspan='5'>No active workers yet.</td></tr>"}
        </tbody>
      </table>
    </div>
    <div class="card">
      <h2>Comments</h2>
      <table>
        <tbody>
          <tr><th>Mode</th><td>{comment_progress.get("status", "pending")}</td></tr>
          <tr><th>Videos queued</th><td>{comment_progress.get("videos_queued", 0)}</td></tr>
          <tr><th>Videos processed</th><td>{comment_progress.get("videos_processed", 0)}</td></tr>
          <tr><th>Comments written</th><td>{comment_progress.get("comments_written", 0)}</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="two" style="margin-top: 16px;">
    <div class="card">
      <h2>Pending Hashtags</h2>
      <p class="small">{", ".join(f"#{h}" for h in hashtags_pending) if hashtags_pending else "None"}</p>
    </div>
    <div class="card">
      <h2>Completed Hashtags</h2>
      <p class="small">{", ".join(f"#{h}" for h in hashtags_done) if hashtags_done else "None yet"}</p>
    </div>
  </div>
</body>
</html>"""


def write_progress_files(args, progress):
    Path(args.progress_json).write_text(json.dumps(progress, indent=2))
    Path(args.dashboard_html).write_text(build_dashboard_html(progress))


async def update_progress(args, progress, progress_lock):
    async with progress_lock:
        progress["last_updated_at"] = utc_now_iso()
        write_progress_files(args, progress)


def load_existing_video_ids(path):
    existing = set()
    file_path = Path(path)
    if not file_path.exists():
        return existing
    with file_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row.get("video_id")
            if video_id:
                existing.add(video_id)
    return existing


async def video_worker(
    api,
    worker_index,
    hashtag_queue,
    seen_video_ids,
    seen_lock,
    write_lock,
    stats_lock,
    stats,
    args,
    start_utc,
    end_utc,
    progress,
    progress_lock,
):
    pending_rows = []
    while True:
        hashtag = await hashtag_queue.get()
        if hashtag is None:
            hashtag_queue.task_done()
            async with progress_lock:
                progress["active_workers"][str(worker_index)] = {
                    "status": "done",
                    "hashtag": None,
                    "accepted": progress["active_workers"].get(str(worker_index), {}).get("accepted", 0),
                    "raw_hits": progress["active_workers"].get(str(worker_index), {}).get("raw_hits", 0),
                }
            await update_progress(args, progress, progress_lock)
            break
        async with progress_lock:
            progress["active_workers"][str(worker_index)] = {
                "status": "running",
                "hashtag": hashtag,
                "accepted": 0,
                "raw_hits": 0,
            }
        await update_progress(args, progress, progress_lock)
        try:
            tag = api.hashtag(name=hashtag)
            async for video in tag.videos(
                count=args.max_videos_per_hashtag, session_index=worker_index
            ):
                async with stats_lock:
                    stats["raw_hits"] += 1
                async with progress_lock:
                    progress["active_workers"][str(worker_index)]["raw_hits"] += 1

                data = video.as_dict
                video_id = str(data.get("id") or "")
                if not video_id:
                    async with stats_lock:
                        stats["missing_video_id"] += 1
                    continue

                async with seen_lock:
                    if video_id in seen_video_ids:
                        duplicate = True
                    else:
                        seen_video_ids.add(video_id)
                        duplicate = False

                if duplicate:
                    async with stats_lock:
                        stats["duplicate_hits"] += 1
                    continue

                row, outcome = build_video_row(data, hashtag, start_utc, end_utc)
                async with stats_lock:
                    stats[outcome] += 1
                if row is None:
                    continue

                pending_rows.append(row)
                async with progress_lock:
                    progress["active_workers"][str(worker_index)]["accepted"] += 1
                if len(pending_rows) >= args.checkpoint_every:
                    async with write_lock:
                        append_rows(args.video_output, VIDEO_FIELDS, pending_rows)
                    pending_rows = []
                    await update_progress(args, progress, progress_lock)

                await jitter_sleep(args.min_sleep, args.max_sleep)
        except Exception:
            async with stats_lock:
                stats["worker_errors"] += 1
            async with progress_lock:
                progress["active_workers"][str(worker_index)]["status"] = "error"
        finally:
            async with progress_lock:
                progress["hashtags_completed"].append(hashtag)
                if hashtag in progress["hashtags_pending"]:
                    progress["hashtags_pending"].remove(hashtag)
                progress["active_workers"][str(worker_index)]["status"] = "idle"
                progress["active_workers"][str(worker_index)]["hashtag"] = None
            hashtag_queue.task_done()
            await update_progress(args, progress, progress_lock)

    if pending_rows:
        async with write_lock:
            append_rows(args.video_output, VIDEO_FIELDS, pending_rows)
        await update_progress(args, progress, progress_lock)


async def collect_comments(
    api,
    video_rows,
    args,
    write_lock,
    progress,
    progress_lock,
):
    pending_rows = []
    limit = args.comment_video_limit or len(video_rows)
    selected_rows = video_rows[:limit]
    async with progress_lock:
        progress["comment_progress"] = {
            "status": "running",
            "videos_queued": len(selected_rows),
            "videos_processed": 0,
            "comments_written": 0,
        }
    await update_progress(args, progress, progress_lock)
    for index, row in enumerate(selected_rows):
        session_index = index % args.workers
        try:
            async for comment in api.video(id=row["video_id"]).comments(
                count=args.comments_per_video, session_index=session_index
            ):
                data = getattr(comment, "as_dict", {}) or {}
                user = data.get("user", {}) or {}
                pending_rows.append(
                    {
                        "video_id": row["video_id"],
                        "comment_id": data.get("cid"),
                        "comment_text": normalize_text(data.get("text")),
                        "comment_like_count": data.get("digg_count"),
                        "comment_created_at": data.get("create_time"),
                        "comment_author_username": user.get("unique_id"),
                        "comment_author_user_id": user.get("uid"),
                    }
                )
                if len(pending_rows) >= 50:
                    async with write_lock:
                        append_rows(args.comment_output, COMMENT_FIELDS, pending_rows)
                    async with progress_lock:
                        progress["comment_progress"]["comments_written"] += len(pending_rows)
                    pending_rows = []
                    await update_progress(args, progress, progress_lock)
                await jitter_sleep(args.min_sleep, args.max_sleep)
        except InvalidResponseException:
            pass
        except Exception:
            pass
        async with progress_lock:
            progress["comment_progress"]["videos_processed"] += 1
        await jitter_sleep(args.min_sleep, args.max_sleep)
        await update_progress(args, progress, progress_lock)

    if pending_rows:
        async with write_lock:
            append_rows(args.comment_output, COMMENT_FIELDS, pending_rows)
        async with progress_lock:
            progress["comment_progress"]["comments_written"] += len(pending_rows)
            progress["comment_progress"]["status"] = "done"
        await update_progress(args, progress, progress_lock)
    else:
        async with progress_lock:
            progress["comment_progress"]["status"] = "done"
        await update_progress(args, progress, progress_lock)


async def main():
    args = parse_args()
    start_utc = parse_utc_date(args.start_date)
    end_utc = parse_utc_date(args.end_date, end_of_day=True)
    if start_utc > end_utc:
        raise ValueError("start-date must be <= end-date")

    ensure_csv(args.video_output, VIDEO_FIELDS)
    ensure_csv(args.comment_output, COMMENT_FIELDS)

    cookie_map = load_cookie_map(args.cookie_file)
    seen_video_ids = load_existing_video_ids(args.video_output)
    seen_lock = asyncio.Lock()
    write_lock = asyncio.Lock()
    stats_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    stats = Counter()
    progress = {
        "started_at": utc_now_iso(),
        "last_updated_at": utc_now_iso(),
        "start_date_utc": start_utc.isoformat(),
        "end_date_utc": end_utc.isoformat(),
        "video_output": args.video_output,
        "comment_output": args.comment_output,
        "summary_output": args.summary_output,
        "hashtags_pending": list(args.hashtags),
        "hashtags_completed": [],
        "active_workers": {},
        "comment_progress": {
            "status": "pending",
            "videos_queued": 0,
            "videos_processed": 0,
            "comments_written": 0,
        },
        "stats": {},
    }
    write_progress_files(args, progress)

    hashtag_queue = asyncio.Queue()
    for hashtag in args.hashtags:
        await hashtag_queue.put(hashtag)
    for _ in range(args.workers):
        await hashtag_queue.put(None)

    async with TikTokApi() as api:
        await api.create_sessions(
            cookies=[cookie_map for _ in range(args.workers)],
            num_sessions=args.workers,
            headless=args.headless,
            sleep_after=5,
            browser=args.browser,
        )

        workers = [
            asyncio.create_task(
                video_worker(
                    api=api,
                    worker_index=worker_index,
                    hashtag_queue=hashtag_queue,
                    seen_video_ids=seen_video_ids,
                    seen_lock=seen_lock,
                    write_lock=write_lock,
                    stats_lock=stats_lock,
                    stats=stats,
                    args=args,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    progress=progress,
                    progress_lock=progress_lock,
                )
            )
            for worker_index in range(args.workers)
        ]

        await hashtag_queue.join()
        await asyncio.gather(*workers)

        video_rows = []
        with Path(args.video_output).open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                created_at = row.get("created_at")
                if created_at:
                    video_rows.append(row)

        video_rows.sort(key=lambda row: row["created_at"], reverse=True)

        if args.comments_per_video > 0:
            await collect_comments(
                api=api,
                video_rows=video_rows,
                args=args,
                write_lock=write_lock,
                progress=progress,
                progress_lock=progress_lock,
            )

    comment_count = 0
    with Path(args.comment_output).open("r", newline="", encoding="utf-8") as f:
        comment_count = sum(1 for _ in csv.DictReader(f))

    progress["stats"] = dict(stats)
    progress["finished_at"] = utc_now_iso()
    progress["comment_progress"]["comments_written"] = comment_count
    write_progress_files(args, progress)

    summary = {
        "start_date_utc": start_utc.isoformat(),
        "end_date_utc": end_utc.isoformat(),
        "hashtag_variants": args.hashtags,
        "workers": args.workers,
        "browser": args.browser,
        "headless": args.headless,
        "video_output": args.video_output,
        "comment_output": args.comment_output,
        "video_rows_written": max(len(seen_video_ids), 0),
        "comment_rows_written": comment_count,
        "selection_rule": {
            "kept_if": [
                "video has a TikTok ID",
                "video createTime is present",
                f"video created_at is between {args.start_date} and {args.end_date} UTC inclusive",
                "video appeared in at least one queried hashtag feed",
                "video_id has not already been written",
            ]
        },
        "stats": dict(stats),
    }
    Path(args.summary_output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
