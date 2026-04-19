import argparse
import asyncio
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from TikTokApi import TikTokApi
from TikTokApi.exceptions import InvalidResponseException


DATE_START_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)
DATE_END_UTC = datetime(2026, 4, 19, 23, 59, 59, tzinfo=timezone.utc)
HASHTAG_VARIANTS = [
    "whatieatinaday",
    "whatieatinday",
    "wieiad",
    "what_i_eat_in_a_day",
    "eatwithme",
    "fooddiary",
    "foodlog",
    "dailymeals",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect TikTok hashtag videos and optional comments for 2026 YTD."
    )
    parser.add_argument(
        "--cookie-file",
        default="cookies.json",
        help="Path to cookies.json converted from your TikTok browser cookies.",
    )
    parser.add_argument(
        "--video-output",
        default="tiktok_videos_2026_ytd.csv",
        help="CSV output path for video-level rows.",
    )
    parser.add_argument(
        "--comment-output",
        default="tiktok_comments_2026_ytd.csv",
        help="CSV output path for comment-level rows.",
    )
    parser.add_argument(
        "--summary-output",
        default="tiktok_collect_2026_summary.json",
        help="JSON output path for collection summary.",
    )
    parser.add_argument(
        "--max-videos-per-hashtag",
        type=int,
        default=400,
        help="Upper bound to scan per hashtag query.",
    )
    parser.add_argument(
        "--comments-per-video",
        type=int,
        default=20,
        help="Max comments to collect per kept video.",
    )
    parser.add_argument(
        "--min-sleep",
        type=float,
        default=1.5,
        help="Minimum seconds to sleep between requests.",
    )
    parser.add_argument(
        "--max-sleep",
        type=float,
        default=4.0,
        help="Maximum seconds to sleep between requests.",
    )
    parser.add_argument(
        "--browser",
        default="chromium",
        choices=["chromium", "firefox", "webkit"],
        help="Playwright browser engine.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless. Default is visible browser because TikTok blocks less often.",
    )
    return parser.parse_args()


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


def parse_video_time(video_dict):
    raw = video_dict.get("createTime") or video_dict.get("create_time")
    if raw is None:
        return None
    try:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)
    except Exception:
        return None


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


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


def build_video_row(video_dict, source_hashtag):
    created_at = parse_video_time(video_dict)
    if created_at is None:
        return None
    if created_at < DATE_START_UTC or created_at > DATE_END_UTC:
        return None

    author = video_dict.get("author", {}) or {}
    stats = video_dict.get("statsV2") or video_dict.get("stats") or {}
    return {
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


async def jitter_sleep(min_sleep, max_sleep):
    await asyncio.sleep(random.uniform(min_sleep, max_sleep))


async def collect_comments_for_video(api, video_id, comments_per_video, min_sleep, max_sleep):
    rows = []
    video = api.video(id=video_id)
    try:
        async for comment in video.comments(count=comments_per_video):
            data = getattr(comment, "as_dict", {}) or {}
            user = data.get("user", {}) or {}
            rows.append(
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
            await jitter_sleep(min_sleep, max_sleep)
    except InvalidResponseException:
        return rows
    except Exception:
        return rows
    return rows


async def collect_videos(api, args):
    seen_ids = set()
    video_rows = []
    stats = {
        "raw_hits": 0,
        "duplicate_hits": 0,
        "kept_videos": 0,
        "out_of_date_range": 0,
        "missing_create_time": 0,
    }

    for hashtag in HASHTAG_VARIANTS:
        tag = api.hashtag(name=hashtag)
        async for video in tag.videos(count=args.max_videos_per_hashtag):
            stats["raw_hits"] += 1
            data = video.as_dict
            video_id = str(data.get("id") or "")
            if not video_id:
                continue
            if video_id in seen_ids:
                stats["duplicate_hits"] += 1
                continue

            created_at = parse_video_time(data)
            if created_at is None:
                stats["missing_create_time"] += 1
                continue
            if created_at < DATE_START_UTC or created_at > DATE_END_UTC:
                stats["out_of_date_range"] += 1
                continue

            seen_ids.add(video_id)
            row = build_video_row(data, hashtag)
            if row is None:
                continue
            video_rows.append(row)
            stats["kept_videos"] += 1

            await jitter_sleep(args.min_sleep, args.max_sleep)

    return video_rows, stats


async def main():
    args = parse_args()
    cookie_map = load_cookie_map(args.cookie_file)

    async with TikTokApi() as api:
        await api.create_sessions(
            cookies=[cookie_map],
            num_sessions=1,
            headless=args.headless,
            sleep_after=5,
            browser=args.browser,
        )

        video_rows, stats = await collect_videos(api, args)
        video_df = pd.DataFrame(video_rows)
        if not video_df.empty:
            video_df = (
                video_df.sort_values("created_at", ascending=False)
                .drop_duplicates(subset=["video_id"])
                .reset_index(drop=True)
            )
        video_df.to_csv(args.video_output, index=False)

        comment_rows = []
        for row in video_df.to_dict(orient="records"):
            comment_rows.extend(
                await collect_comments_for_video(
                    api=api,
                    video_id=row["video_id"],
                    comments_per_video=args.comments_per_video,
                    min_sleep=args.min_sleep,
                    max_sleep=args.max_sleep,
                )
            )
            await jitter_sleep(args.min_sleep, args.max_sleep)

        comment_df = pd.DataFrame(comment_rows)
        if not comment_df.empty:
            comment_df = (
                comment_df.drop_duplicates(subset=["video_id", "comment_id"])
                .reset_index(drop=True)
            )
        comment_df.to_csv(args.comment_output, index=False)

    summary = {
        "date_start_utc": DATE_START_UTC.isoformat(),
        "date_end_utc": DATE_END_UTC.isoformat(),
        "hashtag_variants": HASHTAG_VARIANTS,
        "video_rows_written": int(len(video_df)),
        "comment_rows_written": int(len(comment_df)),
        "video_output": args.video_output,
        "comment_output": args.comment_output,
        "stats": stats,
    }
    Path(args.summary_output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
