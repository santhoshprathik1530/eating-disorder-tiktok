import argparse
import csv
import json
import random
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html import unescape
from pathlib import Path


DATE_START_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)
DEFAULT_QUERY_VARIANTS = [
    "#whatieatinaday",
    "#wieiad",
    "\"what i eat in a day\"",
    "\"full day of eating\"",
    "#fulldayofeating",
    "\"what i ate today\"",
    "\"day of eats\"",
    "\"realistic what i eat in a day\"",
    "\"high protein what i eat in a day\"",
    "\"healthy what i eat in a day\"",
]
RELATED_TERMS = [
    "whatieatinaday",
    "what i eat in a day",
    "wieiad",
    "full day of eating",
    "fulldayofeating",
    "what i ate today",
    "day of eats",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect YouTube Shorts metadata for what-I-eat-in-a-day style posts in 2026."
    )
    parser.add_argument(
        "--output",
        default="youtube_shorts_whatieatinaday_2026_us_en.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--summary-output",
        default="youtube_collect_2026_summary.json",
        help="JSON summary path.",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=25,
        help="Max Shorts candidates to retain from each search page.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=250,
        help="Max unique candidate videos to verify from watch pages.",
    )
    parser.add_argument(
        "--max-length-seconds",
        type=int,
        default=60,
        help="Keep only videos at or below this duration.",
    )
    parser.add_argument(
        "--gl",
        default="US",
        help="YouTube region hint, e.g. US.",
    )
    parser.add_argument(
        "--hl",
        default="en",
        help="YouTube language hint, e.g. en.",
    )
    parser.add_argument(
        "--end-date",
        default="2026-12-31",
        help="Inclusive local date boundary in YYYY-MM-DD form.",
    )
    parser.add_argument(
        "--min-sleep",
        type=float,
        default=0.25,
        help="Minimum seconds to sleep between requests.",
    )
    parser.add_argument(
        "--max-sleep",
        type=float,
        default=0.75,
        help="Maximum seconds to sleep between requests.",
    )
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0",
        help="User-Agent header used for page fetches.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Additional search query. Can be passed multiple times.",
    )
    return parser.parse_args()


def normalize_text(text):
    text = unescape(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def probably_english(text):
    text = text or ""
    if not text:
        return False
    score = sum(
        ch.isascii()
        and (ch.isalpha() or ch.isdigit() or ch.isspace() or ch in "#&'!?,.-:+/()[]")
        for ch in text
    )
    return (score / max(len(text), 1)) >= 0.75


def should_keep_related(text):
    haystack = normalize_text(text).lower()
    return any(term in haystack for term in RELATED_TERMS)


def jitter_sleep(min_sleep, max_sleep):
    time.sleep(random.uniform(min_sleep, max_sleep))


class YouTubeScraper:
    def __init__(self, gl, hl, user_agent):
        self.gl = gl
        self.hl = hl
        self.headers = {"User-Agent": user_agent}

    def fetch_text(self, url):
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.read().decode("utf-8", "ignore")

    def search_shorts(self, query, max_per_query):
        url = (
            "https://www.youtube.com/results?"
            + urllib.parse.urlencode({"search_query": query, "gl": self.gl, "hl": self.hl})
        )
        html = self.fetch_text(url)
        match = re.search(r"var ytInitialData = (\{.*?\});</script>", html)
        if not match:
            return []

        payload = json.loads(match.group(1))
        candidates = []
        seen = set()

        def walk(node):
            if len(candidates) >= max_per_query:
                return
            if isinstance(node, dict):
                short_model = node.get("shortsLockupViewModel")
                if short_model:
                    endpoint = (
                        short_model.get("onTap", {})
                        .get("innertubeCommand", {})
                        .get("reelWatchEndpoint", {})
                    )
                    video_id = endpoint.get("videoId")
                    title = (
                        short_model.get("overlayMetadata", {})
                        .get("primaryText", {})
                        .get("content")
                    ) or short_model.get("accessibilityText", "").split(",")[0]
                    if video_id and video_id not in seen:
                        seen.add(video_id)
                        candidates.append(
                            {
                                "video_id": video_id,
                                "search_title": normalize_text(title),
                                "query": query,
                            }
                        )
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        return candidates

    def fetch_video_metadata(self, video_id):
        html = self.fetch_text(f"https://www.youtube.com/watch?v={video_id}")

        def find(pattern):
            match = re.search(pattern, html)
            return match.group(1) if match else None

        upload_date = find(r'<meta itemprop="uploadDate" content="([^"]+)"')
        title = normalize_text(find(r'<meta name="title" content="([^"]+)"'))
        channel = normalize_text(find(r'<link itemprop="name" content="([^"]+)"'))
        length_seconds = find(r'"lengthSeconds":"(\d+)"')
        short_description = find(r'"shortDescription":"(.*?)"')
        if short_description:
            try:
                short_description = json.loads('"' + short_description + '"')
            except Exception:
                short_description = ""
        short_description = normalize_text(short_description)
        return {
            "video_id": video_id,
            "url": f"https://www.youtube.com/shorts/{video_id}",
            "title": title,
            "channel": channel,
            "upload_date": upload_date or "",
            "length_seconds": int(length_seconds) if length_seconds else None,
            "short_description": short_description,
        }


def build_row(metadata, seed_queries, locale_basis, country_basis):
    return {
        "video_id": metadata["video_id"],
        "url": metadata["url"],
        "title": metadata["title"],
        "channel": metadata["channel"],
        "upload_date": metadata["upload_date"],
        "length_seconds": metadata["length_seconds"],
        "source_queries": "|".join(sorted(seed_queries)),
        "locale_basis": locale_basis,
        "country_basis": country_basis,
    }


def parse_upload_date(raw):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except Exception:
        return None


def main():
    args = parse_args()
    end_date = datetime.fromisoformat(args.end_date + "T23:59:59+00:00")
    queries = DEFAULT_QUERY_VARIANTS + (args.queries or [])
    scraper = YouTubeScraper(gl=args.gl, hl=args.hl, user_agent=args.user_agent)
    locale_basis = f"hl={args.hl};gl={args.gl}"
    country_basis = "best_effort_us_english_locale_only"

    candidate_map = {}
    summary = {
        "queries": queries,
        "locale_basis": locale_basis,
        "date_start_utc": DATE_START_UTC.isoformat(),
        "date_end_utc": end_date.isoformat(),
        "max_per_query": args.max_per_query,
        "max_videos": args.max_videos,
        "max_length_seconds": args.max_length_seconds,
        "raw_candidates": 0,
        "unique_candidates": 0,
        "verified_videos": 0,
        "kept_videos": 0,
        "skipped_missing_metadata": 0,
        "skipped_out_of_range": 0,
        "skipped_too_long": 0,
        "skipped_non_english": 0,
        "skipped_not_related": 0,
    }

    for query in queries:
        try:
            hits = scraper.search_shorts(query, args.max_per_query)
        except Exception:
            hits = []
        summary["raw_candidates"] += len(hits)
        for hit in hits:
            row = candidate_map.setdefault(
                hit["video_id"],
                {
                    "video_id": hit["video_id"],
                    "queries": set(),
                    "search_titles": set(),
                },
            )
            row["queries"].add(hit["query"])
            if hit["search_title"]:
                row["search_titles"].add(hit["search_title"])
        jitter_sleep(args.min_sleep, args.max_sleep)

    summary["unique_candidates"] = len(candidate_map)

    rows = []
    for index, (video_id, seed) in enumerate(candidate_map.items(), start=1):
        if index > args.max_videos:
            break

        try:
            metadata = scraper.fetch_video_metadata(video_id)
        except Exception:
            summary["skipped_missing_metadata"] += 1
            continue

        summary["verified_videos"] += 1
        uploaded_at = parse_upload_date(metadata["upload_date"])
        if uploaded_at is None:
            summary["skipped_missing_metadata"] += 1
            continue
        if uploaded_at < DATE_START_UTC or uploaded_at > end_date:
            summary["skipped_out_of_range"] += 1
            continue
        if not metadata["length_seconds"] or metadata["length_seconds"] > args.max_length_seconds:
            summary["skipped_too_long"] += 1
            continue

        text_blob = " ".join(
            [
                metadata["title"],
                metadata["short_description"],
                " ".join(sorted(seed["search_titles"])),
            ]
        )
        if not probably_english(text_blob):
            summary["skipped_non_english"] += 1
            continue
        if not should_keep_related(text_blob):
            summary["skipped_not_related"] += 1
            continue

        rows.append(
            build_row(
                metadata=metadata,
                seed_queries=seed["queries"],
                locale_basis=locale_basis,
                country_basis=country_basis,
            )
        )
        jitter_sleep(args.min_sleep, args.max_sleep)

    rows.sort(key=lambda row: (row["upload_date"], row["video_id"]))
    summary["kept_videos"] = len(rows)

    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "url",
                "title",
                "channel",
                "upload_date",
                "length_seconds",
                "source_queries",
                "locale_basis",
                "country_basis",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = Path(args.summary_output)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
