#!/usr/bin/env python3
"""Create the baseline TikTok test-set table.

Baseline now means the raw fields we want to pass into the LLM directly:
caption and hashtags, plus identifiers/metadata needed for joining later
audio, visual, and human-label outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TEST_TAGS = ["whatieatinaday", "whatieatinday", "wieiad"]
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def split_hashtags(value: Any) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    return [part.strip().lower().lstrip("#") for part in text.split(",") if part.strip()]


def matching_test_tags(row: pd.Series, include_tags: set[str]) -> tuple[list[str], list[str]]:
    source_tag = normalize_text(row.get("source_hashtag_query")).lower().lstrip("#")
    actual_tags = split_hashtags(row.get("hashtags"))
    matched_source = [source_tag] if source_tag in include_tags else []
    matched_actual = sorted(set(actual_tags).intersection(include_tags))
    return matched_source, matched_actual


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def extract_features(row: pd.Series) -> dict[str, Any]:
    caption = normalize_text(row.get("caption"))
    hashtags = split_hashtags(row.get("hashtags"))
    view_count = safe_int(row.get("view_count"))
    like_count = safe_int(row.get("like_count"))
    comment_count = safe_int(row.get("comment_count"))
    share_count = safe_int(row.get("share_count"))

    return {
        "video_id": str(row["video_id"]),
        "video_url": row["video_url"],
        "created_at": row["created_at"],
        "author_username": row["author_username"],
        "source_hashtag_query": row["source_hashtag_query"],
        "caption": caption,
        "hashtags": ", ".join(hashtags),
        "caption_word_count": len(TOKEN_RE.findall(caption)),
        "hashtag_count": len(hashtags),
        "view_count": view_count,
        "like_count": like_count,
        "comment_count": comment_count,
        "share_count": share_count,
        "like_rate": safe_ratio(like_count, view_count),
        "comment_rate": safe_ratio(comment_count, view_count),
        "share_rate": safe_ratio(share_count, view_count),
        "engagement_rate": safe_ratio(like_count + comment_count + share_count, view_count),
    }


def filter_test_set(df: pd.DataFrame, include_tags: list[str], match_mode: str) -> pd.DataFrame:
    include_tag_set = {tag.lower().lstrip("#") for tag in include_tags}
    rows = []
    for _, row in df.iterrows():
        matched_source, matched_actual = matching_test_tags(row, include_tag_set)
        if match_mode == "source":
            keep = bool(matched_source)
        elif match_mode == "actual":
            keep = bool(matched_actual)
        else:
            keep = bool(matched_source or matched_actual)
        if not keep:
            continue
        row = row.copy()
        row["matched_source_tags"] = ", ".join(matched_source)
        row["matched_actual_tags"] = ", ".join(matched_actual)
        rows.append(row)
    return pd.DataFrame(rows)


def clean_test_set(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["video_id"]).drop_duplicates(subset=["video_url"]).copy()
    duplicate_rows_removed = before - len(df)

    repeated_authors = df["author_username"].value_counts()
    repeated_authors = repeated_authors[repeated_authors > 1]
    repeated_author_count = len(repeated_authors)
    repeated_author_video_count = int(repeated_authors.sum())

    df.attrs["duplicate_rows_removed"] = duplicate_rows_removed
    df.attrs["repeated_author_count"] = repeated_author_count
    df.attrs["repeated_author_video_count"] = repeated_author_video_count
    return df


def write_repeated_author_report(features: pd.DataFrame, output_path: str) -> None:
    repeated = features["author_username"].value_counts()
    repeated = repeated[repeated > 1]
    report_rows = []
    for author, count in repeated.items():
        author_rows = features[features["author_username"] == author]
        report_rows.append(
            {
                "author_username": author,
                "video_count": int(count),
                "video_ids": ";".join(author_rows["video_id"].astype(str)),
                "view_counts": ";".join(author_rows["view_count"].astype(str)),
            }
        )
    pd.DataFrame(report_rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tiktok_videos_2025_07_to_today.csv")
    parser.add_argument(
        "--include-tags",
        nargs="+",
        default=DEFAULT_TEST_TAGS,
        help="Hashtags to include in the test set.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["either", "source", "actual"],
        default="either",
        help="Match source_hashtag_query, actual hashtags, or either.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Optional cap after filtering and sorting. 0 means keep all matched rows.",
    )
    parser.add_argument("--sort-by", default="view_count")
    parser.add_argument("--output-csv", default="testset_baseline_features.csv")
    parser.add_argument("--output-json", default="testset_baseline_feature_summary.json")
    parser.add_argument("--author-report-csv", default="testset_repeated_authors.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.sort_by not in df.columns:
        raise ValueError(f"Sort column not found: {args.sort_by}")

    filtered = filter_test_set(df, args.include_tags, args.match_mode)
    filtered = filtered.sort_values(args.sort_by, ascending=False).copy()
    filtered = clean_test_set(filtered)
    if args.top_n > 0:
        filtered = filtered.head(args.top_n).copy()

    features = pd.DataFrame([extract_features(row) for _, row in filtered.iterrows()])
    if not features.empty:
        matched_source = filtered["matched_source_tags"].reset_index(drop=True)
        matched_actual = filtered["matched_actual_tags"].reset_index(drop=True)
        features.insert(6, "matched_source_tags", matched_source)
        features.insert(7, "matched_actual_tags", matched_actual)
    features.to_csv(args.output_csv, index=False)

    summary_columns = [
        "video_id",
        "video_url",
        "author_username",
        "view_count",
        "caption",
        "hashtags",
        "matched_source_tags",
        "matched_actual_tags",
    ]
    summary = features[summary_columns].to_dict(orient="records")
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_repeated_author_report(features, args.author_report_csv)

    repeated_authors_after_cleaning = features["author_username"].value_counts()
    repeated_authors_after_cleaning = repeated_authors_after_cleaning[
        repeated_authors_after_cleaning > 1
    ]

    print(f"Wrote {len(features)} rows to {args.output_csv}")
    print(f"Included tags: {', '.join(args.include_tags)}")
    print(f"Match mode: {args.match_mode}")
    print(f"Duplicate video/url rows removed: {filtered.attrs.get('duplicate_rows_removed', 0)}")
    print(
        "Authors with >1 matched video after duplicate cleaning: "
        f"{filtered.attrs.get('repeated_author_count', 0)}"
    )
    print(
        "Videos from repeated authors after duplicate cleaning: "
        f"{filtered.attrs.get('repeated_author_video_count', 0)}"
    )
    print(f"Authors with >1 video after duplicate cleaning: {len(repeated_authors_after_cleaning)}")
    print(f"Repeated-author report: {args.author_report_csv}")
    print(f"Wrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
