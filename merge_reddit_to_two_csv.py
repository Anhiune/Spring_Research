#!/usr/bin/env python3
"""Merge all raw Reddit CSV exports into two files ready for cleaning.

Outputs:
- reddit_scraper/output/reddit_all_title_raw.csv
- reddit_scraper/output/reddit_all_selftext_raw.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reddit_scraper" / "output"


def collect_input_files() -> list[Path]:
    files = []
    for p in sorted(OUT_DIR.glob("reddit*.csv")):
        name = p.name.lower()
        # Skip already-cleaned or already-merged outputs.
        if name.endswith("_cleaned.csv"):
            continue
        if name in {
            "reddit_all_title_raw.csv",
            "reddit_all_selftext_raw.csv",
            "reddit_all_merged_raw.csv",
            "reddit_all_submissions_raw.csv",
            "reddit_all_comments_raw.csv",
        }:
            continue
        files.append(p)
    return files


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    required = ["submission_id", "comment_id", "timestamp", "subreddit", "text_type", "text"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    # Keep canonical columns first; preserve any extras after them.
    extras = [c for c in df.columns if c not in required]
    return df[required + extras]


def main() -> None:
    files = collect_input_files()
    if not files:
        raise SystemExit("No input Reddit CSV files found.")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df = ensure_schema(df)
        df["source_file"] = f.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Normalize text_type for reliable splitting.
    merged["text_type"] = merged["text_type"].astype(str).str.strip().str.lower()

    # Keep rows with usable text for cleaning.
    merged = merged[merged["text"].notna()].copy()
    merged["text"] = merged["text"].astype(str)
    merged = merged[merged["text"].str.strip() != ""]

    title_rows = merged[merged["text_type"] == "title"].copy()
    selftext_rows = merged[merged["text_type"] == "selftext"].copy()

    merged_path = OUT_DIR / "reddit_all_merged_raw.csv"
    title_path = OUT_DIR / "reddit_all_title_raw.csv"
    selftext_path = OUT_DIR / "reddit_all_selftext_raw.csv"

    merged.to_csv(merged_path, index=False)
    title_rows.to_csv(title_path, index=False)
    selftext_rows.to_csv(selftext_path, index=False)

    print(f"Input files merged: {len(files)}")
    print(f"Merged rows: {len(merged):,}")
    print(f"Title rows: {len(title_rows):,}")
    print(f"Selftext rows: {len(selftext_rows):,}")
    print(f"Wrote: {merged_path}")
    print(f"Wrote: {title_path}")
    print(f"Wrote: {selftext_path}")


if __name__ == "__main__":
    main()
