#!/usr/bin/env python3
"""
Merge Reddit language labels back onto the full row-level sentiment export.

Why this exists:
  - `reddit_NEW_RUN_with_sentiment.xlsx` has NRC / HF / FinBERT scores but no `lang`.
  - `reddit_all_raw_multilang_cleaned.csv` has `lang` but fewer columns.
  - The poster's cross-language figures need both in the same row-level file.

Merge strategy:
  1) Exact match on timestamp + subreddit + text_type + normalized text_clean
  2) Fallback to timestamp + subreddit + text_type only when that key maps to
     a single language in the multilingual cleaned file

This favors precision over forcing every row to match. The script writes a
simple JSON report so downstream analysis can see merge coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCORED_XLSX = SCRIPT_DIR.parent / "reddit_scraper" / "data" / "reddit_NEW_RUN_with_sentiment.xlsx"
DEFAULT_LANG_CSV = SCRIPT_DIR.parent / "reddit_scraper" / "output" / "reddit_all_raw_multilang_cleaned.csv"
DEFAULT_OUT_CSV = SCRIPT_DIR.parent / "reddit_scraper" / "output" / "reddit_NEW_RUN_with_sentiment_lang.csv"
DEFAULT_REPORT_JSON = SCRIPT_DIR.parent / "reddit_scraper" / "output" / "reddit_NEW_RUN_with_sentiment_lang_report.json"


def normalize_text(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype("string")
    # Normalize punctuation that differs between exports.
    s = s.str.replace("\u201c", '"', regex=False)
    s = s.str.replace("\u201d", '"', regex=False)
    s = s.str.replace("\u2018", "'", regex=False)
    s = s.str.replace("\u2019", "'", regex=False)
    s = s.str.replace("\u00a0", " ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def load_scored_rows(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"subreddit": "string", "text_type": "string", "text_clean": "string"})
    df["timestamp"] = pd.to_datetime(df["time"], utc=True, errors="coerce").astype("int64") // 10**9
    df["_subreddit_norm"] = normalize_text(df["subreddit"]).str.lower()
    df["_text_type_norm"] = normalize_text(df["text_type"]).str.lower()
    df["_text_clean_norm"] = normalize_text(df["text_clean"])
    return df


def load_language_rows(path: Path) -> pd.DataFrame:
    usecols = ["timestamp", "subreddit", "text_type", "text_clean", "lang"]
    df = pd.read_csv(
        path,
        usecols=usecols,
        low_memory=False,
        dtype={"subreddit": "string", "text_type": "string", "text_clean": "string", "lang": "string"},
    )
    df["_subreddit_norm"] = normalize_text(df["subreddit"]).str.lower()
    df["_text_type_norm"] = normalize_text(df["text_type"]).str.lower()
    df["_text_clean_norm"] = normalize_text(df["text_clean"])
    return df


def build_exact_language_map(meta: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["timestamp", "_subreddit_norm", "_text_type_norm", "_text_clean_norm"]
    return meta[key_cols + ["lang"]].drop_duplicates(subset=key_cols)


def build_unique_bucket_language_map(meta: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["timestamp", "_subreddit_norm", "_text_type_norm"]
    counts = (
        meta[key_cols + ["lang"]]
        .drop_duplicates()
        .groupby(key_cols, dropna=False)["lang"]
        .nunique(dropna=True)
        .reset_index(name="_lang_nunique")
    )
    single = counts[counts["_lang_nunique"] == 1][key_cols]
    langs = meta[key_cols + ["lang"]].drop_duplicates()
    return single.merge(langs, on=key_cols, how="left").drop_duplicates(subset=key_cols)


def merge_language(scored: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    key4 = ["timestamp", "_subreddit_norm", "_text_type_norm", "_text_clean_norm"]
    key3 = ["timestamp", "_subreddit_norm", "_text_type_norm"]

    exact_map = build_exact_language_map(meta)
    out = scored.merge(exact_map, on=key4, how="left")
    out["lang_merge_method"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[out["lang"].notna(), "lang_merge_method"] = "exact_text"

    unresolved = out["lang"].isna()
    if unresolved.any():
        bucket_map = build_unique_bucket_language_map(meta)
        rescued = out.loc[unresolved, key3].merge(bucket_map, on=key3, how="left")
        rescued_lang = rescued["lang"]
        out.loc[unresolved, "lang"] = rescued_lang.values
        out.loc[unresolved & out["lang"].notna(), "lang_merge_method"] = "unique_bucket"

    return out


def build_report(out: pd.DataFrame) -> dict:
    matched = out["lang"].notna()
    report = {
        "rows_total": int(len(out)),
        "rows_with_lang": int(matched.sum()),
        "rows_without_lang": int((~matched).sum()),
        "match_rate": float(round(float(matched.mean()), 6)),
        "merge_method_counts": {
            str(k): int(v)
            for k, v in out["lang_merge_method"].fillna("unmatched").value_counts(dropna=False).items()
        },
        "language_counts_top20": {
            str(k): int(v) for k, v in out["lang"].fillna("missing").value_counts(dropna=False).head(20).items()
        },
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge Reddit sentiment rows with language labels.")
    ap.add_argument("--scored-xlsx", type=Path, default=DEFAULT_SCORED_XLSX)
    ap.add_argument("--lang-csv", type=Path, default=DEFAULT_LANG_CSV)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    args = ap.parse_args()

    print(f"Loading scored Reddit rows: {args.scored_xlsx}")
    scored = load_scored_rows(args.scored_xlsx)
    print(f"Scored rows: {len(scored):,}")

    print(f"Loading multilingual language rows: {args.lang_csv}")
    meta = load_language_rows(args.lang_csv)
    print(f"Language rows: {len(meta):,}")

    print("Merging language labels...")
    out = merge_language(scored, meta)

    drop_cols = ["_subreddit_norm", "_text_type_norm", "_text_clean_norm"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote merged CSV: {args.out_csv}")

    report = build_report(out)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote merge report: {args.report_json}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
