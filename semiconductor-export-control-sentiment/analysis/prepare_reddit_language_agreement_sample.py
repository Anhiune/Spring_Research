#!/usr/bin/env python3
"""
Build a relaxed Reddit multilingual sample for cross-language sentiment analysis.

Why this exists:
  - The older "high confidence" Reddit sample was very strict for non-English rows.
  - In particular, requiring non-ASCII characters removed many valid posts written
    in German, French, Spanish, Portuguese, and Italian.
  - This script keeps a quality screen, but relaxes that specific rule:
      * English: keep cleaner label + minimum text length, then sample if desired
      * Non-English: keep cleaner label + minimum text length + detector agreement

Detector agreement means:
  cleaner `lang` == `langdetect` == `langid`

Outputs:
  1) sample CSV with the original Reddit cleaned columns plus audit columns
  2) meta CSV summarizing source / retained rows by language
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

DEFAULT_INPUT = REPO_DIR / "reddit_scraper" / "output" / "reddit_all_raw_multilang_cleaned.csv"
DEFAULT_OUTPUT = REPO_DIR / "reddit_scraper" / "output" / "reddit_language_relaxed_len40_agree_sample.csv"
DEFAULT_META = REPO_DIR / "reddit_scraper" / "output" / "reddit_language_relaxed_len40_agree_sample_meta.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a relaxed Reddit multilingual sample with detector agreement."
    )
    ap.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--meta-csv", type=Path, default=DEFAULT_META)
    ap.add_argument(
        "--languages",
        default="en,de,es,it,fr,pt,nl,ca,sv",
        help="Comma-separated cleaner language labels to consider.",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Minimum text length for inclusion in candidate pools.",
    )
    ap.add_argument(
        "--min-consensus-rows",
        type=int,
        default=30,
        help="Drop non-English languages below this detector-consensus count.",
    )
    ap.add_argument(
        "--english-max-rows",
        type=int,
        default=200,
        help="Cap English rows to this many after filtering. Use 0 for no cap.",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def detect_langdetect(text: str) -> str:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 42
    try:
        return detect(text)
    except Exception:
        return "und"


def detect_langid(text: str) -> str:
    import langid

    try:
        return langid.classify(text)[0]
    except Exception:
        return "und"


def sample_english(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows and len(df) > max_rows:
        return df.sample(n=max_rows, random_state=seed).copy()
    return df.copy()


def build_meta_row(
    *,
    language_code: str,
    total_rows: int,
    candidate_rows: int,
    consensus_rows: int,
    sample_rows: int,
    rule: str,
    included: bool,
) -> dict[str, object]:
    return {
        "language_code": language_code,
        "source_rows_total": int(total_rows),
        "candidate_rows_after_min_chars": int(candidate_rows),
        "consensus_rows": int(consensus_rows),
        "sample_rows": int(sample_rows),
        "included": bool(included),
        "rule": rule,
    }


def main() -> None:
    args = parse_args()

    langs = [s.strip().lower() for s in str(args.languages).split(",") if s.strip()]
    if "en" not in langs:
        langs = ["en"] + langs

    print(f"Loading cleaned Reddit rows: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)
    if "lang" not in df.columns or "text_clean" not in df.columns:
        raise SystemExit("Input CSV must contain `lang` and `text_clean` columns.")

    df["lang"] = df["lang"].fillna("und").astype(str).str.strip().str.lower()
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    df["text_len"] = df["text_clean"].str.len()

    total_by_lang = df["lang"].value_counts(dropna=False).to_dict()
    candidate = df[df["lang"].isin(langs)].copy()
    candidate = candidate[candidate["text_len"] >= args.min_chars].copy()

    out_frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, object]] = []

    # English: trust the cleaner label, require sufficient text, then cap if requested.
    en_pool = candidate[candidate["lang"] == "en"].copy()
    en_sample = sample_english(en_pool, args.english_max_rows, args.seed)
    if not en_sample.empty:
        en_sample["langdetect_label"] = pd.NA
        en_sample["langid_label"] = pd.NA
        en_sample["detector_agree"] = pd.NA
        en_sample["consensus_with_cleaner"] = pd.NA
        en_sample["sample_rule"] = (
            f"cleaner_lang=en + text_len>={args.min_chars}"
            + (f" + sample_max={args.english_max_rows}" if args.english_max_rows else "")
        )
        out_frames.append(en_sample)

    meta_rows.append(
        build_meta_row(
            language_code="en",
            total_rows=int(total_by_lang.get("en", 0)),
            candidate_rows=int(len(en_pool)),
            consensus_rows=int(len(en_pool)),
            sample_rows=int(len(en_sample)),
            included=not en_sample.empty,
            rule=en_sample["sample_rule"].iloc[0] if not en_sample.empty else "excluded",
        )
    )

    # Non-English: require agreement between cleaner label, langdetect, and langid.
    non_en_pool = candidate[candidate["lang"] != "en"].copy()
    if not non_en_pool.empty:
        print(
            "Running detector agreement on non-English candidates "
            f"({len(non_en_pool):,} rows across {non_en_pool['lang'].nunique()} languages)..."
        )
        non_en_pool["langdetect_label"] = non_en_pool["text_clean"].map(detect_langdetect)
        non_en_pool["langid_label"] = non_en_pool["text_clean"].map(detect_langid)
        non_en_pool["detector_agree"] = non_en_pool["langdetect_label"] == non_en_pool["langid_label"]
        non_en_pool["consensus_with_cleaner"] = (
            non_en_pool["detector_agree"] & (non_en_pool["langdetect_label"] == non_en_pool["lang"])
        )

    for lang in [lang for lang in langs if lang != "en"]:
        lang_pool = non_en_pool[non_en_pool["lang"] == lang].copy()
        lang_consensus = lang_pool[lang_pool["consensus_with_cleaner"]].copy()
        include_lang = len(lang_consensus) >= args.min_consensus_rows

        if include_lang:
            lang_consensus["sample_rule"] = (
                f"cleaner_lang={lang} + text_len>={args.min_chars} + "
                "langdetect==langid==cleaner_lang"
            )
            out_frames.append(lang_consensus)
            rule = lang_consensus["sample_rule"].iloc[0]
        else:
            rule = (
                f"dropped_below_min_consensus_{args.min_consensus_rows}"
                f" (had {len(lang_consensus)})"
            )

        meta_rows.append(
            build_meta_row(
                language_code=lang,
                total_rows=int(total_by_lang.get(lang, 0)),
                candidate_rows=int(len(lang_pool)),
                consensus_rows=int(len(lang_consensus)),
                sample_rows=int(len(lang_consensus) if include_lang else 0),
                included=include_lang,
                rule=rule,
            )
        )

    if not out_frames:
        raise SystemExit("No rows met the sampling rules.")

    out = pd.concat(out_frames, ignore_index=True)
    sort_cols = [c for c in ("lang", "timestamp", "submission_id", "comment_id") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    meta = pd.DataFrame(meta_rows)
    kept_langs = meta.loc[meta["included"], "language_code"].tolist()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.meta_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    meta.to_csv(args.meta_csv, index=False)

    print(f"Wrote sample CSV: {args.output_csv}")
    print(f"Wrote meta CSV:   {args.meta_csv}")
    print("\nSample sizes by kept language:")
    print(meta.loc[meta["included"], ["language_code", "sample_rows"]].to_string(index=False))
    print(f"\nLanguages kept: {kept_langs}")
    print(f"Total sample rows: {len(out):,}")


if __name__ == "__main__":
    main()
