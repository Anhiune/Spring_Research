#!/usr/bin/env python3
"""
Compare sentiment scores (NRC net, HF 4-class net, FinBERT net) across language tags.

Typical Bluesky workflow (row-level scores required):
  1) Score posts → bluesky_all_cleaned_with_sentiment.csv (or any CSV with hf_* / nrc / fb columns)
  2) Merge language from cleaned posts on uri:
       python plot_sentiment_by_language.py \\
         --scored-csv ..\\\\bluesky_data\\\\bluesky_all_cleaned_with_sentiment.csv \\
         --meta-csv ..\\\\bluesky_data\\\\bluesky_all_cleaned.csv \\
         --merge-key uri --lang-col lang

If `lang` is always `en` (common after English-only cleaning), use optional `--infer-lang`
to guess language from `text_clean` (uses `langdetect` if installed; slower — caps rows).

Outputs (PNG):
  - sentiment_by_language_mean_bars.png  — grouped bars (mean ± 95% CI of mean)
  - sentiment_by_language_hf_net_hist.png — HF net histograms, one panel per language

Language codes (en, fr, zh-cn, …) are shown as readable names (English, French, …).
Use --show-iso-suffix to also print the raw code under the name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
plt.style.use("seaborn-v0_8-whitegrid")

# ISO-style codes → human-readable names for chart labels (lowercase keys).
ISO_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "eng": "English",
    "fr": "French",
    "fra": "French",
    "fre": "French",
    "de": "German",
    "deu": "German",
    "ger": "German",
    "es": "Spanish",
    "spa": "Spanish",
    "it": "Italian",
    "ita": "Italian",
    "pt": "Portuguese",
    "por": "Portuguese",
    "nl": "Dutch",
    "nld": "Dutch",
    "ru": "Russian",
    "rus": "Russian",
    "pl": "Polish",
    "pol": "Polish",
    "sv": "Swedish",
    "swe": "Swedish",
    "no": "Norwegian",
    "nor": "Norwegian",
    "nb": "Norwegian (Bokmål)",
    "nn": "Norwegian (Nynorsk)",
    "da": "Danish",
    "dan": "Danish",
    "fi": "Finnish",
    "fin": "Finnish",
    "cs": "Czech",
    "ces": "Czech",
    "cze": "Czech",
    "sk": "Slovak",
    "slk": "Slovak",
    "sl": "Slovenian",
    "slv": "Slovenian",
    "hu": "Hungarian",
    "hun": "Hungarian",
    "ro": "Romanian",
    "ron": "Romanian",
    "rum": "Romanian",
    "bg": "Bulgarian",
    "bul": "Bulgarian",
    "el": "Greek",
    "ell": "Greek",
    "gre": "Greek",
    "tr": "Turkish",
    "tur": "Turkish",
    "uk": "Ukrainian",
    "ukr": "Ukrainian",
    "ar": "Arabic",
    "ara": "Arabic",
    "he": "Hebrew",
    "heb": "Hebrew",
    "hi": "Hindi",
    "hin": "Hindi",
    "bn": "Bengali",
    "ben": "Bengali",
    "ta": "Tamil",
    "tam": "Tamil",
    "te": "Telugu",
    "tel": "Telugu",
    "ja": "Japanese",
    "jpn": "Japanese",
    "ko": "Korean",
    "kor": "Korean",
    "zh": "Chinese",
    "zho": "Chinese",
    "chi": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "zh-hans": "Chinese (Simplified)",
    "zh-hant": "Chinese (Traditional)",
    "vi": "Vietnamese",
    "vie": "Vietnamese",
    "th": "Thai",
    "tha": "Thai",
    "id": "Indonesian",
    "ind": "Indonesian",
    "ms": "Malay",
    "msa": "Malay",
    "fil": "Filipino",
    "tl": "Tagalog",
    "sw": "Swahili",
    "swa": "Swahili",
    "fa": "Persian",
    "fas": "Persian",
    "per": "Persian",
    "ur": "Urdu",
    "urd": "Urdu",
    "et": "Estonian",
    "est": "Estonian",
    "lv": "Latvian",
    "lav": "Latvian",
    "lt": "Lithuanian",
    "lit": "Lithuanian",
    "is": "Icelandic",
    "isl": "Icelandic",
    "ga": "Irish",
    "gle": "Irish",
    "cy": "Welsh",
    "wel": "Welsh",
    "ca": "Catalan",
    "cat": "Catalan",
    "eu": "Basque",
    "eus": "Basque",
    "hr": "Croatian",
    "hrv": "Croatian",
    "sr": "Serbian",
    "srp": "Serbian",
    "bs": "Bosnian",
    "bos": "Bosnian",
    "mk": "Macedonian",
    "mkd": "Macedonian",
    "sq": "Albanian",
    "sqi": "Albanian",
    "unknown": "Unknown",
    "und": "Undetermined",
    "undetermined": "Undetermined",
}


def language_display_name(code: str) -> str:
    """Map normalized lang tag (e.g. en, zh-cn, eng) to a readable label."""
    if code is None or pd.isna(code):
        return "Unknown"
    k = str(code).strip().lower().replace("_", "-")
    if not k or k == "nan":
        return "Unknown"
    if k in ISO_LANGUAGE_NAMES:
        return ISO_LANGUAGE_NAMES[k]
    base = k.split("-")[0]
    if base in ISO_LANGUAGE_NAMES:
        tail = k[len(base) + 1 :]
        if tail and tail not in ("latn", "cyrl"):  # script hints — keep compact
            return f"{ISO_LANGUAGE_NAMES[base]} ({tail.upper()})"
        return ISO_LANGUAGE_NAMES[base]
    if len(k) <= 4:
        return k.upper()
    return k.replace("-", " ").title()


def hf_net_4(df: pd.DataFrame) -> pd.Series:
    cols = (
        "hf_very_positive_score",
        "hf_positive_score",
        "hf_very_negative_score",
        "hf_negative_score",
    )
    if not all(c in df.columns for c in cols):
        return pd.Series(np.nan, index=df.index)
    return (
        pd.to_numeric(df[cols[0]], errors="coerce")
        + pd.to_numeric(df[cols[1]], errors="coerce")
        - pd.to_numeric(df[cols[2]], errors="coerce")
        - pd.to_numeric(df[cols[3]], errors="coerce")
    )


def infer_language(series: pd.Series, max_rows: int, seed: int) -> pd.Series:
    try:
        from langdetect import detect
    except ImportError:
        raise SystemExit("pip install langdetect (or rely on --lang-col from your CSV).")

    n = len(series)
    s = series.fillna("").astype(str).str.slice(0, 800)
    rng = np.random.default_rng(seed)
    if max_rows < n:
        pick = np.sort(rng.choice(np.arange(n), size=max_rows, replace=False))
    else:
        pick = np.arange(n)

    out = pd.Series([np.nan] * n, dtype=object, index=series.index)
    for i in pick:
        t = str(s.iloc[i])
        try:
            out.iloc[i] = detect(t) if t.strip() else "und"
        except Exception:
            out.iloc[i] = "und"
    return out


def load_frame(args: argparse.Namespace) -> pd.DataFrame:
    scored_path = args.scored_csv.resolve()
    if not scored_path.exists():
        raise SystemExit(f"Missing scored CSV: {scored_path}")

    df = pd.read_csv(scored_path, low_memory=False)
    print(f"Loaded {len(df):,} scored rows from:\n  {scored_path}")

    meta_cols = [args.merge_key, args.lang_col]

    if args.meta_csv:
        meta_path = args.meta_csv.resolve()
        if not meta_path.exists():
            raise SystemExit(f"Missing meta CSV: {meta_path}")
        extra = [args.text_col] if args.infer_lang else []
        usecols = [c for c in meta_cols + extra if c]
        meta = pd.read_csv(meta_path, usecols=lambda c: c in usecols, low_memory=False)
        keep = [args.merge_key, args.lang_col] + ([args.text_col] if args.text_col in meta.columns else [])
        meta = meta[[c for c in keep if c in meta.columns]].drop_duplicates(subset=[args.merge_key])
        n_before = len(df)
        df = df.merge(meta, on=args.merge_key, how="left", suffixes=("", "_meta"))
        missing_lang = df[args.lang_col].isna().sum()
        if missing_lang:
            print(
                f"After merge on {args.merge_key!r}: {len(df):,} rows "
                f"({missing_lang:,} missing {args.lang_col} — meta join mismatch if unexpected)."
            )
        else:
            print(f"After merge on {args.merge_key!r}: {len(df):,} rows.")

    if args.lang_col not in df.columns:
        raise SystemExit(f"No column {args.lang_col!r}. Pass --meta-csv or use --infer-lang with text column in scored CSV.")

    if args.infer_lang:
        text_col = args.text_col
        if text_col not in df.columns:
            raise SystemExit(f"--infer-lang needs column {text_col!r} on the merged/scored frame.")
        print(f"Inferring language (max {args.infer_max_rows} rows, seed={args.seed}) …")
        inferred = infer_language(df[text_col], args.infer_max_rows, args.seed)
        if args.infer_overwrite:
            df[args.lang_col] = inferred.fillna(df[args.lang_col])
        else:
            cur = df[args.lang_col].astype(str).str.strip()
            missing = cur.eq("") | cur.str.lower().eq("nan") | df[args.lang_col].isna()
            df.loc[missing, args.lang_col] = inferred[missing]

    df["_lang_norm"] = df[args.lang_col].fillna("unknown").astype(str).str.strip().str.lower()
    df["_lang_norm"] = df["_lang_norm"].replace({"": "unknown"})
    return df


def pick_top_languages(df: pd.DataFrame, top: int, min_n: int) -> list[str]:
    vc = df["_lang_norm"].value_counts()
    ok = vc[vc >= min_n].head(top).index.tolist()
    if len(ok) < 2:
        rest = vc.head(top).index.tolist()
        ok = rest if len(rest) >= 2 else vc.index.tolist()[: max(2, len(vc))]
    return ok


def build_language_summary(df: pd.DataFrame, langs: list[str]) -> pd.DataFrame:
    metrics = []
    if "nrc_net_sentiment" in df.columns:
        metrics.append(("nrc_net_sentiment", "nrc_net_sentiment"))
    if "hf_net_4" in df.columns:
        metrics.append(("hf_net_4", "hf_net_4"))
    if "fb_net_sentiment" in df.columns:
        metrics.append(("fb_net_sentiment", "fb_net_sentiment"))
    if not metrics:
        raise SystemExit("Need at least one of: nrc_net_sentiment, HF class cols, fb_net_sentiment")

    sub = df[df["_lang_norm"].isin(langs)].copy()
    rows: list[dict[str, float | int | str]] = []
    for lang in langs:
        chunk = sub[sub["_lang_norm"] == lang]
        row: dict[str, float | int | str] = {
            "language_code": lang,
            "language_name": language_display_name(lang),
            "n_rows": int(len(chunk)),
        }
        for out_col, source_col in metrics:
            values = pd.to_numeric(chunk[source_col], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{out_col}_n"] = int(len(values))
            if len(values) == 0:
                row[f"{out_col}_mean"] = np.nan
                row[f"{out_col}_std"] = np.nan
                row[f"{out_col}_ci95"] = np.nan
                continue
            row[f"{out_col}_mean"] = float(np.mean(values))
            row[f"{out_col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{out_col}_ci95"] = (
                float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_mean_bars(df: pd.DataFrame, langs: list[str], path: Path, show_iso_suffix: bool) -> None:
    metrics = []
    if "nrc_net_sentiment" in df.columns:
        metrics.append(("NRC net", "nrc_net_sentiment"))
    if "hf_net_4" in df.columns:
        metrics.append(("HF net (4-class)", "hf_net_4"))
    if "fb_net_sentiment" in df.columns:
        metrics.append(("FinBERT net", "fb_net_sentiment"))
    if not metrics:
        raise SystemExit("Need at least one of: nrc_net_sentiment, HF class cols, fb_net_sentiment")

    sub = df[df["_lang_norm"].isin(langs)].copy()
    means: dict[str, list[float]] = {m[0]: [] for m in metrics}
    stderrs: dict[str, list[float]] = {m[0]: [] for m in metrics}

    for lang in langs:
        chunk = sub[sub["_lang_norm"] == lang]
        n = len(chunk)
        for label, col in metrics:
            x = pd.to_numeric(chunk[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(x) < 2:
                means[label].append(float(np.nanmean(x)) if len(x) else np.nan)
                stderrs[label].append(np.nan)
                continue
            m = float(np.mean(x))
            se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
            means[label].append(m)
            stderrs[label].append(1.96 * se)

    x = np.arange(len(langs))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(langs)), 5))
    for i, (label, _) in enumerate(metrics):
        offs = (i - (len(metrics) - 1) / 2) * width
        y = np.array(means[label], dtype=float)
        err = np.array(stderrs[label], dtype=float)
        ax.bar(x + offs, y, width, label=label, yerr=err, capsize=3, alpha=0.88)

    ax.set_xticks(x)
    counts = df.groupby("_lang_norm").size()
    labels = []
    for lab in langs:
        pretty = language_display_name(lab)
        if show_iso_suffix:
            labels.append(f"{pretty}\n({lab})\n(n={int(counts.get(lab, 0))})")
        else:
            labels.append(f"{pretty}\n(n={int(counts.get(lab, 0))})")
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("Mean sentiment (score units)")
    ax.set_title("Mean sentiment by language (bars ± approximate 95% CI of mean)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hf_histograms(df: pd.DataFrame, langs: list[str], path: Path, show_iso_suffix: bool) -> None:
    if "hf_net_4" not in df.columns:
        print("Skip HF histogram: hf_net_4 not available.")
        return

    sub = df[df["_lang_norm"].isin(langs)].copy()
    k = len(langs)
    ncols = min(3, k)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    all_vals = pd.to_numeric(sub["hf_net_4"], errors="coerce").dropna()
    xmax = np.nanpercentile(np.abs(all_vals), 99) if len(all_vals) else 1.0
    xmax = max(xmax, 0.05)
    bins = np.linspace(-xmax, xmax, 35)

    for i, lang in enumerate(langs):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        v = pd.to_numeric(sub.loc[sub["_lang_norm"] == lang, "hf_net_4"], errors="coerce").dropna()
        ax.hist(v, bins=bins, color="C0", alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="0.35", lw=0.9)
        mu = float(np.mean(v)) if len(v) else float("nan")
        pretty = language_display_name(lang)
        iso_note = f" [{lang}]" if show_iso_suffix else ""
        ax.set_title(f"{pretty}{iso_note}\n(n={len(v)}, μ={mu:.3f})", fontsize=10)
        ax.set_xlabel("HF net (very pos + pos − very neg − neg)")
        ax.set_ylabel("Count")

    for j in range(k, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Distribution of HuggingFace multilingual net sentiment by language", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bar + histogram sentiment by language.")
    ap.add_argument("--scored-csv", type=Path, required=True, help="Row-level file with sentiment columns")
    ap.add_argument("--meta-csv", type=Path, default=None, help="Optional: uri + lang (merge keys)")
    ap.add_argument("--merge-key", default="uri", help="Join key (default uri for Bluesky)")
    ap.add_argument("--lang-col", default="lang", help="Language code column")
    ap.add_argument("--text-col", default="text_clean", help="For --infer-lang")
    ap.add_argument("--infer-lang", action="store_true", help="Use langdetect on text (needs pip install langdetect)")
    ap.add_argument(
        "--infer-overwrite",
        action="store_true",
        help="Replace existing lang labels with inferred (use when lang is always 'en' but text is mixed)",
    )
    ap.add_argument("--infer-max-rows", type=int, default=15000, help="Cap rows for inference (sampling if larger)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-langs", type=int, default=10)
    ap.add_argument("--min-per-lang", type=int, default=50)
    ap.add_argument(
        "--languages",
        default="",
        help="Optional comma-separated language codes to plot in order (for example: en,de,fr,nl,it,pt,sv,ca)",
    )
    ap.add_argument("-o", "--out-dir", type=Path, default=SCRIPT_DIR / "analysis_output" / "figures" / "sentiment_by_language")
    ap.add_argument(
        "--show-iso-suffix",
        action="store_true",
        help="Also show raw ISO code under the English name (e.g. English\\n(en))",
    )
    args = ap.parse_args()

    df = load_frame(args)
    df["hf_net_4"] = hf_net_4(df)

    nuniq = df["_lang_norm"].nunique()
    print(f"Languages (normalized): {nuniq} distinct codes in data.")
    if nuniq < 2 and not args.infer_lang:
        print(
            "\nNote: Fewer than 2 language buckets in `lang`. "
            "Your Bluesky clean export may be English-only. Options:\n"
            "  • Use raw multilingual scrape with real `lang` tags, or\n"
            "  • Re-run with --infer-lang (needs: pip install langdetect; slower).\n",
            file=sys.stderr,
        )

    requested_langs = [s.strip().lower() for s in str(args.languages).split(",") if s.strip()]
    if requested_langs:
        counts = df["_lang_norm"].value_counts(dropna=False)
        langs = [lang for lang in requested_langs if int(counts.get(lang, 0)) >= args.min_per_lang]
        skipped = [lang for lang in requested_langs if lang not in langs]
        if skipped:
            print(
                "Skipping requested languages below min-per-lang threshold "
                f"({args.min_per_lang}): {skipped}"
            )
        if not langs:
            raise SystemExit(
                "None of the requested --languages met the min-per-lang threshold. "
                "Lower --min-per-lang or change the language list."
            )
    else:
        langs = pick_top_languages(df, args.top_langs, args.min_per_lang)
    print("Plotting languages:", langs)
    print("Display labels:", [language_display_name(L) for L in langs])
    print("\nRow counts per language bucket (what ‘n’ is on the chart):")
    for lab in langs:
        ct = int((df["_lang_norm"] == lab).sum())
        print(f"  {language_display_name(lab)} ({lab}): {ct:,}")
    total_plot = sum(int((df["_lang_norm"] == lab).sum()) for lab in langs)
    print(f"  Sum across plotted languages: {total_plot:,} / total scored rows: {len(df):,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_language_summary(df, langs)
    summary_path = args.out_dir / "sentiment_by_language_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    plot_mean_bars(df, langs, args.out_dir / "sentiment_by_language_mean_bars.png", args.show_iso_suffix)
    print(f"Wrote {args.out_dir / 'sentiment_by_language_mean_bars.png'}")
    plot_hf_histograms(df, langs, args.out_dir / "sentiment_by_language_hf_net_hist.png", args.show_iso_suffix)
    out_hist = args.out_dir / "sentiment_by_language_hf_net_hist.png"
    if out_hist.exists():
        print(f"Wrote {out_hist}")
    print("Done.")


if __name__ == "__main__":
    main()
