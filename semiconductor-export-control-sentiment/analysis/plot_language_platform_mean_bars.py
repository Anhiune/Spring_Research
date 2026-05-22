#!/usr/bin/env python3
"""
Create poster-style grouped mean bar charts by language for Reddit vs Bluesky.

The figure layout matches the user's requested style:
  - left panel: Reddit
  - right panel: Bluesky
  - x-axis: languages
  - grouped bars: NRC net, HF net (4-class), FinBERT net
  - error bars: approximate 95% CI of the mean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
plt.style.use("seaborn-v0_8-whitegrid")


ISO_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "sv": "Swedish",
    "ca": "Catalan",
    "id": "Indonesian",
    "ja": "Japanese",
    "pl": "Polish",
    "tr": "Turkish",
    "unknown": "Unknown",
    "und": "Undetermined",
}


def language_display_name(code: str) -> str:
    if code is None or pd.isna(code):
        return "Unknown"
    k = str(code).strip().lower().replace("_", "-")
    if not k or k == "nan":
        return "Unknown"
    if k in ISO_LANGUAGE_NAMES:
        return ISO_LANGUAGE_NAMES[k]
    base = k.split("-")[0]
    return ISO_LANGUAGE_NAMES.get(base, k.upper() if len(k) <= 4 else k.title())


def hf_net_4(df: pd.DataFrame) -> pd.Series:
    cols = (
        "hf_very_positive_score",
        "hf_positive_score",
        "hf_very_negative_score",
        "hf_negative_score",
    )
    if not all(c in df.columns for c in cols):
        raise SystemExit(f"Missing HF class columns: {cols}")
    return (
        pd.to_numeric(df[cols[0]], errors="coerce")
        + pd.to_numeric(df[cols[1]], errors="coerce")
        - pd.to_numeric(df[cols[2]], errors="coerce")
        - pd.to_numeric(df[cols[3]], errors="coerce")
    )


def load_scored(path: Path, platform: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing scored CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "lang" not in df.columns:
        raise SystemExit(f"{path} is missing `lang`.")
    df["lang"] = df["lang"].fillna("unknown").astype(str).str.strip().str.lower()
    df["hf_net_4"] = hf_net_4(df)
    df["nrc_net_sentiment"] = pd.to_numeric(df.get("nrc_net_sentiment"), errors="coerce")
    df["fb_net_sentiment"] = pd.to_numeric(df.get("fb_net_sentiment"), errors="coerce")
    df["platform"] = platform
    return df


def choose_languages(
    reddit: pd.DataFrame,
    bluesky: pd.DataFrame,
    requested: list[str],
    min_reddit: int,
    min_bluesky: int,
) -> list[str]:
    rc = reddit["lang"].value_counts()
    bc = bluesky["lang"].value_counts()
    if requested:
        langs = [lang for lang in requested if int(rc.get(lang, 0)) >= min_reddit and int(bc.get(lang, 0)) >= min_bluesky]
        if not langs:
            raise SystemExit("No requested languages met the minimum counts on both platforms.")
        return langs

    shared = sorted(set(rc.index) & set(bc.index))
    langs = [lang for lang in shared if int(rc.get(lang, 0)) >= min_reddit and int(bc.get(lang, 0)) >= min_bluesky]
    if not langs:
        raise SystemExit("No shared languages met the minimum counts on both platforms.")
    return langs


def summarize_platform(df: pd.DataFrame, langs: list[str], platform: str) -> pd.DataFrame:
    metrics = [
        ("NRC net", "nrc_net_sentiment"),
        ("HF net (4-class)", "hf_net_4"),
        ("FinBERT net", "fb_net_sentiment"),
    ]
    rows: list[dict[str, object]] = []
    for lang in langs:
        chunk = df[df["lang"] == lang].copy()
        row: dict[str, object] = {
            "platform": platform,
            "language_code": lang,
            "language_name": language_display_name(lang),
            "n_rows": int(len(chunk)),
        }
        for label, col in metrics:
            values = pd.to_numeric(chunk[col], errors="coerce").dropna().to_numpy(dtype=float)
            prefix = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            row[f"{prefix}_n"] = int(len(values))
            if len(values) == 0:
                row[f"{prefix}_mean"] = np.nan
                row[f"{prefix}_ci95"] = np.nan
            else:
                row[f"{prefix}_mean"] = float(np.mean(values))
                row[f"{prefix}_ci95"] = (
                    float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_platform_panel(ax: plt.Axes, summary: pd.DataFrame, title: str) -> None:
    labels = summary["language_name"].tolist()
    x = np.arange(len(labels))
    width = 0.22

    model_specs = [
        ("NRC net", "nrc_net_mean", "nrc_net_ci95", "#4C78A8"),
        ("HF net (4-class)", "hf_net_4_class_mean", "hf_net_4_class_ci95", "#F58518"),
        ("FinBERT net", "finbert_net_mean", "finbert_net_ci95", "#54A24B"),
    ]

    for i, (label, mean_col, ci_col, color) in enumerate(model_specs):
        offs = (i - (len(model_specs) - 1) / 2) * width
        means = pd.to_numeric(summary[mean_col], errors="coerce").to_numpy(dtype=float)
        errs = pd.to_numeric(summary[ci_col], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + offs, means, width, label=label, yerr=errs, capsize=3, color=color, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color="black")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Mean sentiment (score units)")
    ax.grid(True, axis="y", alpha=0.25)


def make_figure(
    reddit_summary: pd.DataFrame,
    bluesky_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 10.0), sharey=True)
    plot_platform_panel(axes[0], reddit_summary, "Reddit")
    plot_platform_panel(axes[1], bluesky_summary, "Bluesky")
    legend = axes[0].legend(
        loc="upper right",
        fontsize=12,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.25",
    )
    for text in legend.get_texts():
        text.set_color("black")
        text.set_fontweight("semibold")
    fig.suptitle("Mean sentiment by languages across platforms", y=0.99, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grouped mean bar charts by language for Reddit vs Bluesky.")
    ap.add_argument("--reddit-scored", type=Path, required=True)
    ap.add_argument("--bluesky-scored", type=Path, required=True)
    ap.add_argument(
        "--languages",
        default="en,fr,de,es,pt",
        help="Comma-separated language order for the chart.",
    )
    ap.add_argument("--min-reddit", type=int, default=30)
    ap.add_argument("--min-bluesky", type=int, default=100)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "analysis_output_platform_compare" / "language_mean_bars",
    )
    args = ap.parse_args()

    requested = [s.strip().lower() for s in str(args.languages).split(",") if s.strip()]
    reddit = load_scored(args.reddit_scored.resolve(), "reddit")
    bluesky = load_scored(args.bluesky_scored.resolve(), "bluesky")
    langs = choose_languages(reddit, bluesky, requested, args.min_reddit, args.min_bluesky)

    print("Plotting languages:", langs)
    reddit_summary = summarize_platform(reddit, langs, "reddit")
    bluesky_summary = summarize_platform(bluesky, langs, "bluesky")
    combined = pd.concat([reddit_summary, bluesky_summary], ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "platform_language_mean_summary.csv"
    combined.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    fig_path = args.out_dir / "platform_language_mean_bars.png"
    make_figure(reddit_summary, bluesky_summary, fig_path)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
