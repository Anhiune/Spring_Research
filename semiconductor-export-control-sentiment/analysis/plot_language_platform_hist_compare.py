#!/usr/bin/env python3
"""
Compare HF multilingual sentiment histograms by language across Reddit and Bluesky.

Inputs:
  - Row-level Reddit scored CSV with HF class scores and `lang`
  - Row-level Bluesky scored CSV with HF class scores and `lang`

Outputs:
  - platform_language_hist_side_by_side.png
      One row per language, Reddit on the left and Bluesky on the right
  - platform_language_hist_overlay.png
      One panel per language with Reddit / Bluesky overlays
  - platform_language_hist_counts.csv
      Counts used for each plotted language / platform
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
        raise SystemExit(f"Missing HF class score columns: {cols}")
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


def pooled_bins(x1: np.ndarray, x2: np.ndarray, n_bins: int) -> np.ndarray:
    pooled = np.concatenate([x1, x2])
    pooled = pooled[np.isfinite(pooled)]
    if len(pooled) == 0:
        return np.linspace(-1, 1, n_bins + 1)
    xmax = np.nanpercentile(np.abs(pooled), 99)
    xmax = max(float(xmax), 0.05)
    return np.linspace(-xmax, xmax, n_bins + 1)


def plot_side_by_side(
    reddit: pd.DataFrame,
    bluesky: pd.DataFrame,
    langs: list[str],
    out_path: Path,
    bins: int,
) -> None:
    nrows = len(langs)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, max(3.0 * nrows, 4.5)), squeeze=False)

    for i, lang in enumerate(langs):
        r = pd.to_numeric(reddit.loc[reddit["lang"] == lang, "hf_net_4"], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(bluesky.loc[bluesky["lang"] == lang, "hf_net_4"], errors="coerce").dropna().to_numpy(dtype=float)
        hist_bins = pooled_bins(r, b, bins)
        label = language_display_name(lang)

        ax_r = axes[i][0]
        ax_b = axes[i][1]

        ax_r.hist(r, bins=hist_bins, color="#C44E52", alpha=0.82, edgecolor="white", linewidth=0.5)
        ax_r.axvline(0, color="0.35", lw=0.9)
        ax_r.axvline(float(np.mean(r)), color="#7A1F24", lw=1.5, ls="--")
        ax_r.set_title(f"{label} — Reddit (n={len(r)})", fontsize=11)
        ax_r.set_xlabel("HF net")
        ax_r.set_ylabel("Count")

        ax_b.hist(b, bins=hist_bins, color="#4C72B0", alpha=0.82, edgecolor="white", linewidth=0.5)
        ax_b.axvline(0, color="0.35", lw=0.9)
        ax_b.axvline(float(np.mean(b)), color="#1C4E80", lw=1.5, ls="--")
        ax_b.set_title(f"{label} — Bluesky (n={len(b)})", fontsize=11)
        ax_b.set_xlabel("HF net")
        ax_b.set_ylabel("Count")

    fig.suptitle("HF multilingual sentiment distribution by language: Reddit vs Bluesky", y=1.01, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(
    reddit: pd.DataFrame,
    bluesky: pd.DataFrame,
    langs: list[str],
    out_path: Path,
    bins: int,
) -> None:
    n = len(langs)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3.5 * nrows, 4.5)), squeeze=False)

    for i, lang in enumerate(langs):
        r = pd.to_numeric(reddit.loc[reddit["lang"] == lang, "hf_net_4"], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(bluesky.loc[bluesky["lang"] == lang, "hf_net_4"], errors="coerce").dropna().to_numpy(dtype=float)
        hist_bins = pooled_bins(r, b, bins)
        ax = axes[i // ncols][i % ncols]
        label = language_display_name(lang)

        ax.hist(r, bins=hist_bins, density=True, color="#C44E52", alpha=0.45, label=f"Reddit (n={len(r)})", edgecolor="none")
        ax.hist(b, bins=hist_bins, density=True, color="#4C72B0", alpha=0.45, label=f"Bluesky (n={len(b)})", edgecolor="none")
        ax.axvline(0, color="0.35", lw=0.9)
        ax.axvline(float(np.mean(r)), color="#7A1F24", lw=1.4, ls="--")
        ax.axvline(float(np.mean(b)), color="#1C4E80", lw=1.4, ls="--")
        ax.set_title(f"{label}", fontsize=11)
        ax.set_xlabel("HF net")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left", fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Overlay comparison of HF multilingual sentiment by language", y=1.01, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_counts(reddit: pd.DataFrame, bluesky: pd.DataFrame, langs: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lang in langs:
        r = pd.to_numeric(reddit.loc[reddit["lang"] == lang, "hf_net_4"], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(bluesky.loc[bluesky["lang"] == lang, "hf_net_4"], errors="coerce").dropna().to_numpy(dtype=float)
        rows.extend(
            [
                {
                    "language_code": lang,
                    "language_name": language_display_name(lang),
                    "platform": "reddit",
                    "n_rows": int(len(r)),
                    "hf_net_mean": float(np.mean(r)) if len(r) else np.nan,
                    "hf_net_std": float(np.std(r, ddof=1)) if len(r) > 1 else np.nan,
                },
                {
                    "language_code": lang,
                    "language_name": language_display_name(lang),
                    "platform": "bluesky",
                    "n_rows": int(len(b)),
                    "hf_net_mean": float(np.mean(b)) if len(b) else np.nan,
                    "hf_net_std": float(np.std(b, ddof=1)) if len(b) > 1 else np.nan,
                },
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare language-level HF histograms across Reddit and Bluesky.")
    ap.add_argument("--reddit-scored", type=Path, required=True)
    ap.add_argument("--bluesky-scored", type=Path, required=True)
    ap.add_argument(
        "--languages",
        default="",
        help="Optional comma-separated language codes to compare. Default: shared languages meeting thresholds.",
    )
    ap.add_argument("--min-reddit", type=int, default=30)
    ap.add_argument("--min-bluesky", type=int, default=100)
    ap.add_argument("--bins", type=int, default=35)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "analysis_output_platform_compare" / "language_histograms",
    )
    args = ap.parse_args()

    requested = [s.strip().lower() for s in str(args.languages).split(",") if s.strip()]
    reddit = load_scored(args.reddit_scored.resolve(), "reddit")
    bluesky = load_scored(args.bluesky_scored.resolve(), "bluesky")
    langs = choose_languages(reddit, bluesky, requested, args.min_reddit, args.min_bluesky)

    print("Comparing languages:", langs)
    print("Display labels:", [language_display_name(lang) for lang in langs])

    counts = build_counts(reddit, bluesky, langs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts_path = args.out_dir / "platform_language_hist_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"Wrote {counts_path}")

    side_path = args.out_dir / "platform_language_hist_side_by_side.png"
    plot_side_by_side(reddit, bluesky, langs, side_path, args.bins)
    print(f"Wrote {side_path}")

    overlay_path = args.out_dir / "platform_language_hist_overlay.png"
    plot_overlay(reddit, bluesky, langs, overlay_path, args.bins)
    print(f"Wrote {overlay_path}")


if __name__ == "__main__":
    main()
