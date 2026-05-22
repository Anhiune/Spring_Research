#!/usr/bin/env python3
"""
Diagnostic plots for daily sentiment: time series (with rolling trends) and
within-model correlation heatmaps (NRC, HuggingFace, FinBERT).

Reads firm complete panels (level columns only, no lag/_lag1).

Outputs (PNG) under analysis_output/figures/sentiment_diagnostics/:
  - ts_{TICKER}_nets.png       — NRC net, HF net, FB net + rolling means + linear trend hints
  - ts_{TICKER}_posts_volatility.png — n_posts (left) and |ret_1d| (right) vs date
  - corr_{TICKER}_nrc.png      — NRC lexicon dimensions + net
  - corr_{TICKER}_hf.png       — HF class probabilities + hf_net_4
  - corr_{TICKER}_fb.png       — FinBERT scores + net
  - corr_{TICKER}_cross_net.png — small heatmap: nets + n_posts + |ret|

Within-panel correlations are **same-day** among sentiment columns. For **sentiment at day t vs stock return on day t+1** (forecast alignment), run:

  python plot_correlation_sentiment_vs_fwd_return.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DEFAULT = SCRIPT_DIR / "reddit_prepared"
OUT_DIR_DEFAULT = SCRIPT_DIR / "analysis_output" / "figures" / "sentiment_diagnostics"

NRC_COLS: tuple[str, ...] = (
    "nrc_anger",
    "nrc_anticipation",
    "nrc_disgust",
    "nrc_fear",
    "nrc_joy",
    "nrc_sadness",
    "nrc_surprise",
    "nrc_trust",
    "nrc_positive",
    "nrc_negative",
    "nrc_net_sentiment",
)

HF_COLS: tuple[str, ...] = (
    "hf_neutral_score",
    "hf_negative_score",
    "hf_very_negative_score",
    "hf_positive_score",
    "hf_very_positive_score",
)

FB_COLS: tuple[str, ...] = (
    "fb_negative_score",
    "fb_neutral_score",
    "fb_positive_score",
    "fb_net_sentiment",
)


def try_seaborn_heatmap(ax, corr: pd.DataFrame) -> None:
    import seaborn as sns  # type: ignore

    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-1,
        vmax=1,
        square=True,
    )


def matplotlib_heatmap(ax, corr: pd.DataFrame) -> None:
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def add_hf_net_4(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    need = (
        "hf_very_positive_score",
        "hf_positive_score",
        "hf_very_negative_score",
        "hf_negative_score",
    )
    if all(c in o.columns for c in need):
        o["hf_net_4"] = (
            pd.to_numeric(o[need[0]], errors="coerce")
            + pd.to_numeric(o[need[1]], errors="coerce")
            - pd.to_numeric(o[need[2]], errors="coerce")
            - pd.to_numeric(o[need[3]], errors="coerce")
        )
    else:
        o["hf_net_4"] = np.nan
    return o


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    m = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return x * 0.0
    return (x - m) / sd


def plot_time_series_nets(df: pd.DataFrame, ticker: str, roll: int, out: Path) -> None:
    d = df.sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"], format="mixed", utc=False)

    for col in ("nrc_net_sentiment", "fb_net_sentiment"):
        if col not in d.columns:
            d[col] = np.nan
    d = add_hf_net_4(d)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    labels = [
        ("nrc_net_sentiment", "NRC net", "#1f77b4"),
        ("hf_net_4", "HF net (4-class)", "#ff7f0e"),
        ("fb_net_sentiment", "FinBERT net", "#2ca02c"),
    ]
    for col, lab, c in labels:
        if col not in d.columns:
            continue
        s = pd.to_numeric(d[col], errors="coerce")
        zz = zscore(s)
        ax.plot(d["date"], zz, alpha=0.35, color=c, linewidth=0.8, label=f"{lab} (z)")
        roll_s = zz.rolling(roll, min_periods=max(3, roll // 3)).mean()
        ax.plot(d["date"], roll_s, color=c, linewidth=2.0, label=f"{lab} roll-{roll}d")

    ax.axhline(0, color="0.5", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Z-scored sentiment")
    ax.set_title(f"{ticker} — daily nets (thin) vs {roll}-day rolling mean (thick); z-score for overlay")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if "n_posts" in d.columns:
        ax2.bar(
            d["date"],
            pd.to_numeric(d["n_posts"], errors="coerce"),
            width=1.5,
            color="steelblue",
            alpha=0.45,
            label="n_posts",
        )
    ax2.set_ylabel("Posts / day")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_posts_return_noise(df: pd.DataFrame, ticker: str, out: Path) -> None:
    d = df.sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"], format="mixed", utc=False)
    fig, ax1 = plt.subplots(figsize=(11, 4))
    if "n_posts" in d.columns:
        ax1.plot(d["date"], pd.to_numeric(d["n_posts"], errors="coerce"), color="steelblue", alpha=0.7, label="n_posts")
    ax1.set_ylabel("n_posts", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    if "ret_1d" in d.columns:
        ax2.plot(
            d["date"],
            pd.to_numeric(d["ret_1d"], errors="coerce").abs(),
            color="coral",
            alpha=0.5,
            linewidth=0.8,
            label="|ret_1d|",
        )
    ax2.set_ylabel("|daily return|", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax1.set_title(f"{ticker} — posting volume vs absolute return (noise / intensity proxy)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def corr_heatmap(df: pd.DataFrame, cols: list[str], title: str, out: Path) -> None:
    present = [c for c in cols if c in df.columns]
    if len(present) < 2:
        return
    sub = df[present].apply(pd.to_numeric, errors="coerce")
    c = sub.corr()
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(present)), max(5, 0.4 * len(present))))
    try:
        try_seaborn_heatmap(ax, c)
    except ImportError:
        matplotlib_heatmap(ax, c)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_net(df: pd.DataFrame, ticker: str, out: Path) -> None:
    d = add_hf_net_4(df.copy())
    cols = []
    if "nrc_net_sentiment" in d.columns:
        cols.append("nrc_net_sentiment")
    if "hf_net_4" in d.columns:
        cols.append("hf_net_4")
    if "fb_net_sentiment" in d.columns:
        cols.append("fb_net_sentiment")
    if "n_posts" in d.columns:
        cols.append("n_posts")
    if "ret_1d" in d.columns:
        d["abs_ret"] = pd.to_numeric(d["ret_1d"], errors="coerce").abs()
        cols.append("abs_ret")
    if len(cols) < 2:
        return
    sub = d[cols].apply(pd.to_numeric, errors="coerce")
    c = sub.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        try_seaborn_heatmap(ax, c)
    except ImportError:
        matplotlib_heatmap(ax, c)
    ax.set_title(f"{ticker} — cross-model nets + volume + |return|")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sentiment diagnostic plots.")
    ap.add_argument(
        "--prep-dir",
        type=Path,
        default=PREP_DEFAULT,
        help="Folder with merged_*_market_sentiment*.csv",
    )
    ap.add_argument("--tickers", nargs="*", default=["NVDA", "AMD", "INTC"])
    ap.add_argument("--rolling", type=int, default=21, help="Rolling window (trading days)")
    ap.add_argument("-o", "--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    prep = args.prep_dir.resolve()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for t in args.tickers:
        path = prep / f"merged_{t}_market_sentiment_complete.csv"
        if not path.exists():
            path = prep / f"merged_{t}_market_sentiment.csv"
        if not path.exists():
            print(f"Skip {t}: missing {path}")
            continue

        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.endswith("_lag1")]

        plot_time_series_nets(df, t, args.rolling, args.out_dir / f"ts_{t}_nets.png")
        plot_posts_return_noise(df, t, args.out_dir / f"ts_{t}_posts_volatility.png")

        nrc = [c for c in NRC_COLS if c in df.columns]
        corr_heatmap(df, nrc, f"{t} — NRC emotion / lexicon correlations", args.out_dir / f"corr_{t}_nrc.png")

        hf_list = list(HF_COLS)
        d2 = add_hf_net_4(df)
        for c in HF_COLS:
            if c not in d2.columns:
                d2[c] = np.nan
        corr_heatmap(
            d2,
            hf_list + (["hf_net_4"] if "hf_net_4" in d2.columns else []),
            f"{t} — HuggingFace class probabilities + HF net",
            args.out_dir / f"corr_{t}_hf.png",
        )

        fb = [c for c in FB_COLS if c in df.columns]
        corr_heatmap(df, fb, f"{t} — FinBERT scores + net", args.out_dir / f"corr_{t}_fb.png")

        plot_cross_net(df, t, args.out_dir / f"corr_{t}_cross_net.png")

        print(f"Wrote figures for {t} -> {args.out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
