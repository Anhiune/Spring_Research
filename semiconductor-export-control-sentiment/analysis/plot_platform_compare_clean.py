#!/usr/bin/env python3
"""
Cleaner platform comparisons than messy line charts.

For each ticker and each sentiment model net, we plot:
  1) Binned (quantile) mean: E[ret_1d(t+1) | sentiment bin at t]
  2) Light scatter + linear fit: ret_1d(t+1) vs sentiment(t)
  3) Sentiment distribution (hist)

Outputs:
  analysis/analysis_output_platform_compare/clean/*

Inputs:
  Uses the same prepared firm panels:
    - reddit:  analysis/reddit_prepared/merged_{T}_market_sentiment_complete.csv
    - bluesky: analysis/bluesky_prepared/merged_{T}_market_sentiment_complete.csv
  If *_complete.csv missing, falls back to merged_{T}_market_sentiment.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def load_panel(prep_dir: Path, ticker: str) -> pd.DataFrame:
    p = prep_dir / f"merged_{ticker}_market_sentiment_complete.csv"
    if not p.exists():
        p = prep_dir / f"merged_{ticker}_market_sentiment.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    df["ret_1d"] = pd.to_numeric(df.get("ret_1d"), errors="coerce")
    df["ret_fwd_1d"] = df["ret_1d"].shift(-1)  # sentiment(t) -> return(t+1)
    return df


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


def get_nets(df: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    out["NRC net"] = pd.to_numeric(df.get("nrc_net_sentiment"), errors="coerce")
    out["HF net"] = hf_net_4(df)
    out["FinBERT net"] = pd.to_numeric(df.get("fb_net_sentiment"), errors="coerce")
    return out


def _finite_xy(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    xx = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yy = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(xx) & np.isfinite(yy)
    return xx[m], yy[m]


def plot_binned_means(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    n_bins: int,
    title: str,
) -> None:
    xx, yy = _finite_xy(x, y)
    if len(xx) < 50:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        ax.set_title(title)
        return

    # quantile bins (robust to outliers)
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(xx, q))
    if len(edges) <= 2:
        ax.text(0.5, 0.5, "Low variation in sentiment", ha="center", va="center")
        ax.set_title(title)
        return

    bin_id = np.digitize(xx, edges[1:-1], right=True)
    centers = []
    means = []
    ses = []
    ns = []
    for b in range(bin_id.min(), bin_id.max() + 1):
        m = bin_id == b
        if m.sum() < 10:
            continue
        xb = float(np.nanmean(xx[m]))
        yb = yy[m]
        mu = float(np.nanmean(yb))
        se = float(np.nanstd(yb, ddof=1) / np.sqrt(max(1, m.sum())))
        centers.append(xb)
        means.append(mu)
        ses.append(se)
        ns.append(int(m.sum()))

    if not centers:
        ax.text(0.5, 0.5, "Bins too sparse", ha="center", va="center")
        ax.set_title(title)
        return

    centers = np.array(centers)
    means = np.array(means)
    ses = np.array(ses)

    ax.plot(centers, means, marker="o", lw=1.8, color="black")
    ax.fill_between(centers, means - 1.96 * ses, means + 1.96 * ses, color="0.8", alpha=0.6)
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_title(title)
    ax.set_xlabel("Sentiment (bin mean)")
    ax.set_ylabel("Next-day return")
    ax.grid(True, alpha=0.25)


def plot_scatter_fit(ax: plt.Axes, x: pd.Series, y: pd.Series, title: str) -> None:
    xx, yy = _finite_xy(x, y)
    if len(xx) < 50:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        ax.set_title(title)
        return
    ax.scatter(xx, yy, s=8, alpha=0.15, color="steelblue", edgecolors="none")
    # linear fit (simple + readable)
    slope, intercept = np.polyfit(xx, yy, 1)
    xs = np.linspace(np.nanpercentile(xx, 2), np.nanpercentile(xx, 98), 100)
    ax.plot(xs, intercept + slope * xs, color="crimson", lw=2.0, label=f"fit slope={slope:.4g}")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_title(title)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Next-day return")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)


def plot_hist(ax: plt.Axes, x: pd.Series, title: str) -> None:
    xx = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    xx = xx[np.isfinite(xx)]
    if len(xx) < 50:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        ax.set_title(title)
        return
    ax.hist(xx, bins=40, color="0.35", alpha=0.8)
    ax.axvline(np.nanmean(xx), color="crimson", lw=1.6, label="mean")
    ax.set_title(title)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)


def make_triptych(
    out_path: Path,
    ticker: str,
    model_label: str,
    reddit_df: pd.DataFrame,
    bluesky_df: pd.DataFrame,
    sentiment_r: pd.Series,
    sentiment_b: pd.Series,
    n_bins: int,
) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 0.9])

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    ax31 = fig.add_subplot(gs[2, 0])
    ax32 = fig.add_subplot(gs[2, 1])

    plot_binned_means(
        ax11,
        sentiment_r,
        reddit_df["ret_fwd_1d"],
        n_bins=n_bins,
        title=f"Reddit â€” binned mean ret(t+1) vs {model_label}",
    )
    plot_binned_means(
        ax12,
        sentiment_b,
        bluesky_df["ret_fwd_1d"],
        n_bins=n_bins,
        title=f"Bluesky â€” binned mean ret(t+1) vs {model_label}",
    )

    plot_scatter_fit(
        ax21,
        sentiment_r,
        reddit_df["ret_fwd_1d"],
        title="Reddit â€” scatter + linear fit",
    )
    plot_scatter_fit(
        ax22,
        sentiment_b,
        bluesky_df["ret_fwd_1d"],
        title="Bluesky â€” scatter + linear fit",
    )

    plot_hist(ax31, sentiment_r, title="Reddit â€” sentiment distribution")
    plot_hist(ax32, sentiment_b, title="Bluesky â€” sentiment distribution")

    fig.suptitle(f"{ticker}: platform comparison for {model_label}", y=1.01, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cleaner Reddit vs Bluesky comparison plots.")
    ap.add_argument("--reddit-prep", type=Path, default=SCRIPT_DIR / "reddit_prepared")
    ap.add_argument("--bluesky-prep", type=Path, default=SCRIPT_DIR / "bluesky_prepared")
    ap.add_argument("--tickers", nargs="*", default=["NVDA", "AMD", "INTC"])
    ap.add_argument("--bins", type=int, default=10, help="Quantile bins for binned-mean plot")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "analysis_output_platform_compare" / "clean",
    )
    args = ap.parse_args()

    rprep = args.reddit_prep.resolve()
    bprep = args.bluesky_prep.resolve()
    out = args.out_dir.resolve()

    for t in args.tickers:
        r = load_panel(rprep, t)
        b = load_panel(bprep, t)

        rn = get_nets(r)
        bn = get_nets(b)
        for model in ["NRC net", "HF net", "FinBERT net"]:
            out_path = out / f"{t}_platform_compare_{model.replace(' ', '_')}.png"
            make_triptych(
                out_path=out_path,
                ticker=t,
                model_label=model,
                reddit_df=r,
                bluesky_df=b,
                sentiment_r=rn[model],
                sentiment_b=bn[model],
                n_bins=int(args.bins),
            )

    print(f"Done. Wrote plots to: {out}")


if __name__ == "__main__":
    main()


