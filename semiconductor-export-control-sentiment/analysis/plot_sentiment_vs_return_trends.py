#!/usr/bin/env python3
"""
Line charts comparing stock dynamics with sentiment from NRC, HuggingFace, and FinBERT.

Writes PNGs under analysis_output/figures/sentiment_return_trends/:
  compare_{TICKER}_stacked_4panel.png   â€” cum return + each model net w/ rolling mean
  compare_{TICKER}_overlay_zscore.png   â€” all three nets + return on one chart (busy)
  compare_{TICKER}_overlay_zscore_split.png â€” same data, **one model per row** vs return (easier read)
  compare_{TICKER}_emotional_change.png â€” short-horizon deltas (5d) in nets vs 5d log return
  compare_{TICKER}_emotional_change_{d}d_split.png â€” **one Î”-sentiment row per model** + shared return bars at bottom
  compare_ALL_cumret_vs_models.png      â€” 3Ã—1 small multiples (NVDA/AMD/INTC cum ret + 3 nets roll)

Usage (from analysis/):
  python plot_sentiment_vs_return_trends.py
  python plot_sentiment_vs_return_trends.py --tickers NVDA INTC --roll 21
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DEFAULT = SCRIPT_DIR / "reddit_prepared"
OUT_DIR_DEFAULT = SCRIPT_DIR / "analysis_output" / "figures" / "sentiment_return_trends"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
    }
)


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


def load_panel(tkr: str, prep: Path) -> pd.DataFrame:
    p = prep / f"merged_{tkr}_market_sentiment_complete.csv"
    if not p.exists():
        p = prep / f"merged_{tkr}_market_sentiment.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df = df.loc[:, ~df.columns.astype(str).str.endswith("_lag1")].copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False)
    df = df.sort_values("date")
    df["ret_1d"] = pd.to_numeric(df["ret_1d"], errors="coerce")
    df["hf_net_4"] = hf_net_4(df)
    if "nrc_net_sentiment" in df.columns:
        df["nrc_net_sentiment"] = pd.to_numeric(df["nrc_net_sentiment"], errors="coerce")
    else:
        df["nrc_net_sentiment"] = np.nan
    if "fb_net_sentiment" in df.columns:
        df["fb_net_sentiment"] = pd.to_numeric(df["fb_net_sentiment"], errors="coerce")
    else:
        df["fb_net_sentiment"] = np.nan
    r = df["ret_1d"].fillna(0.0)
    df["cum_return"] = (1.0 + r).cumprod()
    df["cum_return_pct"] = (df["cum_return"] / df["cum_return"].replace(0, np.nan).iloc[0] - 1.0) * 100.0
    return df


def roll(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 3)).mean()


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return x * 0.0
    return (x - x.mean()) / sd


def chart_stacked_4panel(d: pd.DataFrame, tkr: str, roll_w: int, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    x = d["date"]

    ax = axes[0]
    ax.plot(x, d["cum_return_pct"], color="black", lw=1.4, label="Cum. return vs start (%)")
    ax.set_ylabel("Stock (%)")
    ax.legend(loc="upper left")
    ax.set_title(f"{tkr} â€” cumulative return and sentiment nets (raw scale)")

    specs = [
        ("nrc_net_sentiment", "NRC net", "#1f77b4"),
        ("hf_net_4", "HF net (4-class)", "#ff7f0e"),
        ("fb_net_sentiment", "FinBERT net", "#2ca02c"),
    ]
    for ax, (col, lab, c) in zip(axes[1:], specs):
        if col not in d.columns:
            ax.axis("off")
            continue
        s = pd.to_numeric(d[col], errors="coerce")
        ax.plot(x, s, color=c, lw=0.6, alpha=0.4, label="daily")
        ax.plot(x, roll(s, roll_w), color=c, lw=2.0, label=f"{roll_w}d mean")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_ylabel(lab)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_overlay_zscore(d: pd.DataFrame, tkr: str, roll_w: int, path: Path) -> None:
    x = d["date"]
    fig, ax1 = plt.subplots(figsize=(11, 4.5))

    for col, lab, c in [
        ("nrc_net_sentiment", "NRC (z, roll)", "#1f77b4"),
        ("hf_net_4", "HF (z, roll)", "#ff7f0e"),
        ("fb_net_sentiment", "FB (z, roll)", "#2ca02c"),
    ]:
        if col not in d.columns:
            continue
        s = zscore(roll(pd.to_numeric(d[col], errors="coerce"), roll_w))
        ax1.plot(x, s, lw=1.5, color=c, label=lab, alpha=0.9)

    ax1.axhline(0, color="0.5", lw=0.6)
    ax1.set_ylabel("Z-scored sentiment (rolling)")
    ax1.set_title(
        f"{tkr} â€” emotional trend lines ({roll_w}d rolling, z-scored) vs return trend"
    )

    ax2 = ax1.twinx()
    r = zscore(roll(d["ret_1d"], roll_w))
    ax2.plot(x, r, color="crimson", lw=1.2, alpha=0.75, linestyle="--", label="|trend| ret z")
    ax2.set_ylabel("Z-scored rolling return (dashed)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_emotional_change(d: pd.DataFrame, tkr: str, delta_w: int, path: Path) -> None:
    """Short-horizon change: Î” sentiment over delta_w days vs sum of log-returns over same window."""
    x = d["date"]
    fig, ax1 = plt.subplots(figsize=(11, 4.8))

    for col, lab, c in [
        ("nrc_net_sentiment", f"Î” NRC ({delta_w}d)", "#1f77b4"),
        ("hf_net_4", f"Î” HF ({delta_w}d)", "#ff7f0e"),
        ("fb_net_sentiment", f"Î” FinBERT ({delta_w}d)", "#2ca02c"),
    ]:
        if col not in d.columns:
            continue
        base = pd.to_numeric(d[col], errors="coerce")
        delta = base - base.shift(delta_w)
        ax1.plot(x, delta, lw=1.1, color=c, label=lab, alpha=0.85)

    ax1.axhline(0, color="0.5", lw=0.6)
    ax1.set_ylabel(f"Change in net sentiment ({delta_w} sessions)")
    ax1.set_title(f"{tkr} â€” emotional *changes* vs stock move over {delta_w} days")

    ax2 = ax1.twinx()
    lr = np.log1p(d["ret_1d"].clip(-0.999, 10))
    move = lr.rolling(delta_w, min_periods=1).sum()
    ax2.bar(x, move, width=1.2, color="gray", alpha=0.25, label=f"Sum log(1+r), {delta_w}d")
    ax2.set_ylabel("Cumulative log return (bars)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_overlay_zscore_split(d: pd.DataFrame, tkr: str, roll_w: int, path: Path) -> None:
    """One panel per sentiment model; same crimson dashed return on each (easier to read)."""
    x = d["date"]
    r = zscore(roll(d["ret_1d"], roll_w))
    specs = [
        ("nrc_net_sentiment", "NRC net", "#1f77b4"),
        ("hf_net_4", "HF net (4-class)", "#ff7f0e"),
        ("fb_net_sentiment", "FinBERT net", "#2ca02c"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, (col, lab, c) in zip(axes, specs):
        if col not in d.columns:
            ax.axis("off")
            continue
        s = zscore(roll(pd.to_numeric(d[col], errors="coerce"), roll_w))
        ax.plot(x, s, color=c, lw=2.0, label=f"{lab} â€” z, {roll_w}d roll")
        ax.axhline(0, color="0.65", lw=0.6)
        ax.set_ylabel("Sentiment\n(z-score)", color=c, fontsize=9)
        ax.tick_params(axis="y", labelcolor=c)
        ax2 = ax.twinx()
        ax2.plot(
            x,
            r,
            color="crimson",
            lw=1.2,
            ls="--",
            alpha=0.9,
            label="Return â€” z, same window",
        )
        ax2.set_ylabel("Return\n(z-score)", color="crimson", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="crimson")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, ncol=2)
        ax.set_title(f"{lab} only (compare to same return line in all 3 panels)")
    axes[0].text(
        0.01,
        1.08,
        "Same red dashed line in each row = rolling return z-score; blue/orange/green = that model only.",
        transform=axes[0].transAxes,
        fontsize=8,
        color="0.25",
        va="bottom",
    )
    axes[-1].set_xlabel("Date")
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(
        f"{tkr} â€” split view: one sentiment model per panel vs return trend",
        y=1.02,
        fontsize=12,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_emotional_change_split(d: pd.DataFrame, tkr: str, delta_w: int, path: Path) -> None:
    """Three panels for Î” each net; bottom panel = 5d sum log return only (not repeated)."""
    x = d["date"]
    lr = np.log1p(d["ret_1d"].clip(-0.999, 10))
    move = lr.rolling(delta_w, min_periods=1).sum()
    specs = [
        ("nrc_net_sentiment", "NRC net", "#1f77b4"),
        ("hf_net_4", "HF net", "#ff7f0e"),
        ("fb_net_sentiment", "FinBERT net", "#2ca02c"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.75]})
    for ax, (col, lab, c) in zip(axes[:3], specs):
        if col not in d.columns:
            ax.axis("off")
            continue
        base = pd.to_numeric(d[col], errors="coerce")
        delta = base - base.shift(delta_w)
        ax.plot(x, delta, lw=1.2, color=c, label=f"Î” {lab} ({delta_w} sessions)")
        ax.axhline(0, color="0.55", lw=0.6)
        ax.set_ylabel(f"Î” {lab.split()[0]}\n(net)", fontsize=9, color=c)
        ax.tick_params(axis="y", labelcolor=c)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{lab}: change in net vs previous {delta_w} sessions")
    axb = axes[3]
    axb.bar(x, move, width=1.3, color="dimgray", alpha=0.35, label=f"Sum of log(1+r) over {delta_w} days")
    axb.axhline(0, color="0.4", lw=0.6)
    axb.set_ylabel(f"Stock move\n({delta_w}d Î£ log ret)")
    axb.legend(loc="upper left", fontsize=8)
    axb.set_title("Same stock window for all rows above â€” compare spikes visually")
    axes[-1].set_xlabel("Date")
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(
        f"{tkr} â€” emotional changes (split): one model per row + shared 5d return strip",
        y=1.01,
        fontsize=12,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_all_tickers_cumret_vs_models(tickers: list[str], roll_w: int, path: Path, prep: Path) -> None:
    fig, axes = plt.subplots(len(tickers), 1, figsize=(11, 3.8 * len(tickers)), sharex=False)
    if len(tickers) == 1:
        axes = [axes]
    for ax, tkr in zip(axes, tickers):
        d = load_panel(tkr, prep)
        x = d["date"]
        ax.plot(x, d["cum_return_pct"], color="black", lw=1.5, label="Cum ret %")
        ax2 = ax.twinx()
        for col, lab, c in [
            ("nrc_net_sentiment", "NRC roll", "#1f77b4"),
            ("hf_net_4", "HF roll", "#ff7f0e"),
            ("fb_net_sentiment", "FB roll", "#2ca02c"),
        ]:
            if col not in d.columns:
                continue
            s = roll(pd.to_numeric(d[col], errors="coerce"), roll_w)
            ax2.plot(x, s, lw=1.2, color=c, alpha=0.85, label=lab)
        ax.set_ylabel("Cum return %", color="black")
        ax2.set_ylabel(f"{roll_w}d mean sentiment", color="0.3")
        ax.set_title(tkr)
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Stock path vs rolling sentiment by model", y=1.01, fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", type=Path, default=PREP_DEFAULT)
    ap.add_argument("--tickers", nargs="*", default=["NVDA", "AMD", "INTC"])
    ap.add_argument("--roll", type=int, default=21, help="Rolling mean window (trading days)")
    ap.add_argument("--delta", type=int, default=5, help="Horizon for change charts")
    ap.add_argument("-o", "--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = ap.parse_args()
    prep = args.prep_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for t in args.tickers:
        try:
            d = load_panel(t, prep)
        except FileNotFoundError as e:
            print(f"Skip {t}: {e}")
            continue
        chart_stacked_4panel(
            d, t, args.roll, args.out_dir / f"compare_{t}_stacked_4panel.png"
        )
        chart_overlay_zscore(
            d, t, args.roll, args.out_dir / f"compare_{t}_overlay_zscore.png"
        )
        chart_emotional_change(
            d, t, args.delta, args.out_dir / f"compare_{t}_emotional_change_{args.delta}d.png"
        )
        chart_overlay_zscore_split(
            d, t, args.roll, args.out_dir / f"compare_{t}_overlay_zscore_split.png"
        )
        chart_emotional_change_split(
            d, t, args.delta, args.out_dir / f"compare_{t}_emotional_change_{args.delta}d_split.png"
        )
        print(f"Wrote 5 charts for {t}")

    ok = []
    for t in args.tickers:
        if (prep / f"merged_{t}_market_sentiment_complete.csv").exists() or (
            prep / f"merged_{t}_market_sentiment.csv"
        ).exists():
            ok.append(t)
    if ok:
        chart_all_tickers_cumret_vs_models(
            ok,
            args.roll,
            args.out_dir / "compare_ALL_cumret_vs_models.png",
            prep,
        )
    print(f"Done -> {args.out_dir}")


if __name__ == "__main__":
    main()

