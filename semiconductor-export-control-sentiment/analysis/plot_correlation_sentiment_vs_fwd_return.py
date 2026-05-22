#!/usr/bin/env python3
"""
Correlations aligned with prediction: sentiment at day t vs **next trading day**
stock return (forward return), for each sentiment column and each model family.

  ret_fwd[t] = ret_1d[t+1]   (shift -1 after sorting by date)

Outputs:
  analysis_output/correlation_sentiment_vs_fwd_return.csv
  analysis_output/figures/sentiment_diagnostics/corr_fwd_return_nets_bar.png
  analysis_output/figures/sentiment_diagnostics/corr_fwd_return_heatmap_nrc.png
  analysis_output/figures/sentiment_diagnostics/corr_fwd_return_heatmap_hf.png
  analysis_output/figures/sentiment_diagnostics/corr_fwd_return_heatmap_fb.png

Usage (from analysis/):
  python plot_correlation_sentiment_vs_fwd_return.py
  python plot_correlation_sentiment_vs_fwd_return.py --tickers NVDA AMD
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DEFAULT = SCRIPT_DIR / "reddit_prepared"
OUT_CSV_DEFAULT = SCRIPT_DIR / "analysis_output" / "correlation_sentiment_vs_fwd_return.csv"
FIG_DIR_DEFAULT = SCRIPT_DIR / "analysis_output" / "figures" / "sentiment_diagnostics"

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

NET_KEYS = ("nrc_net_sentiment", "hf_net_4", "fb_net_sentiment")


def hf_net_4(df: pd.DataFrame) -> pd.Series:
    need = (
        "hf_very_positive_score",
        "hf_positive_score",
        "hf_very_negative_score",
        "hf_negative_score",
    )
    if not all(c in df.columns for c in need):
        return pd.Series(np.nan, index=df.index)
    return (
        pd.to_numeric(df[need[0]], errors="coerce")
        + pd.to_numeric(df[need[1]], errors="coerce")
        - pd.to_numeric(df[need[2]], errors="coerce")
        - pd.to_numeric(df[need[3]], errors="coerce")
    )


def load_level_panel(tkr: str, prep: Path) -> pd.DataFrame:
    p = prep / f"merged_{tkr}_market_sentiment_complete.csv"
    if not p.exists():
        p = prep / f"merged_{tkr}_market_sentiment.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df = df.loc[:, ~df.columns.astype(str).str.endswith("_lag1")].copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False)
    df = df.sort_values("date").reset_index(drop=True)
    df["ret_1d"] = pd.to_numeric(df["ret_1d"], errors="coerce")
    df["ret_fwd"] = df["ret_1d"].shift(-1)
    df["hf_net_4"] = hf_net_4(df)
    return df


def model_tag(col: str) -> str:
    if col.startswith("nrc_"):
        return "NRC"
    if col.startswith("hf_") or col == "hf_net_4":
        return "HF"
    if col.startswith("fb_"):
        return "FinBERT"
    return "other"


def corr_pair(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 10:
        return (np.nan, np.nan, n)
    a, b = x[m].astype(float), y[m].astype(float)
    p = a.corr(b, method="pearson")
    s = a.corr(b, method="spearman")
    return (float(p) if np.isfinite(p) else np.nan, float(s) if np.isfinite(s) else np.nan, n)


def try_heatmap(ax, mat: pd.DataFrame, title: str) -> None:
    try:
        import seaborn as sns

        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0.0,
            vmin=-1,
            vmax=1,
        )
    except ImportError:
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(mat.shape[1]) + 0.5, labels=mat.columns)
        ax.set_yticks(np.arange(mat.shape[0]) + 0.5, labels=mat.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                t = "" if not np.isfinite(v) else f"{v:.2f}"
                ax.text(j + 0.5, i + 0.5, t, ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.035)
    ax.set_title(title)
    ax.set_xlabel("Ticker (sentiment day t; ret is t+1)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", type=Path, default=PREP_DEFAULT)
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="correlation_sentiment_vs_fwd_return.csv path",
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="PNG output directory",
    )
    ap.add_argument("--tickers", nargs="*", default=["NVDA", "AMD", "INTC"])
    args = ap.parse_args()

    prep = args.prep_dir.resolve()
    out_csv = args.out_csv or OUT_CSV_DEFAULT
    fig_dir = args.fig_dir or FIG_DIR_DEFAULT

    sentiment_groups = [
        ("NRC", list(NRC_COLS)),
        ("HF", list(HF_COLS) + ["hf_net_4"]),
        ("FinBERT", list(FB_COLS)),
    ]

    rows = []
    pearson_blocks: dict[str, pd.DataFrame] = {g: [] for g, _ in sentiment_groups}

    for tkr in args.tickers:
        try:
            d = load_level_panel(tkr, prep)
        except FileNotFoundError as e:
            print(f"Skip {tkr}: {e}")
            continue
        y = d["ret_fwd"]
        for group_name, cols in sentiment_groups:
            block = {}
            for c in cols:
                if c not in d.columns and c != "hf_net_4":
                    continue
                if c not in d.columns:
                    continue
                x = pd.to_numeric(d[c], errors="coerce")
                pear, spear, n = corr_pair(x, y)
                rows.append(
                    {
                        "ticker": tkr,
                        "model": model_tag(c),
                        "sentiment_column": c,
                        "pearson_corr_sent_t_vs_ret_t1": pear,
                        "spearman_corr_sent_t_vs_ret_t1": spear,
                        "n_pairs": n,
                    }
                )
                block[c] = pear
            if block:
                pearson_blocks[group_name].append(pd.Series(block, name=tkr))

    if not rows:
        print("No rows written â€” check tickers / panel paths.")
        return

    out_df = pd.DataFrame(rows)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    for group_name, _ in sentiment_groups:
        series_list = pearson_blocks[group_name]
        if not series_list:
            continue
        mat = pd.concat(series_list, axis=1)
        fig, ax = plt.subplots(figsize=(max(7, 1.2 + mat.shape[1]), max(4, 0.35 * mat.shape[0])))
        try_heatmap(
            ax,
            mat,
            f"{group_name}: Pearson r vs next-day return (sentiment at t, ret_1d at t+1)",
        )
        fig.tight_layout()
        out_p = fig_dir / f"corr_fwd_return_heatmap_{group_name.lower()}.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_p}")

    nets_long = out_df[out_df["sentiment_column"].isin(NET_KEYS)].copy()
    if len(nets_long):
        order = {k: i for i, k in enumerate(NET_KEYS)}
        nets_long["_o"] = nets_long["sentiment_column"].map(order)
        nets_long = nets_long.sort_values(["ticker", "_o"])

        tickers_u = list(dict.fromkeys(nets_long["ticker"].tolist()))
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(tickers_u))
        width = 0.25
        for i, key in enumerate(NET_KEYS):
            heights = []
            for t in tickers_u:
                row = nets_long[(nets_long["ticker"] == t) & (nets_long["sentiment_column"] == key)]
                heights.append(float(row["pearson_corr_sent_t_vs_ret_t1"].iloc[0]) if len(row) else np.nan)
            ax.bar(x + (i - 1) * width, heights, width, label=key.replace("_", " "))
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(tickers_u)
        ax.set_ylabel("Pearson r")
        ax.set_xlabel("Ticker (same-day sentiment vs next-day return)")
        ax.legend(title="Sentiment (day t)", fontsize=8)
        ax.set_title("Net measures vs next-day stock return")
        fig.tight_layout()
        barp = fig_dir / "corr_fwd_return_nets_bar.png"
        fig.savefig(barp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {barp}")

    print("Done.")


if __name__ == "__main__":
    main()

