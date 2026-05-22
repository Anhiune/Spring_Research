#!/usr/bin/env python3
"""
Build one daily series from market_daily_panel.csv for "overall universe" trends.

For each trading day, aggregates across all tickers in the panel:
  - Equal-weight mean/median of ret_1d (simple cross-sectional average return)
  - Cross-sectional std (dispersion)
  - Count of tickers with non-null ret_1d that day
  - Sum of volume
  - Cumulative index from compounding ew_ret_mean (starts 1.0)

Use with pooled / broad sentiment (e.g. merged_pooled_SPY.csv) for aligned
"overall mood vs overall chip-universe move" comparisons.

Usage:
  python build_universe_trend_daily.py
  python build_universe_trend_daily.py --market path/to/market_daily_panel.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MARKET = SCRIPT_DIR / "market_daily_panel.csv"
DEFAULT_OUT = SCRIPT_DIR / "universe_12tickers_trend_daily.csv"
DEFAULT_POOLED = SCRIPT_DIR / "reddit_prepared" / "pooled_daily_on_trading_calendar.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Daily equal-weight universe trend from market panel.")
    p.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--merge-pooled-sentiment",
        type=Path,
        nargs="?",
        const=DEFAULT_POOLED,
        default=None,
        help="If set, write merged universe x pooled sentiment CSV (path to pooled_daily_on_trading_calendar.csv)",
    )
    p.add_argument(
        "--merged-universe-out",
        type=Path,
        default=None,
        help="Output path for merged_universe_pooled_sentiment.csv (default: next to --output)",
    )
    args = p.parse_args()

    df = pd.read_csv(args.market)
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False).dt.normalize()

    g = df.groupby("date", sort=True)
    vol = g["volume"].sum().rename("total_volume")
    ret = g["ret_1d"].agg(
        ew_ret_mean="mean",
        ew_ret_median="median",
        ret_xs_std="std",
        n_with_ret=lambda s: s.notna().sum(),
    )
    out = ret.join(vol, how="left").reset_index()

    # Cumulative equal-weight wealth index (compound daily mean returns; NaN days unchanged)
    r = out["ew_ret_mean"].fillna(0.0)
    out["ew_cum_index"] = (1.0 + r).cumprod()
    out["ew_ret_mean_lag1"] = out["ew_ret_mean"].shift(1)

    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} days)")
    print(f"Tickers in panel: {sorted(df['ticker'].unique().tolist())}")

    if args.merge_pooled_sentiment is not None and args.merge_pooled_sentiment.exists():
        sent = pd.read_csv(args.merge_pooled_sentiment)
        sent["date"] = pd.to_datetime(sent["date"], format="mixed", utc=False).dt.normalize()
        merged = out.merge(sent, on="date", how="inner", suffixes=("", "_sent"))
        sent_cols = [c for c in sent.columns if c != "date"]
        merged = merged.sort_values("date")
        for c in sent_cols:
            if c in merged.columns:
                merged[f"{c}_lag1"] = merged[c].shift(1)
        mpath = args.merged_universe_out if args.merged_universe_out is not None else (
            args.output.parent / "merged_universe_pooled_sentiment.csv"
        )
        merged.to_csv(mpath, index=False)
        print(f"Wrote {mpath} ({len(merged)} rows, inner join universe x pooled sentiment)")


if __name__ == "__main__":
    main()
