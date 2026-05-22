#!/usr/bin/env python3
"""
Map pooled daily sentiment (one row per CALENDAR date, no ticker) onto the same
trading calendar as market_daily_panel.csv, then optionally join SOXX/SPY.

Weekend/holiday calendar rows that map to the same trading session are averaged.

Inputs (defaults under reddit_prepared/):
  reddit_NEW_RUN_daily_sentiment.csv

Outputs:
  reddit_prepared/pooled_daily_on_trading_calendar.csv
  reddit_prepared/merged_pooled_SOXX.csv   (if benchmark_SOXX.csv exists)
  reddit_prepared/merged_pooled_SPY.csv    (if benchmark_SPY.csv exists)

This file is NOT per-company. For firm-level merges use sentiment_firm_*_daily.csv
or merged_*_market_sentiment.csv from prepare_reddit_dual_layers.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from merge_sentiment_to_market import load_trading_day_index, next_trading_day

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MARKET = SCRIPT_DIR / "market_daily_panel.csv"
PREP_DIR = SCRIPT_DIR / "reddit_prepared"
DEFAULT_INPUT = PREP_DIR / "reddit_NEW_RUN_daily_sentiment.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Trading-calendar prep for pooled daily sentiment CSV.")
    p.add_argument(
        "--prep-dir",
        type=Path,
        default=None,
        help="Folder for pooled CSV + benchmark joins (default: parent of --input, or reddit_prepared)",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Daily sentiment CSV (default: <prep-dir>/reddit_NEW_RUN_daily_sentiment.csv)",
    )
    p.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Write pooled_daily_on_trading_calendar.csv here (default: same as --prep-dir)",
    )
    args = p.parse_args()

    prep_dir = args.prep_dir
    if args.input is None:
        if prep_dir is not None:
            broad = prep_dir / "sentiment_broad_daily.csv"
            legacy = prep_dir / "reddit_NEW_RUN_daily_sentiment.csv"
            args.input = broad if broad.exists() else legacy
        else:
            args.input = PREP_DIR / "reddit_NEW_RUN_daily_sentiment.csv"
    if prep_dir is None:
        prep_dir = args.input.parent
    out_dir = args.out_dir if args.out_dir is not None else prep_dir
    args.out_dir = out_dir

    if not args.input.exists():
        raise SystemExit(f"Missing input: {args.input}")

    market = pd.read_csv(args.market)
    market["date"] = pd.to_datetime(market["date"], format="mixed", utc=False).dt.normalize()
    sorted_days, _ = load_trading_day_index(market)

    df = pd.read_csv(args.input)
    if "date" not in df.columns:
        raise SystemExit(f"Expected column 'date'; got {list(df.columns)}")

    df = df.copy()
    df["cal_date"] = pd.to_datetime(df["date"], format="mixed", utc=False).dt.normalize()
    df["date"] = df["cal_date"].map(lambda d: next_trading_day(d, sorted_days))
    df = df.dropna(subset=["date"])
    df = df.drop(columns=["cal_date"])

    sentiment_cols = [c for c in df.columns if c != "date"]
    # Multiple calendar days (e.g. Sat+Sun) can share one trading date — average.
    g = df.groupby("date", sort=True)
    agg = g[sentiment_cols].mean()
    n_cal = g.size().rename("n_calendar_source_days")
    out = agg.join(n_cal, how="left").reset_index()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "pooled_daily_on_trading_calendar.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} trading days)")

    def add_lags(m: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        m = m.sort_values("date")
        for c in cols:
            if c in m.columns:
                m[f"{c}_lag1"] = m[c].shift(1)
        return m

    lag_cols = [c for c in sentiment_cols if c in out.columns] + ["n_calendar_source_days"]

    for bench in ("SOXX", "SPY", "VIX"):
        bp = prep_dir / f"benchmark_{bench}.csv"
        if not bp.exists():
            print(
                f"Skip merged_pooled_{bench}: no {bp.name} "
                "(run prepare_*_dual_layers.py with benchmarks / yfinance)"
            )
            continue
        bench_df = pd.read_csv(bp)
        bench_df["date"] = pd.to_datetime(bench_df["date"], format="mixed", utc=False).dt.normalize()
        merged = bench_df.merge(out, on="date", how="inner")
        merged = add_lags(merged, lag_cols)
        mp = args.out_dir / f"merged_pooled_{bench}.csv"
        merged.to_csv(mp, index=False)
        print(f"Wrote {mp} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
