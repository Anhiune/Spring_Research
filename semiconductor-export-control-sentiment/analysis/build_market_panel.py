#!/usr/bin/env python3
"""
Stack per-ticker Yahoo CSVs into one long table for merging with sentiment.

Input : this folder, files named <TICKER>.csv with columns Date, Open, ...
Output: market_daily_panel.csv (long format: one row per ticker per trading day)

Usage:
  python build_market_panel.py
  python build_market_panel.py --start 2023-01-01 --end 2026-02-29
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_START = "2023-01-01"
DEFAULT_END = None  # no upper trim


def read_ticker_csv(path: Path) -> pd.DataFrame:
    ticker = path.stem.upper()
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path.name}: expected a 'Date' column, got {list(df.columns)}")
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    })
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False)
    df["ticker"] = ticker
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Merge ticker CSVs into one market panel.")
    p.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"First calendar date to keep (default: {DEFAULT_START}). Use empty string for no start trim.",
    )
    p.add_argument(
        "--end",
        default="",
        help="Last calendar date to keep (inclusive). Omit for no end trim.",
    )
    p.add_argument(
        "-o", "--output",
        default="market_daily_panel.csv",
        help="Output filename inside this folder (default: market_daily_panel.csv)",
    )
    args = p.parse_args()

    csv_paths = sorted(SCRIPT_DIR.glob("*.csv"))
    csv_paths = [x for x in csv_paths if x.name.lower() != args.output.lower()]

    if not csv_paths:
        raise SystemExit(f"No CSV files found in {SCRIPT_DIR}")

    frames = []
    for path in csv_paths:
        frames.append(read_ticker_csv(path))

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    if args.start:
        out = out[out["date"] >= pd.Timestamp(args.start)]
    if args.end:
        out = out[out["date"] <= pd.Timestamp(args.end)]

    # Simple return on split-adjusted price (within ticker)
    out["ret_1d"] = out.groupby("ticker", sort=False)["adj_close"].pct_change()

    numeric = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits", "ret_1d"]
    for c in numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out_path = SCRIPT_DIR / args.output
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} rows x {len(out.columns)} cols to {out_path}")
    print(f"Tickers: {sorted(out['ticker'].unique().tolist())}")
    print(f"Date range: {out['date'].min().date()} to {out['date'].max().date()}")


if __name__ == "__main__":
    main()
