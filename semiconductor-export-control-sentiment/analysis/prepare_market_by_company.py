#!/usr/bin/env python3
"""
Split market_daily_panel.csv into one merge-ready file per ticker (date + OHLCV + returns).

Use these to left-merge any daily sentiment (or other) series on `date`.

Output directory (default): analysis/company_market_panels/
  MARKET_AMAT_daily.csv
  MARKET_AMD_daily.csv
  ...

Usage:
  python prepare_market_by_company.py
  python prepare_market_by_company.py --market path/to/market_daily_panel.csv -o out/dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MARKET = SCRIPT_DIR / "market_daily_panel.csv"
DEFAULT_OUT = SCRIPT_DIR / "company_market_panels"


def main() -> None:
    p = argparse.ArgumentParser(description="Split market panel into one CSV per ticker.")
    p.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    p.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.market)
    m["date"] = pd.to_datetime(m["date"], format="mixed", utc=False).dt.normalize()

    for ticker, g in m.groupby("ticker", sort=True):
        g = g.sort_values("date").drop(columns=["ticker"]).reset_index(drop=True)
        g["ret_1d_lag1"] = g["ret_1d"].shift(1)
        path = args.out_dir / f"MARKET_{ticker}_daily.csv"
        g.to_csv(path, index=False)
        print(f"Wrote {path} ({len(g)} rows)")

    print(f"Done. {m['ticker'].nunique()} tickers -> {args.out_dir}")


if __name__ == "__main__":
    main()

