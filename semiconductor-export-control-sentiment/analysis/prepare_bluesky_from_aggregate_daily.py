#!/usr/bin/env python3
"""
When Bluesky is only available as a pooled DAILY aggregate (e.g. bluesky_all_cleaned_daily_sentiment.csv
with *_mean columns), map calendar dates â†’ trading days, pull benchmarks, build pooled merges,
and attach the **same** aggregate sentiment series to NVDA / AMD / INTC market panels.

This is not firm-specific sampling (that needs row-level scored posts + keywords). Here all three
names share one chip-related Bluesky mood index; returns still differ by ticker.

Usage (from analysis/):
  python prepare_bluesky_from_aggregate_daily.py
  python prepare_bluesky_from_aggregate_daily.py --input ..\\\\bluesky_data\\\\bluesky_all_cleaned_daily_sentiment.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from merge_sentiment_to_market import add_lagged_sentiment, load_trading_day_index, next_trading_day

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MARKET = SCRIPT_DIR / "market_daily_panel.csv"
DEFAULT_INPUT = SCRIPT_DIR.parent / "bluesky_data" / "bluesky_all_cleaned_daily_sentiment.csv"
PREP_DEFAULT = SCRIPT_DIR / "bluesky_prepared"
TICKERS = ("NVDA", "AMD", "INTC")


def benchmark_safe_stem(symbol: str) -> str:
    return symbol.strip().upper().lstrip("^")


def download_benchmark(ticker: str, start: pd.Timestamp, end: pd.Timestamp, out_path: Path) -> bool:
    try:
        import yfinance as yf
    except ImportError:
        print(f"Skip {ticker}: pip install yfinance")
        return False
    d = yf.download(ticker, start=start, end=end + pd.Timedelta(days=1), progress=False, auto_adjust=True)
    if d.empty:
        print(f"Empty download for {ticker}")
        return False
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)
    d = d.rename_axis("date").reset_index()
    d["date"] = pd.to_datetime(d["date"], format="mixed", utc=False).dt.normalize()
    close_col = "Close" if "Close" in d.columns else "Adj Close"
    d = d.rename(columns={close_col: "adj_close"})
    vol_col = "Volume" if "Volume" in d.columns else None
    d["ret_1d"] = d["adj_close"].pct_change()
    d["benchmark"] = ticker
    out_cols = ["date", "adj_close", "ret_1d", "benchmark"]
    if vol_col:
        d = d.rename(columns={vol_col: "volume"})
        out_cols.insert(2, "volume")
    d[out_cols].to_csv(out_path, index=False)
    print(f"Wrote benchmark {ticker} -> {out_path.name}")
    return True


def mean_to_firm_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """nrc_net_sentiment_mean -> nrc_net_sentiment for firm panels."""
    out = df.copy()
    rename = {}
    for c in out.columns:
        if c.endswith("_mean"):
            rename[c] = c[: -len("_mean")]
    return out.rename(columns=rename, errors="ignore")


def main() -> None:
    p = argparse.ArgumentParser(description="Bluesky pooled daily aggregate â†’ prep folder layout.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    p.add_argument("-o", "--prep-dir", type=Path, default=PREP_DEFAULT)
    p.add_argument("--benchmarks", default="SOXX,SPY,^VIX")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing {args.input}")

    args.prep_dir.mkdir(parents=True, exist_ok=True)

    market = pd.read_csv(args.market)
    market["date"] = pd.to_datetime(market["date"], format="mixed", utc=False).dt.normalize()
    sorted_days, _ = load_trading_day_index(market)

    df = pd.read_csv(args.input)
    if "date" not in df.columns:
        raise SystemExit(f"Expected column date; got {list(df.columns)}")

    df = df.copy()
    df["cal_date"] = pd.to_datetime(df["date"], format="mixed", utc=False).dt.normalize()
    df["date"] = df["cal_date"].map(lambda d: next_trading_day(d, sorted_days))
    df = df.dropna(subset=["date"])
    df = df.drop(columns=["cal_date"])

    sentiment_cols = [c for c in df.columns if c != "date"]
    g = df.groupby("date", sort=True)
    agg = g[sentiment_cols].mean()
    n_cal = g.size().rename("n_calendar_source_days")
    pooled = agg.join(n_cal, how="left").reset_index()

    broad_path = args.prep_dir / "sentiment_broad_daily.csv"
    pooled.to_csv(broad_path, index=False)
    print(f"Wrote {broad_path} ({len(pooled)} rows)")

    pooled_path = args.prep_dir / "pooled_daily_on_trading_calendar.csv"
    pooled.to_csv(pooled_path, index=False)
    print(f"Wrote {pooled_path} ({len(pooled)} rows)")

    dmin, dmax = pooled["date"].min(), pooled["date"].max()
    if args.benchmarks.strip():
        for raw in (s.strip() for s in args.benchmarks.split(",") if s.strip()):
            sym = raw.upper() if not raw.startswith("^") else f"^{raw[1:].upper()}"
            stem = benchmark_safe_stem(sym)
            bp = args.prep_dir / f"benchmark_{stem}.csv"
            if not bp.exists():
                download_benchmark(sym, dmin, dmax, bp)

    def add_lags(m: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        m = m.sort_values("date")
        for c in cols:
            if c in m.columns:
                m[f"{c}_lag1"] = m[c].shift(1)
        return m

    lag_cols_pooled = [c for c in sentiment_cols if c in pooled.columns] + ["n_calendar_source_days"]
    for bench in ("SOXX", "SPY", "VIX"):
        bpath = args.prep_dir / f"benchmark_{bench}.csv"
        if not bpath.exists():
            print(f"Skip merged_pooled_{bench}: no {bpath.name}")
            continue
        bench_df = pd.read_csv(bpath)
        bench_df["date"] = pd.to_datetime(bench_df["date"], format="mixed", utc=False).dt.normalize()
        merged = bench_df.merge(pooled, on="date", how="inner")
        merged = add_lags(merged, lag_cols_pooled)
        mp = args.prep_dir / f"merged_pooled_{bench}.csv"
        merged.to_csv(mp, index=False)
        print(f"Wrote {mp} ({len(merged)} rows)")

    firm_sent = mean_to_firm_column_names(pooled)
    firm_sent["n_posts"] = np.nan

    _skip_lag = frozenset({"n_posts", "n_calendar_source_days"})
    lag_cols_firm = [c for c in firm_sent.columns if c != "date" and c not in _skip_lag]

    for tkr in TICKERS:
        m = market[market["ticker"] == tkr].copy()
        out = m.merge(firm_sent, on="date", how="left")
        out = add_lagged_sentiment(out, lag_cols_firm)
        if "nrc_net_sentiment" in out.columns and "nrc_net_sentiment_lag1" in out.columns:
            out["nrc_net_sentiment_ew"] = out["nrc_net_sentiment"]
            out["nrc_net_sentiment_ew_lag1"] = out["nrc_net_sentiment_lag1"]
        fp = args.prep_dir / f"merged_{tkr}_market_sentiment.csv"
        out.to_csv(fp, index=False)
        sub_cols = ["date"] + [c for c in lag_cols_firm if c in firm_sent.columns]
        if "n_posts" in firm_sent.columns:
            sub_cols.append("n_posts")
        sub = firm_sent[sub_cols].copy()
        dp = args.prep_dir / f"sentiment_firm_{tkr}_daily.csv"
        sub.to_csv(dp, index=False)
        print(f"Wrote {fp.name} ({len(out)} rows); {dp.name}")

    print("Done (aggregate-daily path: shared sentiment across NVDA/AMD/INTC).")


if __name__ == "__main__":
    main()

