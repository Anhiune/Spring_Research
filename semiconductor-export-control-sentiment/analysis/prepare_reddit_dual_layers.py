#!/usr/bin/env python3
"""
Split Reddit sentiment into (1) firm-specific subs vs (2) broad subs, write daily
CSVs, and build per-company merged panels with market_daily_panel.csv.

Firm layer: subreddits in --firm-subreddits (default nvidia, amd, intel) ->
  sentiment_firm_NVDA_daily.csv, ... and merged_NVDA_market_sentiment.csv

Broad layer: all other rows -> sentiment_broad_daily.csv (merge later with SOXX/SPY)

Usage:
  python prepare_reddit_dual_layers.py
  python prepare_reddit_dual_layers.py --reddit-xlsx ..\\reddit_scraper\\data\\file.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from merge_sentiment_to_market import (
    DEFAULT_SUBREDDIT_TO_TICKER,
    add_lagged_sentiment,
    detect_sentiment_columns,
    et_calendar_date,
    load_trading_day_index,
    next_trading_day,
    read_sentiment_table,
    subreddit_to_ticker,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MARKET = SCRIPT_DIR / "market_daily_panel.csv"
DEFAULT_REDDIT = SCRIPT_DIR.parent / "reddit_scraper" / "data" / "reddit_NEW_RUN_with_sentiment.xlsx"
OUT_DIR_DEFAULT = SCRIPT_DIR / "reddit_prepared"

EXCLUDE_SENTIMENT = {
    "submission_id",
    "comment_id",
    "timestamp",
    "subreddit",
    "text_type",
    "text",
    "text_raw",
    "text_clean",
    "uri",
    "cid",
    "author_handle",
    "author_display_name",
    "keyword",
    "truncated",
    "lang",
    "text_en",
    "cleaning_version",
    "created_at",
    "time",
    "date",
    "Date",
    "nrc_dominant_emotion",
    "hf_dominant_sentiment",
    "fb_dominant_sentiment",
}


def assign_trading_dates(
    df: pd.DataFrame, time_col: str, sorted_days: pd.DatetimeIndex
) -> pd.Series:
    cal = et_calendar_date(df[time_col])
    return cal.map(lambda d: next_trading_day(d, sorted_days))


def engagement_weight(df: pd.DataFrame) -> pd.Series:
    """
    Reddit-style engagement weights: log1p(score or ups) * log1p(num_comments).
    If no engagement columns exist, weights are 1 (EW sentiment equals simple mean).
    """
    n = len(df)
    w = np.ones(n, dtype=float)
    if "score" in df.columns:
        x = pd.to_numeric(df["score"], errors="coerce").fillna(0).to_numpy()
        w *= 1.0 + np.log1p(np.maximum(x, 0.0))
    elif "ups" in df.columns:
        x = pd.to_numeric(df["ups"], errors="coerce").fillna(0).to_numpy()
        w *= 1.0 + np.log1p(np.maximum(x, 0.0))
    if "num_comments" in df.columns:
        c = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0).to_numpy()
        w *= 1.0 + np.log1p(np.maximum(c, 0.0))
    return pd.Series(w, index=df.index, dtype=float)


def _grouped_nrc_net_ew(
    work: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Per group: engagement-weighted mean of nrc_net_sentiment (NaN if no valid values)."""
    rows = []
    for key, sub in work.groupby(group_cols, sort=True):
        w = np.maximum(sub["_ew"].to_numpy(dtype=float), 1e-12)
        v = pd.to_numeric(sub["nrc_net_sentiment"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(v)
        if not m.any():
            val = np.nan
        else:
            ww, vv = w[m], v[m]
            val = float(np.dot(ww, vv) / ww.sum())
        if isinstance(key, tuple):
            rows.append((*key, val))
        else:
            rows.append((key, val))
    return pd.DataFrame(rows, columns=[*group_cols, "nrc_net_sentiment_ew"])


def aggregate_by_ticker_date(
    df: pd.DataFrame,
    sentiment_cols: list[str],
    ticker: pd.Series,
    trading_date: pd.Series,
) -> pd.DataFrame:
    agg = df.loc[:, sentiment_cols].copy()
    agg["ticker"] = ticker.values
    agg["date"] = trading_date.values
    agg = agg.dropna(subset=["ticker", "date"])
    agg["_ew"] = engagement_weight(df).values
    g = agg.groupby(["ticker", "date"], sort=True)
    means = g[sentiment_cols].mean()
    counts = g.size().rename("n_posts")
    out = means.join(counts, how="left")
    if "nrc_net_sentiment" in agg.columns:
        ew = _grouped_nrc_net_ew(agg, ["ticker", "date"])
        out = out.reset_index().merge(ew, on=["ticker", "date"], how="left")
    else:
        out = out.reset_index()
    return out


def aggregate_by_date_only(
    df: pd.DataFrame,
    sentiment_cols: list[str],
    trading_date: pd.Series,
) -> pd.DataFrame:
    agg = df.loc[:, sentiment_cols].copy()
    agg["date"] = trading_date.values
    agg = agg.dropna(subset=["date"])
    agg["_ew"] = engagement_weight(df).values
    g = agg.groupby("date", sort=True)
    means = g[sentiment_cols].mean()
    counts = g.size().rename("n_posts")
    out = means.join(counts, how="left")
    if "nrc_net_sentiment" in agg.columns:
        ew = _grouped_nrc_net_ew(agg, ["date"])
        out = out.reset_index().merge(ew, on="date", how="left")
    else:
        out = out.reset_index()
    return out


def merge_firm_with_market(
    market: pd.DataFrame, daily_firm: pd.DataFrame, ticker: str, sentiment_cols: list[str]
) -> pd.DataFrame:
    m = market[market["ticker"] == ticker].copy()
    out = m.merge(daily_firm[daily_firm["ticker"] == ticker].drop(columns=["ticker"]), on="date", how="left")
    lag_cols = sentiment_cols + (["n_posts"] if "n_posts" in out.columns else [])
    lag_cols = [c for c in lag_cols if c in out.columns]
    return add_lagged_sentiment(out, lag_cols)


def benchmark_safe_stem(symbol: str) -> str:
    """File stem for Yahoo symbol (e.g. ^VIX -> VIX)."""
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


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare firm vs broad Reddit sentiment layers.")
    p.add_argument("--reddit-xlsx", type=Path, default=DEFAULT_REDDIT)
    p.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    p.add_argument("-o", "--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--time-col", default="time")
    p.add_argument("--subreddit-col", default="subreddit")
    p.add_argument(
        "--firm-subreddits",
        default="nvidia,amd,intel",
        help="Lowercase subreddit names treated as company-specific (comma-separated)",
    )
    p.add_argument(
        "--benchmarks",
        default="SOXX,SPY,^VIX",
        help="Yahoo symbols to save for broad-layer merge (comma-separated); empty to skip",
    )
    args = p.parse_args()

    firm_subs = frozenset(s.strip().lower() for s in args.firm_subreddits.split(",") if s.strip())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.reddit_xlsx} ...")
    sent = read_sentiment_table(args.reddit_xlsx)
    if args.time_col not in sent.columns or args.subreddit_col not in sent.columns:
        raise SystemExit(f"Need columns {args.time_col!r} and {args.subreddit_col!r}")

    market = pd.read_csv(args.market)
    market["date"] = pd.to_datetime(market["date"], format="mixed", utc=False).dt.normalize()
    sorted_days, _ = load_trading_day_index(market)

    sentiment_cols = detect_sentiment_columns(sent, EXCLUDE_SENTIMENT)
    if not sentiment_cols:
        raise SystemExit("No numeric sentiment columns found")

    sub_lower = sent[args.subreddit_col].astype(str).str.strip().str.lower()
    tdates = assign_trading_dates(sent, args.time_col, sorted_days)

    firm_mask = sub_lower.isin(firm_subs)
    broad_mask = ~firm_mask

    # --- Firm layer: map subreddit -> ticker only for firm subs
    tick_firm = sent.loc[firm_mask, args.subreddit_col].map(
        lambda s: subreddit_to_ticker(s, DEFAULT_SUBREDDIT_TO_TICKER)
    )
    firm_df = sent.loc[firm_mask].copy()
    daily_firm = aggregate_by_ticker_date(
        firm_df, sentiment_cols, tick_firm.reset_index(drop=True), tdates[firm_mask].reset_index(drop=True)
    )

    sentiment_cols_merge = list(sentiment_cols)
    if "nrc_net_sentiment_ew" in daily_firm.columns:
        sentiment_cols_merge.append("nrc_net_sentiment_ew")

    for tkr in sorted(daily_firm["ticker"].dropna().unique()):
        sub = daily_firm[daily_firm["ticker"] == tkr].drop(columns=["ticker"])
        fp = args.out_dir / f"sentiment_firm_{tkr}_daily.csv"
        sub.to_csv(fp, index=False)
        print(f"Wrote {fp} ({len(sub)} days)")

        merged = merge_firm_with_market(market, daily_firm, tkr, sentiment_cols_merge)
        mp = args.out_dir / f"merged_{tkr}_market_sentiment.csv"
        merged.to_csv(mp, index=False)
        print(f"Wrote {mp} ({len(merged)} rows)")

    # --- Broad layer
    broad_df = sent.loc[broad_mask].copy()
    daily_broad = aggregate_by_date_only(
        broad_df, sentiment_cols, tdates[broad_mask].reset_index(drop=True)
    )
    broad_path = args.out_dir / "sentiment_broad_daily.csv"
    daily_broad.to_csv(broad_path, index=False)
    print(f"Wrote {broad_path} ({len(daily_broad)} days, n_posts broad aggregate)")

    # Benchmarks for broad validation
    if args.benchmarks.strip():
        dmin, dmax = daily_broad["date"].min(), daily_broad["date"].max()
        for raw in (s.strip() for s in args.benchmarks.split(",") if s.strip()):
            sym = raw.upper() if not raw.startswith("^") else f"^{raw[1:].upper()}"
            stem = benchmark_safe_stem(sym)
            download_benchmark(sym, dmin, dmax, args.out_dir / f"benchmark_{stem}.csv")

    def write_broad_benchmark_preview(stem: str) -> None:
        bench_path = args.out_dir / f"benchmark_{stem}.csv"
        if not bench_path.exists():
            return
        bench = pd.read_csv(bench_path)
        bench["date"] = pd.to_datetime(bench["date"], format="mixed", utc=False).dt.normalize()
        broad_m = bench.merge(daily_broad, on="date", how="inner")
        lag_cols = [c for c in sentiment_cols if c in broad_m.columns] + ["n_posts"]
        if "nrc_net_sentiment_ew" in broad_m.columns:
            lag_cols.append("nrc_net_sentiment_ew")
        broad_m = broad_m.sort_values("date")
        for c in lag_cols:
            if c in broad_m.columns:
                broad_m[f"{c}_lag1"] = broad_m[c].shift(1)
        preview = args.out_dir / f"merged_BROAD_{stem}_preview.csv"
        broad_m.to_csv(preview, index=False)
        print(f"Wrote {preview} (inner join broad sentiment x {stem})")

    for raw in (s.strip() for s in args.benchmarks.split(",") if s.strip()):
        sym = raw.upper() if not raw.startswith("^") else f"^{raw[1:].upper()}"
        write_broad_benchmark_preview(benchmark_safe_stem(sym))

    n_firm = int(firm_mask.sum())
    n_broad = int(broad_mask.sum())
    print(f"Done. Firm-sub rows: {n_firm:,} | Broad-sub rows: {n_broad:,}")


if __name__ == "__main__":
    main()
