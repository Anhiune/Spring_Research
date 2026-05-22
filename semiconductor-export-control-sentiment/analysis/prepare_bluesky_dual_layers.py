#!/usr/bin/env python3
"""
Prepare Bluesky row-level sentiment (NRC + HF + FinBERT) for the same downstream
pipeline as Reddit: firm-keyword layer (NVDA / AMD / INTC), broad layer, daily
aggregates on the US/Eastern â†’ trading calendar, and merges with market_daily_panel.

Expects output from sentiment_scoring (e.g. bluesky_all_cleaned_with_sentiment.csv).

Usage (from analysis/):
  python prepare_bluesky_dual_layers.py \\
    --bluesky-csv ..\\\\bluesky_data\\\\bluesky_all_cleaned_with_sentiment.csv

Outputs under bluesky_prepared/ by default (mirror reddit_prepared/ layout):
  sentiment_firm_{NVDA,AMD,INTC}_daily.csv, merged_*_market_sentiment.csv,
  sentiment_broad_daily.csv, benchmark_*.csv, merged_BROAD_*_preview.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from merge_sentiment_to_market import (
    add_lagged_sentiment,
    detect_sentiment_columns,
    et_calendar_date,
    keyword_to_ticker,
    load_trading_day_index,
    next_trading_day,
    read_sentiment_table,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MARKET = SCRIPT_DIR / "market_daily_panel.csv"
DEFAULT_BLUESKY = SCRIPT_DIR.parent / "bluesky_data" / "bluesky_all_cleaned_with_sentiment.csv"
OUT_DIR_DEFAULT = SCRIPT_DIR / "bluesky_prepared"

FIRM_TICKERS = frozenset({"NVDA", "AMD", "INTC"})

EXCLUDE_SENTIMENT = {
    "uri",
    "cid",
    "author_handle",
    "author_display_name",
    "keyword",
    "text",
    "text_raw",
    "text_clean",
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
    "like_count",
    "repost_count",
    "reply_count",
    "quote_count",
}


def assign_trading_dates(
    df: pd.DataFrame, time_col: str, sorted_days: pd.DatetimeIndex
) -> pd.Series:
    cal = et_calendar_date(df[time_col])
    return cal.map(lambda d: next_trading_day(d, sorted_days))


def engagement_weight_bluesky(df: pd.DataFrame) -> pd.Series:
    """Bluesky-style weights: product of log1p(+counts) across engagement columns."""
    n = len(df)
    w = np.ones(n, dtype=float)
    for col in ("like_count", "repost_count", "reply_count", "quote_count"):
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        w *= 1.0 + np.log1p(np.maximum(x, 0.0))
    return pd.Series(w, index=df.index, dtype=float)


def _grouped_nrc_net_ew(
    work: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
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
    agg["_ew"] = engagement_weight_bluesky(df).values
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
    agg["_ew"] = engagement_weight_bluesky(df).values
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
    out = m.merge(
        daily_firm[daily_firm["ticker"] == ticker].drop(columns=["ticker"]),
        on="date",
        how="left",
    )
    lag_cols = sentiment_cols + (["n_posts"] if "n_posts" in out.columns else [])
    lag_cols = [c for c in lag_cols if c in out.columns]
    return add_lagged_sentiment(out, lag_cols)


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


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare firm vs broad Bluesky sentiment layers.")
    p.add_argument("--bluesky-csv", type=Path, default=DEFAULT_BLUESKY, help="Row-level scored CSV")
    p.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    p.add_argument("-o", "--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--time-col", default="created_at")
    p.add_argument("--keyword-col", default="keyword")
    p.add_argument(
        "--benchmarks",
        default="SOXX,SPY,^VIX",
        help="Yahoo symbols for broad-layer merge (comma-separated); empty to skip",
    )
    args = p.parse_args()

    if not args.bluesky_csv.exists():
        raise SystemExit(
            f"Missing {args.bluesky_csv}\n"
            "Run sentiment scoring first, e.g.:\n"
            "  python sentiment_scoring_updated.py bluesky_data/bluesky_all_cleaned.csv "
            "--text-col text_clean --date-col created_at --output-dir bluesky_data"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.bluesky_csv} ...")
    sent = read_sentiment_table(args.bluesky_csv)
    if args.time_col not in sent.columns or args.keyword_col not in sent.columns:
        raise SystemExit(f"Need columns {args.time_col!r} and {args.keyword_col!r}")

    market = pd.read_csv(args.market)
    market["date"] = pd.to_datetime(market["date"], format="mixed", utc=False).dt.normalize()
    sorted_days, _ = load_trading_day_index(market)

    sentiment_cols = detect_sentiment_columns(sent, EXCLUDE_SENTIMENT)
    if not sentiment_cols:
        raise SystemExit("No numeric sentiment columns found (run scoring on this file first).")

    kw = sent[args.keyword_col].map(lambda k: keyword_to_ticker(str(k), {}))
    firm_mask = kw.isin(FIRM_TICKERS)
    broad_mask = ~firm_mask

    tdates = assign_trading_dates(sent, args.time_col, sorted_days)

    firm_df = sent.loc[firm_mask].copy()
    tick_firm = kw.loc[firm_mask]
    daily_firm = aggregate_by_ticker_date(
        firm_df,
        sentiment_cols,
        tick_firm.reset_index(drop=True),
        tdates[firm_mask].reset_index(drop=True),
    )

    sentiment_cols_merge = list(sentiment_cols)
    if "nrc_net_sentiment_ew" in daily_firm.columns:
        sentiment_cols_merge.append("nrc_net_sentiment_ew")

    for tkr in sorted(daily_firm["ticker"].dropna().unique()):
        merged = merge_firm_with_market(market, daily_firm, tkr, sentiment_cols_merge)
        sub = daily_firm[daily_firm["ticker"] == tkr].drop(columns=["ticker"])
        fp = args.out_dir / f"sentiment_firm_{tkr}_daily.csv"
        sub.to_csv(fp, index=False)
        print(f"Wrote {fp} ({len(sub)} days)")

        mp = args.out_dir / f"merged_{tkr}_market_sentiment.csv"
        merged.to_csv(mp, index=False)
        print(f"Wrote {mp} ({len(merged)} rows)")

    broad_df = sent.loc[broad_mask].copy()
    daily_broad = aggregate_by_date_only(
        broad_df, sentiment_cols, tdates[broad_mask].reset_index(drop=True)
    )
    mean_rename = {c: f"{c}_mean" for c in sentiment_cols if c in daily_broad.columns}
    if mean_rename:
        daily_broad = daily_broad.rename(columns=mean_rename)
    broad_sent_cols = list(mean_rename.values()) if mean_rename else list(sentiment_cols)
    broad_path = args.out_dir / "sentiment_broad_daily.csv"
    daily_broad.to_csv(broad_path, index=False)
    print(f"Wrote {broad_path} ({len(daily_broad)} days, n_posts broad aggregate)")

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
        lag_cols = [c for c in broad_sent_cols if c in broad_m.columns] + ["n_posts"]
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
    print(f"Done. Firm-keyword rows: {n_firm:,} | Broad/other rows: {n_broad:,}")


if __name__ == "__main__":
    main()

