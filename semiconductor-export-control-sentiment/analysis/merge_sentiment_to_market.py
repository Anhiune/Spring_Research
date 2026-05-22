#!/usr/bin/env python3
"""
Merge daily sentiment (row-level or pre-aggregated) with market_daily_panel.csv.

Timing rule (default):
  - Post timestamp -> US/Eastern calendar date -> trading_date using dates that
    exist in the market panel (weekends/holidays roll FORWARD to next trading day).
  - Row-level: aggregate mean + n_posts per (ticker, trading_date).
  - Baseline spec helpers: sentiment_*_lag1 = previous trading day's value per ticker.

Usage (row-level, e.g. Reddit with created_at):
  python merge_sentiment_to_market.py --sentiment path/to/scored.csv \\
      --time-col created_at --ticker-col ticker

  If you only have a keyword column:
  python merge_sentiment_to_market.py --sentiment scored.csv \\
      --time-col created_at --keyword-col keyword

Pre-aggregated daily (one firm, no ticker column):
  python merge_sentiment_to_market.py --sentiment daily.csv --time-col date \\
      --default-ticker INTC --mode daily

Outputs:
  sentiment_by_trading_day.csv   (optional, --save-sentiment-daily)
  market_with_sentiment.csv      merged panel (left join from market)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MARKET = SCRIPT_DIR / "market_daily_panel.csv"

# Scrape keywords / names -> Yahoo ticker (extend via --keyword-map JSON)
# r/<name> (lowercase key) -> Yahoo ticker for rows without explicit ticker column
DEFAULT_SUBREDDIT_TO_TICKER = {
    "nvidia": "NVDA",
    "amd": "AMD",
    "intel": "INTC",
    "tsmc": "TSM",
    "micron": "MU",
    "asml": "ASML",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "marvell": "MRVL",
    "nvda": "NVDA",
    "intc": "INTC",
    "mu": "MU",
    "qcom": "QCOM",
    "avgo": "AVGO",
    "mrvl": "MRVL",
}

DEFAULT_KEYWORD_TO_TICKER = {
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "amd": "AMD",
    "intel": "INTC",
    "intc": "INTC",
    "tsmc": "TSM",
    "tsm": "TSM",
    "qualcomm": "QCOM",
    "qcom": "QCOM",
    "micron": "MU",
    "mu": "MU",
    "asml": "ASML",
    "applied materials": "AMAT",
    "amat": "AMAT",
    "lam research": "LRCX",
    "lrcx": "LRCX",
    "klac": "KLAC",
    "kla": "KLAC",
    "broadcom": "AVGO",
    "avgo": "AVGO",
    "marvell": "MRVL",
    "mrvl": "MRVL",
}


def load_trading_day_index(market: pd.DataFrame) -> tuple[pd.DatetimeIndex, set[pd.Timestamp]]:
    days = pd.to_datetime(market["date"], format="mixed", utc=False).dt.normalize().unique()
    days = pd.DatetimeIndex(sorted(days))
    return days, set(days)


def next_trading_day(cal_day: pd.Timestamp, sorted_days: pd.DatetimeIndex) -> pd.Timestamp | pd.NaT:
    """Roll calendar day to same day if it trades, else next session in sorted_days."""
    cal_day = pd.Timestamp(cal_day).normalize()
    idx = sorted_days.searchsorted(cal_day, side="left")
    if idx < len(sorted_days) and sorted_days[idx] == cal_day:
        return cal_day
    if idx < len(sorted_days):
        return sorted_days[idx]
    return pd.NaT


def et_calendar_date(series: pd.Series) -> pd.Series:
    """Parse datetimes and return normalized calendar date in America/New_York."""
    dt = pd.to_datetime(series, format="mixed", utc=True)
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    else:
        dt = dt.dt.tz_convert("UTC")
    et = dt.dt.tz_convert("America/New_York").dt.normalize()
    return et.dt.tz_localize(None)


def read_sentiment_table(path: Path) -> pd.DataFrame:
    """Load CSV or Excel sentiment export."""
    suf = path.suffix.lower()
    if suf == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if suf == ".xls":
        return pd.read_excel(path, engine="xlrd")
    return pd.read_csv(path, low_memory=False)


def subreddit_to_ticker(name: str, sub_map: dict[str, str]) -> str | None:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = str(name).strip().lower()
    if not s:
        return None
    return sub_map.get(s)


def keyword_to_ticker(
    kw: str,
    extra_map: dict[str, str],
) -> str | None:
    if kw is None or (isinstance(kw, float) and pd.isna(kw)):
        return None
    s = str(kw).strip()
    if not s:
        return None
    merged = {**DEFAULT_KEYWORD_TO_TICKER, **extra_map}
    u = s.upper()
    if re.fullmatch(r"[A-Z]{1,5}", u):
        return u
    key = s.lower()
    return merged.get(key)


def detect_sentiment_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype in ("float64", "float32", "int64", "int32"):
            cols.append(c)
        else:
            try:
                pd.to_numeric(df[c], errors="raise")
                cols.append(c)
            except (ValueError, TypeError):
                pass
    return cols


def aggregate_row_level(
    sent: pd.DataFrame,
    time_col: str,
    sorted_days: pd.DatetimeIndex,
    ticker_series: pd.Series,
    sentiment_cols: list[str],
) -> pd.DataFrame:
    cal = et_calendar_date(sent[time_col])
    tdates = cal.map(lambda d: next_trading_day(d, sorted_days))
    agg = sent.loc[:, sentiment_cols].copy()
    agg["ticker"] = ticker_series.values
    agg["trading_date"] = tdates.values
    agg = agg.dropna(subset=["ticker", "trading_date"])
    g = agg.groupby(["ticker", "trading_date"], sort=True)
    means = g[sentiment_cols].mean()
    counts = g.size().rename("n_posts")
    out = means.join(counts, how="left").reset_index()
    out = out.rename(columns={"trading_date": "date"})
    return out


def add_lagged_sentiment(df: pd.DataFrame, sentiment_cols: list[str]) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    for c in sentiment_cols:
        out[f"{c}_lag1"] = out.groupby("ticker", sort=False)[c].shift(1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Merge sentiment CSV with market_daily_panel.")
    p.add_argument("--sentiment", required=True, type=Path, help="Path to sentiment CSV")
    p.add_argument("--market", type=Path, default=DEFAULT_MARKET, help="market_daily_panel.csv")
    p.add_argument(
        "--mode",
        choices=("auto", "row", "daily"),
        default="auto",
        help="row = one row per post/sentence; daily = already daily aggregates",
    )
    p.add_argument("--time-col", default=None, help="created_at / timestamp / date")
    p.add_argument("--ticker-col", default=None, help="Column with ticker symbol")
    p.add_argument("--keyword-col", default=None, help="Map keyword to ticker if no ticker col")
    p.add_argument(
        "--subreddit-col",
        default=None,
        help="Map subreddit name to ticker (e.g. nvidia->NVDA); extend with --subreddit-map JSON",
    )
    p.add_argument(
        "--subreddit-map",
        type=Path,
        default=None,
        help='JSON {"nvidia": "NVDA", ...} merged into default subreddit->ticker map',
    )
    p.add_argument(
        "--keyword-map",
        type=Path,
        default=None,
        help="JSON object {\"nvidia\": \"NVDA\", ...} merged into default map",
    )
    p.add_argument(
        "--default-ticker",
        default=None,
        help="If daily file has no ticker column, set this (e.g. INTC)",
    )
    p.add_argument(
        "--sentiment-cols",
        default="",
        help="Comma-separated sentiment columns to keep (default: all numeric except ids)",
    )
    p.add_argument("-o", "--output", type=Path, default=SCRIPT_DIR / "market_with_sentiment.csv")
    p.add_argument(
        "--save-sentiment-daily",
        type=Path,
        default=None,
        help="Write aggregated sentiment-by-trading-day table to this path",
    )
    args = p.parse_args()

    market = pd.read_csv(args.market)
    market["date"] = pd.to_datetime(market["date"], format="mixed", utc=False).dt.normalize()
    sorted_days, _ = load_trading_day_index(market)

    extra_map: dict[str, str] = {}
    if args.keyword_map and args.keyword_map.exists():
        extra_map = {k.lower(): v.upper() for k, v in json.loads(args.keyword_map.read_text(encoding="utf-8")).items()}

    sub_map: dict[str, str] = {**DEFAULT_SUBREDDIT_TO_TICKER}
    if args.subreddit_map and args.subreddit_map.exists():
        sub_map.update(
            {k.lower(): v.upper() for k, v in json.loads(args.subreddit_map.read_text(encoding="utf-8")).items()}
        )

    print(f"Loading {args.sentiment} ...")
    sent = read_sentiment_table(args.sentiment)
    cols_lower = {c.lower(): c for c in sent.columns}

    mode = args.mode
    if mode == "auto":
        if args.time_col:
            mode = "row"
        elif "created_at" in sent.columns or "timestamp" in cols_lower or "time" in sent.columns:
            mode = "row"
        elif "date" in cols_lower or "Date" in sent.columns:
            mode = "daily"
        else:
            raise SystemExit("Could not infer --mode; set --mode row|daily and --time-col")

    time_col = args.time_col
    if mode == "row":
        if not time_col:
            if "created_at" in sent.columns:
                time_col = "created_at"
            elif "time" in sent.columns:
                time_col = "time"
            else:
                time_col = cols_lower.get("timestamp")
        if not time_col or time_col not in sent.columns:
            raise SystemExit(f"Row mode needs a valid time column; got {time_col!r}, columns={list(sent.columns)}")

    exclude = {
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

    if args.sentiment_cols.strip():
        sentiment_cols = [c.strip() for c in args.sentiment_cols.split(",") if c.strip()]
    else:
        sentiment_cols = detect_sentiment_columns(sent, exclude)
        if not sentiment_cols:
            raise SystemExit("No numeric sentiment columns found; set --sentiment-cols")

    if mode == "daily":
        dcol = args.time_col or ("date" if "date" in sent.columns else cols_lower.get("date"))
        if not dcol or dcol not in sent.columns:
            raise SystemExit(f"Daily mode needs date column; use --time-col. Columns={list(sent.columns)}")
        sent = sent.copy()
        sent["_cal"] = pd.to_datetime(sent[dcol], format="mixed", utc=False).dt.normalize()
        sent["date"] = sent["_cal"].map(lambda d: next_trading_day(d, sorted_days))
        sent = sent.dropna(subset=["date"])
        if args.ticker_col and args.ticker_col in sent.columns:
            sent["ticker"] = sent[args.ticker_col].astype(str).str.upper()
        elif args.default_ticker:
            sent["ticker"] = args.default_ticker.upper()
        else:
            raise SystemExit("Daily mode: set --default-ticker or --ticker-col")
        daily = sent.groupby(["ticker", "date"], sort=True)[sentiment_cols].mean().reset_index()
        if "n_posts" not in daily.columns:
            counts = sent.groupby(["ticker", "date"]).size().rename("n_posts").reset_index()
            daily = daily.merge(counts, on=["ticker", "date"], how="left")
    else:
        if args.ticker_col and args.ticker_col in sent.columns:
            tick = sent[args.ticker_col].astype(str).str.upper()
        elif args.keyword_col and args.keyword_col in sent.columns:
            tick = sent[args.keyword_col].map(lambda k: keyword_to_ticker(k, extra_map))
        elif args.subreddit_col and args.subreddit_col in sent.columns:
            tick = sent[args.subreddit_col].map(lambda s: subreddit_to_ticker(s, sub_map))
        else:
            raise SystemExit(
                "Row mode needs --ticker-col, --keyword-col, or --subreddit-col (or use --mode daily)"
            )
        daily = aggregate_row_level(sent, time_col, sorted_days, tick, sentiment_cols)
        miss = tick.isna() | (tick == "NONE")
        if miss.any():
            dropped = int(miss.sum())
            print(f"Note: dropped {dropped} rows with unmapped ticker/subreddit/keyword")

    if args.save_sentiment_daily:
        daily.to_csv(args.save_sentiment_daily, index=False)
        print(f"Wrote aggregated sentiment: {args.save_sentiment_daily}")

    merged = market.merge(daily, on=["ticker", "date"], how="left", suffixes=("", "_sent"))
    merged = add_lagged_sentiment(merged, sentiment_cols + (["n_posts"] if "n_posts" in merged.columns else []))

    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged):,} rows to {args.output}")
    print(f"Sentiment columns (with _lag1): {sentiment_cols[:6]}{'...' if len(sentiment_cols) > 6 else ''}")


if __name__ == "__main__":
    main()
