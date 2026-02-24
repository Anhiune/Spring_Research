# -*- coding: utf-8 -*-
"""
Free TSLA data workflow (Yahoo Finance) with robust timezone/index handling.

Outputs:
- prices (daily OHLCV + Adj Return)
- income/balance/cashflow (annual & quarterly, tidy)
- earnings dates (EPS est/actual/surprise)
- quarterly TTM fields (rev, NI, etc.)
- daily MarketCap & EV (approx), plus quarterly ratios:
  - P_E_TTM, P_S_TTM, Debt_Equity, EV_EBITDA_TTM (approx)
- modeling_daily sheet (fundamentals forward-filled to trading days)
- OPTIONAL: modeling_daily_with_sentiment if SENTIMENT_CSV is provided

Requires:
  pip install yfinance pandas openpyxl numpy
"""

import pandas as pd
import numpy as np
import yfinance as yf

# ------------ Settings ------------
TICKER = "TSLA"
START = "2015-01-01"
END   = None  # None = today
OUT_XLSX = "tesla_yf_fundamentals.xlsx"

# OPTIONAL: merge your sentiment file (CSV) by Date
# Set to a path like r"c:\path\to\sentiment.csv" (or leave as None to skip)
SENTIMENT_CSV = None
SENTIMENT_DATE_COL = "Date"
SENTIMENT_VALUE_COL = "Sentiment_Score"

# ------------ Helpers ------------
def _to_naive_datetime_index(idx) -> pd.DatetimeIndex:
    """Return tz-naive DatetimeIndex (convert tz->UTC then drop tz if needed)."""
    idx = pd.to_datetime(idx)
    tz = getattr(idx, "tz", None)
    if tz is not None:
        try:
            idx = idx.tz_convert("UTC").tz_localize(None)
        except Exception:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                pass
    return idx

def clean_unique_index(obj):
    """Ensure tz-naive, single-level DatetimeIndex; sorted; drop duplicates (keep last)."""
    if obj is None or (hasattr(obj, "empty") and obj.empty):
        return obj
    out = obj.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out.index = _to_naive_datetime_index(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out

def safe_t(df):
    """Transpose Yahoo fundamentals to date-indexed DataFrame, cleaned."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.T.copy()
    return clean_unique_index(out)

def add_ttm_columns(quarterly_df, cols):
    """Build TTM (trailing 12 months) from quarterly sums for flow items."""
    if quarterly_df is None or quarterly_df.empty:
        return quarterly_df
    q = quarterly_df.sort_index()
    for c in cols:
        if c in q.columns:
            q[f"{c}_TTM"] = q[c].rolling(4, min_periods=2).sum()
    return q

def extract_col(df, names):
    """Return first present column from 'names' within df."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.columns:
            return df[n]
    return None

def first_nonnull(*args):
    """Return first non-null/non-empty Series among inputs."""
    for s in args:
        if s is not None and hasattr(s, "empty") and not s.empty:
            return s
    return None

def as_series(x, name=None):
    """Coerce DataFrame/ndarray/scalar to a pandas Series."""
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        s = x.squeeze()
    elif isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    if name:
        s.name = name
    return s

def align_to_prices(df_like, prices_idx):
    """Return df_like aligned to prices_idx with tz-naive single-level index."""
    if df_like is None:
        return pd.DataFrame(index=prices_idx)
    df = df_like.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.reindex(prices_idx)

# ------------ Fetch Yahoo Data ------------
tk = yf.Ticker(TICKER)

# Daily OHLCV
prices = yf.download(TICKER, start=START, end=END, auto_adjust=False)
prices.index.name = "Date"
prices = clean_unique_index(prices)
prices["Adj Return"] = prices["Adj Close"].pct_change()

# Fundamentals (annual & quarterly)
is_a = safe_t(tk.financials)                # income statement (annual)
is_q = safe_t(tk.quarterly_financials)      # income statement (quarterly)
bs_a = safe_t(tk.balance_sheet)             # balance sheet (annual)
bs_q = safe_t(tk.quarterly_balance_sheet)   # balance sheet (quarterly)
cf_a = safe_t(tk.cashflow)                  # cash flow (annual)
cf_q = safe_t(tk.quarterly_cashflow)        # cash flow (quarterly)

# Earnings dates (EPS actual/estimate/surprise)
try:
    earn = tk.get_earnings_dates(limit=40)  # recent years
    earnings_dates = earn[["Earnings Date", "EPS Estimate", "Reported EPS", "Surprise(%)"]].copy()
    earnings_dates.rename(columns={
        "Earnings Date": "earnings_date",
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_actual",
        "Surprise(%)": "surprise_pct"
    }, inplace=True)
    earnings_dates["earnings_date"] = _to_naive_datetime_index(earnings_dates["earnings_date"])
except Exception:
    earnings_dates = pd.DataFrame(columns=["earnings_date","eps_estimate","eps_actual","surprise_pct"])

# ------------ Create TTM & derived metrics ------------
# Build TTM flows from quarterly IS/CF (flows only)
is_q = add_ttm_columns(is_q, cols=[
    "Total Revenue", "Operating Income", "Gross Profit", "Net Income"
])
cf_q = add_ttm_columns(cf_q, cols=[
    "Free Cash Flow", "Operating Cash Flow"
])

# Clean again after transformation (ensures tz-naive, deduped)
is_q = clean_unique_index(is_q)
bs_q = clean_unique_index(bs_q)

# ---- Shares Outstanding & MarketCap (robust handling) ----
shares_series = None
try:
    sh = tk.get_shares_full(start=START)
    if sh is not None and not sh.empty:
        if isinstance(sh, pd.DataFrame) and sh.shape[1] >= 1:
            col = sh.columns[sh.notna().sum().argmax()]
            shares_series = as_series(sh[col], name="Shares Outstanding")
        else:
            shares_series = as_series(sh, name="Shares Outstanding")
        shares_series.index = _to_naive_datetime_index(shares_series.index)
        shares_series = clean_unique_index(shares_series)
except Exception:
    shares_series = None

# Fallback 1: "Ordinary Shares Number" from quarterly balance sheet
if (shares_series is None or shares_series.empty) and ("Ordinary Shares Number" in bs_q.columns):
    shares_series = as_series(bs_q["Ordinary Shares Number"], name="Shares Outstanding")
    shares_series = clean_unique_index(shares_series)

# Fallback 2: fast_info scalar -> fill across trading days
if (shares_series is None) or shares_series.empty:
    so = None
    try:
        fi = getattr(tk, "fast_info", None)
        if isinstance(fi, dict):
            so = fi.get("sharesOutstanding", None)
    except Exception:
        so = None
    if so is not None and not pd.isna(so):
        shares_series = pd.Series(so, index=prices.index, name="Shares Outstanding")
        shares_series = clean_unique_index(shares_series)

# MarketCap (daily) if we have shares
mcap_daily = None
if shares_series is not None and not shares_series.empty:
    shares_daily = shares_series.reindex(prices.index).ffill()
    shares_daily = as_series(shares_daily, name="Shares Outstanding")
    adj_close = prices["Adj Close"]
    if isinstance(adj_close, pd.DataFrame):
        adj_close = adj_close.iloc[:, 0].squeeze()
    adj_close = as_series(adj_close, name="Adj Close")
    mcap_series = adj_close * shares_daily
    mcap_series.name = "MarketCap_Daily"
    mcap_daily = mcap_series.to_frame()

# ---- EV approximation ----
# EV ≈ MarketCap + TotalDebt - Cash (& short-term investments)
total_debt_q = first_nonnull(
    extract_col(bs_q, ["Total Debt", "Total Debt Net", "Short Long Term Debt Total", "Short Long Term Debt"])
)
cash_q = first_nonnull(
    extract_col(bs_q, ["Cash And Cash Equivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"])
)

ev_daily = None
if (mcap_daily is not None) and (total_debt_q is not None) and (cash_q is not None):
    comp_q = pd.concat(
        [
            as_series(total_debt_q, "TotalDebt_Q"),
            as_series(cash_q, "Cash_Q")
        ],
        axis=1
    ).dropna(how="all")
    comp_q = clean_unique_index(comp_q)
    comp_daily = comp_q.reindex(prices.index).ffill()
    ev_series = mcap_daily["MarketCap_Daily"] + comp_daily["TotalDebt_Q"] - comp_daily["Cash_Q"]
    ev_series.name = "EV_Daily"
    ev_daily = ev_series.to_frame()

# ---- Ratios (quarterly index) ----
ratios_q = pd.DataFrame(index=is_q.index)

# Align last available close before/at quarter date (ffill on price)
price_q = prices["Adj Close"].reindex(is_q.index, method="ffill")
if isinstance(price_q, pd.DataFrame):
    price_q = price_q.iloc[:, 0].squeeze()

# Revenue TTM & Net Income TTM
rev_ttm = is_q.get("Total Revenue_TTM")
ni_ttm  = is_q.get("Net Income_TTM")

# Shares for quarterly dates
shares_q = None
if shares_series is not None and not shares_series.empty:
    shares_q = as_series(shares_series.reindex(is_q.index, method="ffill"), "Shares Outstanding Q")

# P/S (TTM) ≈ MarketCap / Revenue_TTM
if (rev_ttm is not None) and (shares_q is not None):
    mcap_q = price_q * shares_q
    ratios_q["P_S_TTM"] = mcap_q / rev_ttm.replace(0, np.nan)

# EPS TTM ≈ NetIncome_TTM / Shares
eps_ttm = None
if (ni_ttm is not None) and (shares_q is not None):
    eps_ttm = ni_ttm / shares_q.replace(0, np.nan)
    ratios_q["EPS_TTM"] = eps_ttm

# P/E (TTM) ≈ Price / EPS_TTM
if eps_ttm is not None:
    ratios_q["P_E_TTM"] = price_q / eps_ttm.replace(0, np.nan)

# Debt/Equity (quarterly)
total_equity_q = first_nonnull(
    extract_col(bs_q, ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Stockholders Equity"])
)
if (total_debt_q is not None) and (total_equity_q is not None):
    ratios_q["Debt_Equity"] = total_debt_q / total_equity_q.replace(0, np.nan)

# EV/EBITDA (approx): need EBITDA_TTM; if missing, proxy with Operating Income_TTM
ebitda_ttm = is_q.get("EBITDA_TTM")
if ebitda_ttm is None:
    ebitda_ttm = is_q.get("Operating Income_TTM")  # rough proxy
if (ev_daily is not None) and (ebitda_ttm is not None):
    ev_q = ev_daily["EV_Daily"].reindex(is_q.index, method="ffill")
    ratios_q["EV_EBITDA_TTM"] = ev_q / ebitda_ttm.replace(0, np.nan)

# ------------ Build a daily modeling table (concat, not chained joins) ------------
# Prepare aligned pieces
prices = clean_unique_index(prices)
left_core   = align_to_prices(prices[["Adj Close","Adj Return","Volume"]], prices.index)
mcap_aligned = align_to_prices(mcap_daily, prices.index)
ev_aligned   = align_to_prices(ev_daily, prices.index)

funds_q_all = (
    ratios_q
    .join(is_q[[c for c in is_q.columns if c.endswith("_TTM")]], how="left")
    .join(
        bs_q[[
            c for c in ["Total Debt", "Cash And Cash Equivalents", "Total Stockholder Equity"]
            if c in bs_q.columns
        ]],
        how="left"
    )
    .sort_index()
)
exog_aligned = align_to_prices(funds_q_all, prices.index).ffill()

# Concatenate to avoid index-level mismatch
modeling_daily = pd.concat(
    [left_core, mcap_aligned, ev_aligned, exog_aligned],
    axis=1
)

# ------------ OPTIONAL: merge sentiment into modeling_daily ------------
if SENTIMENT_CSV:
    try:
        sent = pd.read_csv(SENTIMENT_CSV)
        # Standardize column names
        if SENTIMENT_DATE_COL not in sent.columns or SENTIMENT_VALUE_COL not in sent.columns:
            # try to infer
            date_guess = [c for c in sent.columns if "date" in c.lower()][0]
            val_guess  = [c for c in sent.columns if c.lower() != date_guess.lower()][0]
            SENTIMENT_DATE_COL = date_guess
            SENTIMENT_VALUE_COL = val_guess
        sent[SENTIMENT_DATE_COL] = pd.to_datetime(sent[SENTIMENT_DATE_COL])
        sent.set_index(SENTIMENT_DATE_COL, inplace=True)
        sent.index = _to_naive_datetime_index(sent.index)
        sent = sent.sort_index()
        sent = sent[~sent.index.duplicated(keep="last")]
        sent = sent.rename(columns={SENTIMENT_VALUE_COL: "Sentiment"})
        sent_aligned = align_to_prices(sent[["Sentiment"]], prices.index)
        modeling_daily_with_sentiment = pd.concat([modeling_daily, sent_aligned], axis=1)

    except Exception as e:
        print(f"⚠️ Could not merge sentiment: {e}")
        modeling_daily_with_sentiment = None
else:
    modeling_daily_with_sentiment = None

# ------------ Write Excel ------------
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
    prices.to_excel(xw, sheet_name="prices")
    is_a.to_excel(xw, sheet_name="income_annual")
    is_q.to_excel(xw, sheet_name="income_quarterly")
    bs_a.to_excel(xw, sheet_name="balance_annual")
    bs_q.to_excel(xw, sheet_name="balance_quarterly")
    cf_a.to_excel(xw, sheet_name="cashflow_annual")
    cf_q.to_excel(xw, sheet_name="cashflow_quarterly")
    earnings_dates.to_excel(xw, sheet_name="earnings_dates", index=False)
    ratios_q.to_excel(xw, sheet_name="ratios_quarterly")
    modeling_daily.to_excel(xw, sheet_name="modeling_daily")
    if modeling_daily_with_sentiment is not None:
        modeling_daily_with_sentiment.to_excel(xw, sheet_name="modeling_daily_with_sentiment")

print(f"✅ Wrote {OUT_XLSX}")
print("Sheets: prices, income_annual, income_quarterly, balance_annual, balance_quarterly, cashflow_annual, cashflow_quarterly, earnings_dates, ratios_quarterly, modeling_daily"
      + (", modeling_daily_with_sentiment" if modeling_daily_with_sentiment is not None else ""))
