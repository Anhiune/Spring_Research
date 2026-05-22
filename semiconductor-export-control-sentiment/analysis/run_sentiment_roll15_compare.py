#!/usr/bin/env python3
"""
Compare predictors of **next-day** return:

  y[t] = ret_1d[t+1]   (forward one trading row)

  â€¢ lag1 â€” single-day feature: sentiment on the **previous** trading day vs row date
          x[t] = sentiment.shift(1)  (same timing idea as *lag1* columns / conservative)

  â€¢ roll15 â€” **15-trading-day** trailing mean of sentiment ending at **t**
          x[t] = sentiment.rolling(15, min_periods=10).mean()

Sentiment series per run: NRC net, HF net (4-class), FinBERT net.

Reports in-sample HAC OLS (Neweyâ€“West) and a simple **pseudo OOS** split
(first 80% train, last 20% test, OLS coefficients fixed from train, MSE on test).

Output: analysis_output/sentiment_roll15_compare_results.csv

Usage (from analysis/):
  python run_sentiment_roll15_compare.py
  python run_sentiment_roll15_compare.py --window 21
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PREP = SCRIPT_DIR / "reddit_prepared"
OUT_CSV = SCRIPT_DIR / "analysis_output" / "sentiment_roll15_compare_results.csv"
TICKERS = ("NVDA", "AMD", "INTC")


def nw_maxlags(n: int) -> int:
    if n <= 1:
        return 1
    return int(min(20, max(1, math.floor(4 * (n / 100) ** (2 / 9)))))


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


def load_firm(tkr: str) -> pd.DataFrame:
    p = PREP / f"merged_{tkr}_market_sentiment_complete.csv"
    if not p.exists():
        p = PREP / f"merged_{tkr}_market_sentiment.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df = df.loc[:, ~df.columns.astype(str).str.endswith("_lag1")].copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False)
    df = df.sort_values("date").reset_index(drop=True)
    df["ret_1d"] = pd.to_numeric(df["ret_1d"], errors="coerce")
    df["hf_net_4"] = hf_net_4(df)
    return df


def fit_hac(
    y: pd.Series,
    x_col: pd.Series,
    sent_name: str,
) -> tuple[int, float, float, float, float, float]:
    import statsmodels.api as sm

    m = y.notna() & x_col.notna()
    y2, x2 = y[m].astype(float), x_col[m].astype(float)
    n = int(len(y2))
    if n < 30:
        return n, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    X = sm.add_constant(pd.DataFrame({sent_name: x2}))
    res = sm.OLS(y2, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_maxlags(n)})
    lo = res.params.index.get_loc(sent_name)
    return (
        n,
        float(res.rsquared),
        float(res.params.iloc[lo]),
        float(res.bse.iloc[lo]),
        float(res.tvalues.iloc[lo]),
        float(res.pvalues.iloc[lo]),
    )


def oos_mse(
    y: pd.Series,
    x: pd.Series,
    train_frac: float = 0.8,
) -> float:
    """OLS y ~ const + x; fit train, MSE on test."""
    m = y.notna() & x.notna()
    yv = y[m].to_numpy(float)
    xv = x[m].to_numpy(float)
    n = len(yv)
    if n < 40:
        return float("nan")
    cut = int(math.floor(n * train_frac))
    if cut < 25 or n - cut < 10:
        return float("nan")
    y_tr, x_tr = yv[:cut], xv[:cut]
    y_te, x_te = yv[cut:], xv[cut:]
    X_tr = np.column_stack([np.ones(len(x_tr)), x_tr])
    beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    pred = beta[0] + beta[1] * x_te
    return float(np.mean((y_te - pred) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=15, help="Rolling mean window (trading days)")
    ap.add_argument("--train-frac", type=float, default=0.8)
    args = ap.parse_args()

    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError:
        print("Install: pip install statsmodels", file=sys.stderr)
        sys.exit(1)

    w = args.window
    sent_specs = [
        ("nrc_net_sentiment", "NRC_net"),
        ("hf_net_4", "HF_net_4"),
        ("fb_net_sentiment", "FB_net"),
    ]

    rows = []
    for tkr in TICKERS:
        try:
            d = load_firm(tkr)
        except FileNotFoundError as e:
            print(f"Skip {tkr}: {e}")
            continue

        y = d["ret_1d"].shift(-1)

        for col, slab in sent_specs:
            if col not in d.columns:
                continue
            s = pd.to_numeric(d[col], errors="coerce")
            x_lag1 = s.shift(1)
            x_roll = s.rolling(w, min_periods=max(10, w // 2)).mean()

            for spec_name, x in (("lag1_prevday_sent", x_lag1), (f"roll{w}_mean_sent", x_roll)):
                n, r2, coef, se, tstat, pval = fit_hac(y, x, "sent")
                mse_oos = oos_mse(y, x, args.train_frac)
                rows.append(
                    {
                        "ticker": tkr,
                        "sentiment": slab,
                        "predictor": spec_name,
                        "window": w,
                        "n_insample": n,
                        "r2_insample": r2,
                        "coef": coef,
                        "se_hac": se,
                        "t_hac": tstat,
                        "pvalue": pval,
                        "mse_oos_last20pct": mse_oos,
                    }
                )

    if not rows:
        print("No rows â€” check panel paths.")
        return

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()

