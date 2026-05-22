#!/usr/bin/env python3
"""
ARIMAX models for next-day stock returns using the Tabularis multilingual
sentiment feature already prepared in the HF panel files.

In this codebase, the Tabularis-derived net sentiment feature is stored as:
  hf_net_4_lag1

The dependent variable is:
  ret_1d

The model uses:
  - autoregressive / moving-average history in SARIMAX
  - exogenous regressors:
      hf_net_4_lag1
      soxx_ret_1d
      spy_ret_1d
      vix_ret_1d
      n_posts_lag1 (when available)

Usage:
  cd "historical data"
  python run_tabularis_arimax.py
  python run_tabularis_arimax.py --prep-dir bluesky_prepared --out-dir analysis_output_bluesky
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DEFAULT = SCRIPT_DIR / "reddit_prepared"
OUT_DIR_DEFAULT = SCRIPT_DIR / "analysis_output"
TICKERS = ("NVDA", "AMD", "INTC")


def load_panel(prep_dir: Path, ticker: str) -> pd.DataFrame:
    path = prep_dir / f"merged_{ticker}_market_sentiment_regression_hf.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def exog_columns(df: pd.DataFrame) -> list[str]:
    cols = ["hf_net_4_lag1", "soxx_ret_1d", "spy_ret_1d", "vix_ret_1d"]
    if "n_posts_lag1" in df.columns:
        cols.append("n_posts_lag1")
    return cols


def fit_best_arimax(df: pd.DataFrame, ticker: str) -> tuple[object, dict]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    exog_cols = exog_columns(df)
    keep = ["date", "ret_1d"] + exog_cols
    work = df[keep].copy()
    for col in ["ret_1d"] + exog_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna().reset_index(drop=True)

    if len(work) < 80:
        raise ValueError(f"{ticker}: too few complete rows for ARIMAX ({len(work)})")

    y = work["ret_1d"].astype(float)
    x = work[exog_cols].astype(float)

    best_res = None
    best_meta = None
    orders = [(p, 0, q) for p in (1, 2, 3) for q in (0, 1)]

    for order in orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y,
                    exog=x,
                    order=order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, cov_type="robust")
        except Exception:
            continue

        if not np.isfinite(res.aic):
            continue

        meta = {
            "ticker": ticker,
            "n": int(len(work)),
            "order": str(order),
            "aic": float(res.aic),
            "bic": float(res.bic) if np.isfinite(res.bic) else float("nan"),
            "exog_cols": ",".join(exog_cols),
        }
        if best_res is None or meta["aic"] < best_meta["aic"]:
            best_res = res
            best_meta = meta

    if best_res is None or best_meta is None:
        raise RuntimeError(f"{ticker}: no ARIMAX model converged")

    return best_res, best_meta


def summarize_result(result: object, meta: dict) -> dict:
    row = dict(meta)
    params = result.params
    pvals = result.pvalues
    ses = result.bse

    def add_term(term: str, label: str) -> None:
        row[f"{label}_coef"] = float(params.get(term, np.nan))
        row[f"{label}_se"] = float(ses.get(term, np.nan))
        row[f"{label}_pvalue"] = float(pvals.get(term, np.nan))

    add_term("hf_net_4_lag1", "tabularis")
    add_term("soxx_ret_1d", "soxx")
    add_term("spy_ret_1d", "spy")
    add_term("vix_ret_1d", "vix")
    if "n_posts_lag1" in meta["exog_cols"]:
        add_term("n_posts_lag1", "n_posts")

    for name in params.index:
        if name.startswith("ar.L"):
            row[name.replace(".", "_") + "_coef"] = float(params[name])
            row[name.replace(".", "_") + "_pvalue"] = float(pvals.get(name, np.nan))
        if name.startswith("ma.L"):
            row[name.replace(".", "_") + "_coef"] = float(params[name])
            row[name.replace(".", "_") + "_pvalue"] = float(pvals.get(name, np.nan))

    return row


def write_text_summary(out_path: Path, ticker: str, result: object, meta: dict) -> None:
    lines = [
        f"Ticker: {ticker}",
        f"ARIMAX order: {meta['order']}",
        f"N complete rows: {meta['n']}",
        f"AIC: {meta['aic']:.4f}",
        f"BIC: {meta['bic']:.4f}" if np.isfinite(meta["bic"]) else "BIC: nan",
        f"Exogenous variables: {meta['exog_cols']}",
        "",
        str(result.summary()),
        "",
        "Notes:",
        "  - ret_1d is the dependent variable.",
        "  - hf_net_4_lag1 is the Tabularis multilingual sentiment feature used in this project.",
        "  - AR / MA terms capture historical return dynamics directly inside the model.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIMAX for returns with Tabularis sentiment and market controls.")
    parser.add_argument("--prep-dir", type=Path, default=PREP_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--tickers", nargs="*", default=list(TICKERS))
    args = parser.parse_args()

    prep_dir = args.prep_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ticker in args.tickers:
        tkr = ticker.upper()
        df = load_panel(prep_dir, tkr)
        result, meta = fit_best_arimax(df, tkr)
        rows.append(summarize_result(result, meta))
        write_text_summary(out_dir / f"arimax_{tkr}_tabularis_report.txt", tkr, result, meta)
        print(
            f"{tkr}: order={meta['order']} n={meta['n']} "
            f"Tabularis coef={result.params.get('hf_net_4_lag1', np.nan):.6f} "
            f"p={result.pvalues.get('hf_net_4_lag1', np.nan):.4g}"
        )

    out_csv = out_dir / "tabularis_arimax_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
