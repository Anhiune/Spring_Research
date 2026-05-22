#!/usr/bin/env python3
"""Shared helpers for VAR / Granger scripts (ADF, lag caps, winsorization, multivariate causality)."""

from __future__ import annotations

from typing import Callable, TextIO

import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    """Clip to empirical quantiles (inclusive)."""
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def adf_report(series: pd.Series, name: str) -> dict:
    from statsmodels.tsa.stattools import adfuller

    x = series.dropna().astype(float)
    if len(x) < 20:
        return {"variable": name, "n": len(x), "error": "too_few_obs"}
    res = adfuller(x, autolag="AIC")
    return {
        "variable": name,
        "n": len(x),
        "adf_stat": res[0],
        "p_value": res[1],
        "used_lag": res[2],
        "stationary_5pct": res[1] < 0.05,
    }


def default_max_lag_cap(n_obs: int, override: int | None) -> int:
    if override is not None and override > 0:
        return int(override)
    return min(10, max(2, n_obs // 10))


def granger_block_bivariate(
    grangercausalitytests,
    y_name: str,
    x_name: str,
    arr: np.ndarray,
    gmax: int,
) -> pd.DataFrame:
    """arr columns [y, x]: H0 x does not Granger-cause y."""
    rows = []
    gc = grangercausalitytests(arr, maxlag=gmax, verbose=False)
    for lag in range(1, gmax + 1):
        tests = gc[lag][0]
        ftest = tests.get("ssr_ftest")
        if ftest is not None:
            rows.append(
                {
                    "direction": f"{x_name} -> {y_name}",
                    "lag": lag,
                    "test": "ssr_ftest",
                    "stat": ftest[0],
                    "p_value": ftest[1],
                }
            )
        lr = tests.get("lrtest")
        if lr is not None:
            rows.append(
                {
                    "direction": f"{x_name} -> {y_name}",
                    "lag": lag,
                    "test": "lrtest",
                    "stat": lr[0],
                    "p_value": lr[1],
                }
            )
    return pd.DataFrame(rows)


def log_adf_rows(
    panel: pd.DataFrame,
    cols: list[tuple[str, str]],
    log: Callable[[str], None],
) -> list[dict]:
    rows_out = []
    for col, label in cols:
        r = adf_report(panel[col], label)
        rows_out.append(r)
        if "error" in r:
            log(f"{label}: {r['error']}")
            continue
        log(
            f"{label}: ADF={r['adf_stat']:.4f}, p={r['p_value']:.4g}, "
            f"stationary@5%={r['stationary_5pct']}"
        )
    return rows_out


def weekly_aggregate(
    df: pd.DataFrame,
    date_col: str,
    compound_cols: list[str],
    mean_cols: list[str],
) -> pd.DataFrame:
    """
    End-of-week (FRI) aggregation: compound daily simple returns within each week for
    every column in compound_cols; arithmetic mean for mean_cols (e.g. sentiment).
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], format="mixed", utc=False)
    d = d.set_index(date_col).sort_index()
    weeks: list[dict] = []
    for _, wk in d.resample("W-FRI"):
        if wk.empty:
            continue
        row: dict = {date_col: wk.index.max().normalize()}
        ok = True
        for c in compound_cols:
            if c not in wk.columns:
                ok = False
                break
            r = wk[c].dropna().astype(float)
            if r.empty:
                ok = False
                break
            row[c] = float((1.0 + r).prod() - 1.0)
        if not ok:
            continue
        for c in mean_cols:
            if c not in wk.columns:
                ok = False
                break
            row[c] = wk[c].mean()
        if not ok:
            continue
        weeks.append(row)
    return pd.DataFrame(weeks).dropna(how="any")


def multivariate_var_causality(
    endog: pd.DataFrame,
    max_lag_cap: int,
    log: Callable[[str], None],
    ret_name: str,
    sent_name: str,
) -> tuple[object, int, pd.DataFrame]:
    """
    Fit VAR on endog columns (all numeric), BIC lag, Wald tests:
      sent -> ret and ret -> sent (conditional on full system).
    Returns (var_results, lag_bic, causality_summary_df).
    """
    from statsmodels.tsa.api import VAR

    data = endog.astype(float).dropna()
    if len(data) < 40:
        raise ValueError("Too few rows for multivariate VAR.")

    model = VAR(data)
    order_res = model.select_order(maxlags=max_lag_cap)
    lag_bic = int(order_res.selected_orders.get("bic", 1))
    if lag_bic <= 0:
        lag_bic = 1

    log(f"--- Multivariate VAR order selection (maxlags={max_lag_cap}) ---")
    log(str(order_res.summary()))
    log(f"Selected lag (BIC): {lag_bic}")
    log("")

    res = model.fit(lag_bic)
    log(f"--- Multivariate VAR fit: {list(endog.columns)}, lags={lag_bic} ---")
    log(res.summary().__str__())
    log("")

    rows = []
    t1 = res.test_causality(ret_name, [sent_name], kind="f")
    rows.append(
        {
            "hypothesis": f"{sent_name} -> {ret_name} (block Wald, conditional on system)",
            "test": "f",
            "p_value": float(t1.pvalue),
            "conclusion_5pct": "reject_H0" if t1.pvalue < 0.05 else "fail_to_reject",
        }
    )
    t2 = res.test_causality(sent_name, [ret_name], kind="f")
    rows.append(
        {
            "hypothesis": f"{ret_name} -> {sent_name} (block Wald, conditional on system)",
            "test": "f",
            "p_value": float(t2.pvalue),
            "conclusion_5pct": "reject_H0" if t2.pvalue < 0.05 else "fail_to_reject",
        }
    )
    log("--- Multivariate Granger / block causality (VAR Wald) ---")
    log(str(t1))
    log(str(t2))
    return res, lag_bic, pd.DataFrame(rows)


def write_report_footer(buf: TextIO, log: Callable[[str], None]) -> None:
    log("")
    log("Notes:")
    log("  - Granger / Wald tests are in-sample linear predictability, not structural causation.")
    log("  - Multivariate tests ask whether lags of one variable help predict another,")
    log("    given lags of all variables in the VAR.")
    log("  - Engagement-weighted sentiment (nrc_net_sentiment_ew) uses log1p(score/ups)")
    log("    and log1p(num_comments) when those columns exist in the Reddit export.")
