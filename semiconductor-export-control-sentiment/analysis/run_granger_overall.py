#!/usr/bin/env python3
"""
Overall panel: equal-weight universe return vs pooled Reddit sentiment (NRC net mean).

Same enhancements as run_granger_firm.py: winsorized bivariate, optional weekly,
multivariate VAR with SOXX/SPY/VIX when columns exist.

Input : merged_universe_pooled_regression.csv
Output: analysis_output/granger_overall_report.txt, granger_overall_summary.csv, granger_overall_adf.csv

Usage:
  python run_granger_overall.py
  python run_granger_overall.py --winsorize 0.01 0.99 --max-lag 12
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

from granger_utils import (
    default_max_lag_cap,
    granger_block_bivariate,
    log_adf_rows,
    multivariate_var_causality,
    weekly_aggregate,
    winsorize_series,
    write_report_footer,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR_DEFAULT = SCRIPT_DIR / "analysis_output"
DEFAULT_DATA = SCRIPT_DIR / "merged_universe_pooled_regression.csv"


def run_bivariate_section(
    panel: pd.DataFrame,
    ret_name: str,
    sent_name: str,
    section_title: str,
    grangercausalitytests,
    max_lag_override: int | None,
    log,
) -> tuple[pd.DataFrame, list[dict], int]:
    from statsmodels.tsa.api import VAR

    data = panel[[ret_name, sent_name]].astype(float).dropna()
    T = len(data)
    if T < 40:
        log(f"Skip {section_title}: too few obs ({T}).")
        return pd.DataFrame(), [], 1

    lag_cap = default_max_lag_cap(T, max_lag_override)
    arr = data.values
    model = VAR(arr)
    order_res = model.select_order(maxlags=lag_cap)
    lag_bic = int(order_res.selected_orders.get("bic", 1))
    if lag_bic <= 0:
        lag_bic = 1

    log(f"=== {section_title} ===")
    log(f"N observations: {T}")
    log("")
    log("--- Augmented Dickey-Fuller (H0: unit root) ---")
    adf_rows = log_adf_rows(
        data.rename(columns={ret_name: "ret", sent_name: "sent"}),
        [("ret", ret_name), ("sent", sent_name)],
        log,
    )
    for r in adf_rows:
        r["section"] = section_title
    log("")
    log(f"--- VAR order selection (maxlags={lag_cap}) ---")
    log(str(order_res.summary()))
    log(f"Selected lag (BIC): {lag_bic}")
    log("")

    gmax = min(max(lag_bic, 2), lag_cap)
    log(f"--- Bivariate Granger (max lag = {gmax}) ---")
    g1 = granger_block_bivariate(
        grangercausalitytests,
        ret_name,
        sent_name,
        data[[ret_name, sent_name]].values,
        gmax,
    )
    log(f"{sent_name} -> {ret_name}:")
    for lag in range(1, gmax + 1):
        sub = g1[(g1["lag"] == lag) & (g1["test"] == "ssr_ftest")]
        if not sub.empty:
            log(f"  lag {lag}: F p-value = {sub['p_value'].iloc[0]:.4g}")
    log("")
    g2 = granger_block_bivariate(
        grangercausalitytests,
        sent_name,
        ret_name,
        data[[sent_name, ret_name]].values,
        gmax,
    )
    log(f"{ret_name} -> {sent_name}:")
    for lag in range(1, gmax + 1):
        sub = g2[(g2["lag"] == lag) & (g2["test"] == "ssr_ftest")]
        if not sub.empty:
            log(f"  lag {lag}: F p-value = {sub['p_value'].iloc[0]:.4g}")
    log("")
    log(f"--- VAR fit, lags = {lag_bic} ---")
    res = model.fit(lag_bic)
    log(res.summary().__str__())
    log("")

    g1["section"] = section_title
    g2["section"] = section_title
    return pd.concat([g1, g2], ignore_index=True), adf_rows, lag_bic


def main() -> None:
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        print("Install: pip install statsmodels", file=sys.stderr)
        raise SystemExit(1)

    p = argparse.ArgumentParser(description="Universe Granger / VAR with enhancements.")
    p.add_argument("--input", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--max-lag", type=int, default=0, help="0 = auto")
    p.add_argument(
        "--winsorize",
        type=float,
        nargs=2,
        metavar=("LOW_Q", "HIGH_Q"),
        default=None,
    )
    p.add_argument("--weekly", action="store_true")
    p.add_argument("--skip-multivariate", action="store_true")
    p.add_argument("--skip-winsorize", action="store_true")
    args = p.parse_args()

    OUT_DIR = args.out_dir.resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = args.input
    if not path.exists():
        raise SystemExit(f"Missing {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False)

    ret_col = "ew_ret_mean"
    sent_col = "nrc_net_sentiment_mean"
    if ret_col not in df.columns or sent_col not in df.columns:
        raise SystemExit(f"Need columns {ret_col!r} and {sent_col!r}")

    base = df[[ret_col, sent_col, "date"]].copy()
    if args.weekly:
        compound = [ret_col] + [c for c in ("soxx_ret_1d", "spy_ret_1d", "vix_ret_1d") if c in df.columns]
        cols = ["date"] + list(dict.fromkeys(compound + [sent_col]))
        wk = df[[c for c in cols if c in df.columns]].copy()
        base = weekly_aggregate(wk, "date", compound, [sent_col])
        base["date"] = pd.to_datetime(base["date"], format="mixed", utc=False)

    buf = io.StringIO()

    def log(msg: str = "") -> None:
        print(msg)
        buf.write(msg + "\n")

    max_lag_arg = args.max_lag if args.max_lag > 0 else None

    log("=== Overall: enhanced Granger / VAR ===")
    log(f"Return: {ret_col}; sentiment: {sent_col}")
    log("Reference: Rodriguez-Ibanez et al. (2023), ESWA.")
    log(f"Date range: {base['date'].min()} .. {base['date'].max()}")
    log("")

    summaries: list[pd.DataFrame] = []
    adf_all: list[dict] = []

    s1, a1, _ = run_bivariate_section(
        base[[ret_col, sent_col]],
        ret_col,
        sent_col,
        "Overall baseline (bivariate)",
        grangercausalitytests,
        max_lag_arg,
        log,
    )
    if not s1.empty:
        summaries.append(s1)
        adf_all.extend(a1)

    if (
        args.winsorize is not None
        and not args.skip_winsorize
    ):
        wdf = base[[ret_col, sent_col, "date"]].copy()
        wdf[sent_col] = winsorize_series(wdf[sent_col].astype(float), args.winsorize[0], args.winsorize[1])
        s2, a2, _ = run_bivariate_section(
            wdf[[ret_col, sent_col]],
            ret_col,
            sent_col,
            f"Overall winsorized sentiment ({args.winsorize[0]:.3f}, {args.winsorize[1]:.3f})",
            grangercausalitytests,
            max_lag_arg,
            log,
        )
        if not s2.empty:
            summaries.append(s2)
            adf_all.extend(a2)

    if not args.skip_multivariate:
        ctrl = [c for c in ("soxx_ret_1d", "spy_ret_1d", "vix_ret_1d") if c in df.columns]
        if ctrl:
            cols = [ret_col, sent_col] + ctrl
            sub = df[cols].astype(float).dropna()
            if len(sub) >= 60:
                try:
                    endog = sub.rename(
                        columns={
                            ret_col: "ewret",
                            sent_col: "sent",
                            "soxx_ret_1d": "soxx",
                            "spy_ret_1d": "spy",
                            "vix_ret_1d": "vix",
                        }
                    )
                    log("=== Overall multivariate VAR (+ benchmarks) ===")
                    log(f"N complete rows: {len(endog)}")
                    log("")
                    m_adf = log_adf_rows(endog, [(c, c) for c in endog.columns], log)
                    for r in m_adf:
                        r["section"] = "overall_multivariate"
                    adf_all.extend(m_adf)
                    log("")
                    _, _, ctab = multivariate_var_causality(
                        endog,
                        default_max_lag_cap(len(endog), max_lag_arg),
                        log,
                        "ewret",
                        "sent",
                    )
                    ctab["section"] = "overall_multivariate"
                    summaries.append(ctab)
                except Exception as ex:  # noqa: BLE001
                    log(f"Multivariate block failed: {ex}")
            else:
                log(f"Skip multivariate: only {len(sub)} complete rows.")
        else:
            log("Skip multivariate: add benchmarks via make_regression_ready_panels.py.")

    write_report_footer(buf, log)

    report_path = OUT_DIR / "granger_overall_report.txt"
    report_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nWrote {report_path}")

    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            OUT_DIR / "granger_overall_summary.csv", index=False
        )
        print(f"Wrote {OUT_DIR / 'granger_overall_summary.csv'}")
    if adf_all:
        pd.DataFrame(adf_all).to_csv(OUT_DIR / "granger_overall_adf.csv", index=False)


if __name__ == "__main__":
    main()
