#!/usr/bin/env python3
"""
Firm panel (e.g. NVDA): stock return vs company-subreddit NRC sentiment.

Enhancements beyond baseline bivariate VAR/Granger:
  - Engagement-weighted sentiment bivariate block (if nrc_net_sentiment_ew non-missing)
  - Winsorized sentiment (optional quantiles)
  - Multivariate VAR with SOXX / SPY / VIX daily returns when merged (block Wald causality)
  - Optional weekly frequency (Friday week-ends; return compounded within week)

Input : reddit_prepared/merged_{TICKER}_market_sentiment_regression.csv
Output: analysis_output/granger_{TICKER}_report.txt, granger_{TICKER}_summary.csv, granger_{TICKER}_adf.csv

Usage:
  python run_granger_firm.py --ticker NVDA
  python run_granger_firm.py --ticker AMD --max-lag 12 --winsorize 0.01 0.99
  python run_granger_firm.py --ticker NVDA --weekly --skip-multivariate
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

from granger_utils import (
    adf_report,
    default_max_lag_cap,
    granger_block_bivariate,
    log_adf_rows,
    multivariate_var_causality,
    weekly_aggregate,
    winsorize_series,
    write_report_footer,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DEFAULT = SCRIPT_DIR / "reddit_prepared"
OUT_DIR_DEFAULT = SCRIPT_DIR / "analysis_output"


def run_bivariate_section(
    panel: pd.DataFrame,
    ret_name: str,
    sent_name: str,
    ticker: str,
    section_title: str,
    grangercausalitytests,
    max_lag_cap: int | None,
    log,
) -> tuple[pd.DataFrame, list[dict], int]:
    """VAR select_order, Granger both directions, fit VAR(BIC). Returns (summary_df, adf_rows, lag_bic)."""
    from statsmodels.tsa.api import VAR

    data = panel[[ret_name, sent_name]].astype(float).dropna()
    T = len(data)
    if T < 40:
        log(f"Skip {section_title}: too few obs ({T}).")
        return pd.DataFrame(), [], 1

    lag_cap = default_max_lag_cap(T, max_lag_cap if max_lag_cap else None)
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
        [("ret", f"{ticker}_ret"), ("sent", f"{ticker}_{sent_name}")],
        log,
    )
    log("")
    log(f"--- VAR order selection (maxlags={lag_cap}) ---")
    log(str(order_res.summary()))
    log(f"Selected lag (BIC): {lag_bic}")
    log("")

    gmax = min(max(lag_bic, 2), lag_cap)
    log(f"--- Bivariate Granger (H0: col2 does not GC col1), max lag = {gmax} ---")
    g1 = granger_block_bivariate(
        grangercausalitytests,
        "ret_1d",
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
        "ret_1d",
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

    for r in adf_rows:
        r["section"] = section_title
    g1["section"] = section_title
    g2["section"] = section_title
    return pd.concat([g1, g2], ignore_index=True), adf_rows, lag_bic


def run_one(args: argparse.Namespace) -> None:
    from statsmodels.tsa.stattools import grangercausalitytests

    OUT_DIR = args.out_dir.resolve()
    PREP = args.prep_dir.resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ticker = args.ticker.strip().upper()
    path = Path(args.input) if args.input else PREP / f"merged_{ticker}_market_sentiment_regression.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=False)

    ret_col = "ret_1d"
    sent_col = args.sentiment_col
    if ret_col not in df.columns:
        raise SystemExit(f"Need column {ret_col!r}")
    if sent_col not in df.columns:
        raise SystemExit(f"Need column {sent_col!r} (try default nrc_net_sentiment or nrc_net_sentiment_ew)")

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

    log(f"=== {ticker}: enhanced Granger / VAR ===")
    log(f"Primary sentiment column: {sent_col}")
    log("Reference: Rodriguez-Ibanez et al. (2023), ESWA (temporal / predictive framing).")
    log(f"Date range: {base['date'].min()} .. {base['date'].max()}")
    log("")

    all_summaries: list[pd.DataFrame] = []
    all_adf: list[dict] = []

    max_lag_arg = args.max_lag if args.max_lag > 0 else None

    # --- Baseline bivariate ---
    s1, a1, _ = run_bivariate_section(
        base[[ret_col, sent_col]],
        ret_col,
        sent_col,
        ticker,
        f"{ticker} baseline (bivariate)",
        grangercausalitytests,
        max_lag_arg,
        log,
    )
    if not s1.empty:
        all_summaries.append(s1)
        all_adf.extend(a1)

    # --- Engagement-weighted bivariate (same file, non-null ew) ---
    if not args.skip_ew and sent_col == "nrc_net_sentiment" and "nrc_net_sentiment_ew" in df.columns:
        ew = df[[ret_col, "nrc_net_sentiment_ew", "date"]].dropna()
        if len(ew) >= 40:
            s2, a2, _ = run_bivariate_section(
                ew,
                ret_col,
                "nrc_net_sentiment_ew",
                ticker,
                f"{ticker} engagement-weighted NRC net (bivariate)",
                grangercausalitytests,
                max_lag_arg,
                log,
            )
            if not s2.empty:
                all_summaries.append(s2)
                all_adf.extend(a2)

    # --- Winsorized sentiment ---
    if args.winsorize is not None and not args.skip_winsorize:
        lo, hi = args.winsorize
        wdf = base[[ret_col, sent_col, "date"]].copy()
        wdf[sent_col] = winsorize_series(wdf[sent_col].astype(float), lo, hi)
        s3, a3, _ = run_bivariate_section(
            wdf,
            ret_col,
            sent_col,
            ticker,
            f"{ticker} winsorized sentiment ({lo:.3f}, {hi:.3f}) (bivariate)",
            grangercausalitytests,
            max_lag_arg,
            log,
        )
        if not s3.empty:
            all_summaries.append(s3)
            all_adf.extend(a3)

    # --- Multivariate VAR ---
    if not args.skip_multivariate:
        ctrl = [c for c in ("soxx_ret_1d", "spy_ret_1d", "vix_ret_1d") if c in df.columns]
        if ctrl:
            cols = [ret_col, sent_col] + ctrl
            sub = df[cols].astype(float).dropna()
            if len(sub) >= 60:
                try:
                    endog = sub.rename(
                        columns={
                            ret_col: "ret",
                            sent_col: "sent",
                            "soxx_ret_1d": "soxx",
                            "spy_ret_1d": "spy",
                            "vix_ret_1d": "vix",
                        }
                    )
                    log(f"=== {ticker} multivariate VAR (+ {', '.join(ctrl)}) ===")
                    log(f"N observations (complete on all series): {len(endog)}")
                    log("")
                    m_adf = log_adf_rows(
                        endog,
                        [(c, c) for c in endog.columns],
                        log,
                    )
                    for r in m_adf:
                        r["section"] = f"{ticker}_multivariate"
                    all_adf.extend(m_adf)
                    log("")
                    _, _, ctab = multivariate_var_causality(
                        endog,
                        default_max_lag_cap(len(endog), max_lag_arg),
                        log,
                        "ret",
                        "sent",
                    )
                    ctab["section"] = f"{ticker}_multivariate"
                    all_summaries.append(ctab)
                except Exception as ex:  # noqa: BLE001
                    log(f"Multivariate block failed: {ex}")
            else:
                log(f"Skip multivariate: only {len(sub)} complete rows (need >= 60).")
        else:
            log("Skip multivariate: no soxx_ret_1d / spy_ret_1d / vix_ret_1d in panel (run make_regression_ready_panels.py after benchmarks exist).")

    write_report_footer(buf, log)

    report_path = OUT_DIR / f"granger_{ticker}_report.txt"
    report_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nWrote {report_path}")

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            OUT_DIR / f"granger_{ticker}_summary.csv", index=False
        )
        print(f"Wrote {OUT_DIR / f'granger_{ticker}_summary.csv'}")
    if all_adf:
        pd.DataFrame(all_adf).to_csv(OUT_DIR / f"granger_{ticker}_adf.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Firm-level Granger / VAR with enhancements.")
    p.add_argument("--prep-dir", type=Path, default=PREP_DEFAULT, help="merged_* panels folder")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT, help="granger_* outputs")
    p.add_argument("--ticker", default="NVDA")
    p.add_argument(
        "--sentiment-col",
        default="nrc_net_sentiment",
        help="Sentiment series (default nrc_net_sentiment; use nrc_net_sentiment_ew with EW regression file)",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override path to merged_*_market_sentiment_regression.csv",
    )
    p.add_argument("--max-lag", type=int, default=0, help="Cap for VAR maxlags (0 = auto T//10)")
    p.add_argument(
        "--winsorize",
        type=float,
        nargs=2,
        metavar=("LOW_Q", "HIGH_Q"),
        default=None,
        help="e.g. 0.01 0.99 for sentiment winsorization before bivariate block",
    )
    p.add_argument("--weekly", action="store_true", help="Friday week-end aggregation before tests")
    p.add_argument("--skip-multivariate", action="store_true")
    p.add_argument("--skip-ew", action="store_true", help="Skip engagement-weighted bivariate block")
    p.add_argument("--skip-winsorize", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run_one(args)
    except ImportError:
        print("Install: pip install statsmodels", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
