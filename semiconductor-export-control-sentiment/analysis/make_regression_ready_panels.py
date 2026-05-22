#!/usr/bin/env python3
"""
Build regression-ready CSVs (no keyword augmentation — complete cases only).

1) Firm tickers (NVDA, AMD, INTC):
   - *_complete.csv     : rows where subreddit-based sentiment exists (nrc_net_sentiment not NaN)
   - *_regression.csv    : ret_1d + nrc_net_sentiment_lag1 non-NaN (+ optional min_posts, benchmarks)
   - *_complete_hf.csv   : all five HF probability columns non-NaN
   - *_regression_hf.csv : ret_1d + hf_net_4_lag1 non-NaN (HF net = bull minus bear probs; see below)

2) Overall universe + pooled sentiment:
   - merged_universe_pooled_regression.csv : ew_ret_mean + nrc_net_sentiment_mean_lag1 non-NaN
   - merged_universe_pooled_regression_hf.csv : ew_ret_mean + hf_net_4_mean_lag1 non-NaN

HF net (daily aggregate, same definition on levels and on lag1 columns):
  hf_net_4 = (hf_very_positive + hf_positive) - (hf_very_negative + hf_negative)
  Pooled/universe: use *_mean and *_mean_lag1 columns.

Enrichment: merges benchmark_{SOXX,SPY,VIX}.csv daily returns as soxx_ret_1d, spy_ret_1d, vix_ret_1d
when those files exist under reddit_prepared/ (for multivariate VAR / controls).

Usage:
  python make_regression_ready_panels.py
  python make_regression_ready_panels.py --min-posts 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DEFAULT = SCRIPT_DIR / "reddit_prepared"
TICKERS = ("NVDA", "AMD", "INTC")

BENCHMARK_STEMS = ("SOXX", "SPY", "VIX")

HF_LEVEL_COLS = (
    "hf_neutral_score",
    "hf_negative_score",
    "hf_very_negative_score",
    "hf_positive_score",
    "hf_very_positive_score",
)

HF_LAG1_SUFFIX = (
    "hf_very_positive_score_lag1",
    "hf_positive_score_lag1",
    "hf_very_negative_score_lag1",
    "hf_negative_score_lag1",
)

HF_MEAN_LAG1_SUFFIX = (
    "hf_very_positive_score_mean_lag1",
    "hf_positive_score_mean_lag1",
    "hf_very_negative_score_mean_lag1",
    "hf_negative_score_mean_lag1",
)


def attach_benchmark_returns(df: pd.DataFrame, prep_dir: Path) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        return out
    out["date"] = pd.to_datetime(out["date"], format="mixed", utc=False).dt.normalize()
    for stem in BENCHMARK_STEMS:
        p = prep_dir / f"benchmark_{stem}.csv"
        if not p.exists():
            continue
        b = pd.read_csv(p)
        if "date" not in b.columns or "ret_1d" not in b.columns:
            continue
        b["date"] = pd.to_datetime(b["date"], format="mixed", utc=False).dt.normalize()
        col = f"{stem.lower()}_ret_1d"
        out = out.merge(b[["date", "ret_1d"]].rename(columns={"ret_1d": col}), on="date", how="left")
    return out


def _sumcols(df: pd.DataFrame, names: tuple[str, ...], out: str, signs: tuple[int, int, int, int]) -> pd.DataFrame:
    if not all(c in df.columns for c in names):
        return df
    o = df.copy()
    parts = [signs[i] * pd.to_numeric(o[names[i]], errors="coerce") for i in range(len(names))]
    o[out] = parts[0]
    for p in parts[1:]:
        o[out] = o[out] + p
    return o


def add_hf_net_4_lag1_firm(df: pd.DataFrame) -> pd.DataFrame:
    """hf_net_4_lag1 from lagged HF class probabilities (firm / stacked panels)."""
    return _sumcols(df, HF_LAG1_SUFFIX, "hf_net_4_lag1", (1, 1, -1, -1))


def add_hf_net_4_mean_lag1(df: pd.DataFrame) -> pd.DataFrame:
    """hf_net_4_mean_lag1 for pooled / universe daily means."""
    return _sumcols(df, HF_MEAN_LAG1_SUFFIX, "hf_net_4_mean_lag1", (1, 1, -1, -1))


def main() -> None:
    p = argparse.ArgumentParser(description="Regression-ready complete-case panels + benchmark columns.")
    p.add_argument(
        "--prep-dir",
        type=Path,
        default=PREP_DEFAULT,
        help="Folder with merged_*_market_sentiment.csv and benchmark_*.csv",
    )
    p.add_argument(
        "--universe-merged-in",
        type=Path,
        default=None,
        help="merged_universe_pooled_sentiment.csv path (default: <historical data>/merged_universe_pooled_sentiment.csv)",
    )
    p.add_argument(
        "--universe-regression-out",
        type=Path,
        default=None,
        help="Output for merged_universe_pooled_regression.csv",
    )
    p.add_argument(
        "--universe-regression-hf-out",
        type=Path,
        default=None,
        help="Output for merged_universe_pooled_regression_hf.csv",
    )
    p.add_argument(
        "--min-posts",
        type=int,
        default=1,
        help="Drop firm rows with n_posts below this (default 1 = keep all)",
    )
    args = p.parse_args()

    PREP = args.prep_dir.resolve()
    uni_in = (
        args.universe_merged_in
        if args.universe_merged_in is not None
        else SCRIPT_DIR / "merged_universe_pooled_sentiment.csv"
    )
    uni_out = (
        args.universe_regression_out
        if args.universe_regression_out is not None
        else SCRIPT_DIR / "merged_universe_pooled_regression.csv"
    )
    uni_hf_out = (
        args.universe_regression_hf_out
        if args.universe_regression_hf_out is not None
        else SCRIPT_DIR / "merged_universe_pooled_regression_hf.csv"
    )

    for t in TICKERS:
        src = PREP / f"merged_{t}_market_sentiment.csv"
        if not src.exists():
            raise SystemExit(f"Missing {src}")
        df = pd.read_csv(src)
        sent = "nrc_net_sentiment"
        if sent not in df.columns:
            raise SystemExit(f"No {sent} in {src}")

        complete = df[df[sent].notna()].copy()
        if "n_posts" in complete.columns and args.min_posts > 1:
            complete = complete[complete["n_posts"] >= args.min_posts].copy()

        out_c = PREP / f"merged_{t}_market_sentiment_complete.csv"
        complete = attach_benchmark_returns(complete, PREP)
        complete.to_csv(out_c, index=False)
        print(f"{out_c.name}: {len(complete)} rows (sentiment present)")

        lag = f"{sent}_lag1"
        reg = complete[complete["ret_1d"].notna() & complete[lag].notna()].copy()
        out_r = PREP / f"merged_{t}_market_sentiment_regression.csv"
        reg.to_csv(out_r, index=False)
        print(f"{out_r.name}: {len(reg)} rows (ret_1d + lag1 sentiment non-NaN)")

        ew_lag = "nrc_net_sentiment_ew_lag1"
        if ew_lag in reg.columns:
            reg_ew = reg[reg[ew_lag].notna()].copy()
            out_ew = PREP / f"merged_{t}_market_sentiment_regression_ew.csv"
            reg_ew.to_csv(out_ew, index=False)
            print(f"{out_ew.name}: {len(reg_ew)} rows (engagement-weighted sentiment lag1)")

        if all(c in df.columns for c in HF_LEVEL_COLS):
            hf_complete = df.loc[df[list(HF_LEVEL_COLS)].notna().all(axis=1)].copy()
            if "n_posts" in hf_complete.columns and args.min_posts > 1:
                hf_complete = hf_complete[hf_complete["n_posts"] >= args.min_posts].copy()
            hf_complete = attach_benchmark_returns(hf_complete, PREP)
            hf_complete = add_hf_net_4_lag1_firm(hf_complete)
            out_chf = PREP / f"merged_{t}_market_sentiment_complete_hf.csv"
            hf_complete.to_csv(out_chf, index=False)
            print(f"{out_chf.name}: {len(hf_complete)} rows (all HF probs present)")
            reg_hf = hf_complete[
                hf_complete["ret_1d"].notna() & hf_complete["hf_net_4_lag1"].notna()
            ].copy()
            out_rhf = PREP / f"merged_{t}_market_sentiment_regression_hf.csv"
            reg_hf.to_csv(out_rhf, index=False)
            print(f"{out_rhf.name}: {len(reg_hf)} rows (ret_1d + hf_net_4_lag1)")
        else:
            print(f"Skip HF panels for {t}: need columns {HF_LEVEL_COLS}")

    uni = uni_in
    if uni.exists():
        u = pd.read_csv(uni)
        u = u[u["ew_ret_mean"].notna() & u["nrc_net_sentiment_mean_lag1"].notna()].copy()
        u = attach_benchmark_returns(u, PREP)
        u.to_csv(uni_out, index=False)
        print(f"{uni_out.name}: {len(u)} rows")

        uh = pd.read_csv(uni)
        uh = add_hf_net_4_mean_lag1(uh)
        if "hf_net_4_mean_lag1" in uh.columns:
            uh = uh[uh["ew_ret_mean"].notna() & uh["hf_net_4_mean_lag1"].notna()].copy()
            uh = attach_benchmark_returns(uh, PREP)
            uh.to_csv(uni_hf_out, index=False)
            print(f"{uni_hf_out.name}: {len(uh)} rows (universe + HF net lag1)")
        else:
            print("Skip universe HF regression (missing HF mean lag columns)")
    else:
        print(f"Skip universe regression file (no {uni})")

    stack_paths = [PREP / f"merged_{t}_market_sentiment_regression_hf.csv" for t in TICKERS]
    if all(sp.exists() for sp in stack_paths):
        stacked = pd.concat([pd.read_csv(sp) for sp in stack_paths], ignore_index=True)
        stacked_out = PREP / "firm_merged_regression_NVDA_AMD_INTC_stacked.csv"
        stacked.to_csv(stacked_out, index=False)
        print(f"{stacked_out.name}: {len(stacked)} rows (stacked NVDA+AMD+INTC HF regression panel)")


if __name__ == "__main__":
    main()
