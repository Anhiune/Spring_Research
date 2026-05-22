#!/usr/bin/env python3
"""
Tier-1 HF regressions: OLS with Newey–West HAC on daily panels.

Primary sentiment: hf_net_4_lag1 = (very_pos + pos - very_neg - neg) on lag-1 HF
probabilities (same construction as make_regression_ready_panels).

Specs per dataset:
  A) ret ~ const + hf_net_4_lag1
  B) ret ~ const + hf_net_4_lag1 + soxx_ret_1d + spy_ret_1d + vix_ret_1d (drop Missing)

Inputs: run make_regression_ready_panels.py first (creates *_regression_hf.csv and universe *_hf).

Output: analysis_output/hf_ols_tier1_results.csv

Usage:
  cd "historical data"
  python run_hf_ols_tier1.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PREP_DEFAULT = SCRIPT_DIR / "reddit_prepared"
OUT_DIR_DEFAULT = SCRIPT_DIR / "analysis_output"
BENCHMARK_STEMS = ("SOXX", "SPY", "VIX")


def nw_maxlags(n: int) -> int:
    if n <= 1:
        return 1
    return int(min(20, max(1, math.floor(4 * (n / 100) ** (2 / 9)))))


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


def add_hf_net_pooled(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        "hf_very_positive_score_mean_lag1",
        "hf_positive_score_mean_lag1",
        "hf_very_negative_score_mean_lag1",
        "hf_negative_score_mean_lag1",
    )
    if not all(c in df.columns for c in cols):
        return df
    o = df.copy()
    o["hf_net_4_lag1"] = (
        pd.to_numeric(o[cols[0]], errors="coerce")
        + pd.to_numeric(o[cols[1]], errors="coerce")
        - pd.to_numeric(o[cols[2]], errors="coerce")
        - pd.to_numeric(o[cols[3]], errors="coerce")
    )
    return o


def fit_row(
    dataset: str,
    spec: str,
    y: pd.Series,
    x: pd.DataFrame,
    sent_col: str,
) -> dict:
    import numpy as np
    import statsmodels.api as sm

    m = np.isfinite(y.to_numpy(float))
    for c in x.columns:
        m &= np.isfinite(x[c].to_numpy(float))
    y2 = y.loc[m]
    x2 = x.loc[m].astype(float)
    n = int(y2.shape[0])
    if n < 30:
        return {
            "dataset": dataset,
            "spec": spec,
            "n": n,
            "r2": float("nan"),
            "coef_sentiment": float("nan"),
            "se_sentiment": float("nan"),
            "t_sentiment": float("nan"),
            "pvalue_sentiment": float("nan"),
            "hac_maxlags": nw_maxlags(n),
            "note": "n<30 skip",
        }
    x_const = sm.add_constant(x2, has_constant="add")
    res = sm.OLS(y2.astype(float), x_const).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_maxlags(n)}
    )
    lo = res.params.index.get_loc(sent_col)
    return {
        "dataset": dataset,
        "spec": spec,
        "n": n,
        "r2": float(res.rsquared),
        "coef_sentiment": float(res.params.iloc[lo]),
        "se_sentiment": float(res.bse.iloc[lo]),
        "t_sentiment": float(res.tvalues.iloc[lo]),
        "pvalue_sentiment": float(res.pvalues.iloc[lo]),
        "hac_maxlags": nw_maxlags(n),
        "note": "",
    }


def run_firm(path: Path, label: str, rows: list) -> None:
    df = pd.read_csv(path)
    if "hf_net_4_lag1" not in df.columns or "ret_1d" not in df.columns:
        rows.append(
            {
                "dataset": label,
                "spec": "missing_cols",
                "n": 0,
                "r2": float("nan"),
                "coef_sentiment": float("nan"),
                "se_sentiment": float("nan"),
                "t_sentiment": float("nan"),
                "pvalue_sentiment": float("nan"),
                "hac_maxlags": 0,
                "note": "hf_net_4_lag1 or ret_1d missing",
            }
        )
        return
    ctrl = ["soxx_ret_1d", "spy_ret_1d", "vix_ret_1d"]
    rows.append(
        fit_row(
            label,
            "A_bivariate",
            df["ret_1d"],
            df[["hf_net_4_lag1"]],
            "hf_net_4_lag1",
        )
    )
    have = all(c in df.columns for c in ctrl)
    if have:
        dx = df[["hf_net_4_lag1"] + ctrl].copy()
        rows.append(
            fit_row(
                label,
                "B_plus_soxx_spy_vix",
                df["ret_1d"],
                dx,
                "hf_net_4_lag1",
            )
        )
    else:
        rows.append(
            {
                "dataset": label,
                "spec": "B_plus_soxx_spy_vix",
                "n": 0,
                "r2": float("nan"),
                "coef_sentiment": float("nan"),
                "se_sentiment": float("nan"),
                "t_sentiment": float("nan"),
                "pvalue_sentiment": float("nan"),
                "hac_maxlags": 0,
                "note": "benchmark columns missing",
            }
        )


def run_universe(path: Path, rows: list) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "hf_net_4_mean_lag1" not in df.columns:
        return
    y = df["ew_ret_mean"]
    rows.append(
        fit_row(
            "universe_pooled_hf",
            "A_bivariate",
            y,
            df[["hf_net_4_mean_lag1"]].rename(columns={"hf_net_4_mean_lag1": "hf_net_4_lag1"}),
            "hf_net_4_lag1",
        )
    )
    ctrl = ["soxx_ret_1d", "spy_ret_1d", "vix_ret_1d"]
    if all(c in df.columns for c in ctrl):
        z = df[["hf_net_4_mean_lag1"] + ctrl].rename(
            columns={"hf_net_4_mean_lag1": "hf_net_4_lag1"}
        )
        rows.append(
            fit_row(
                "universe_pooled_hf",
                "B_plus_soxx_spy_vix",
                y,
                z,
                "hf_net_4_lag1",
            )
        )


def run_pooled(stem: str, rows: list, prep: Path) -> None:
    path = prep / f"merged_pooled_{stem}.csv"
    if not path.exists():
        return
    df = attach_benchmark_returns(pd.read_csv(path), prep)
    df = add_hf_net_pooled(df)
    if "hf_net_4_lag1" not in df.columns or "ret_1d" not in df.columns:
        return
    label = f"pooled_broad_x_{stem}_return"
    rows.append(
        fit_row(
            label,
            "A_bivariate",
            df["ret_1d"],
            df[["hf_net_4_lag1"]],
            "hf_net_4_lag1",
        )
    )
    # Avoid using the same benchmark as both y and x control (perfect collinearity).
    ctrl = ["soxx_ret_1d", "spy_ret_1d", "vix_ret_1d"]
    if stem == "SOXX":
        ctrl = [c for c in ctrl if c != "soxx_ret_1d"]
    elif stem == "SPY":
        ctrl = [c for c in ctrl if c != "spy_ret_1d"]
    spec_b_name = "B_plus_other_benchmark_controls"
    if all(c in df.columns for c in ctrl) and ctrl:
        rows.append(
            fit_row(
                label,
                spec_b_name,
                df["ret_1d"],
                df[["hf_net_4_lag1"] + ctrl],
                "hf_net_4_lag1",
            )
        )
    else:
        rows.append(
            {
                "dataset": label,
                "spec": spec_b_name,
                "n": 0,
                "r2": float("nan"),
                "coef_sentiment": float("nan"),
                "se_sentiment": float("nan"),
                "t_sentiment": float("nan"),
                "pvalue_sentiment": float("nan"),
                "hac_maxlags": 0,
                "note": "benchmark columns missing",
            }
        )


def run_stacked(path: Path, rows: list) -> None:
    if not path.exists():
        return
    try:
        from patsy import dmatrices
    except ImportError:
        rows.append(
            {
                "dataset": "stacked_NVDA_AMD_INTC",
                "spec": "patsy",
                "n": 0,
                "r2": float("nan"),
                "coef_sentiment": float("nan"),
                "se_sentiment": float("nan"),
                "t_sentiment": float("nan"),
                "pvalue_sentiment": float("nan"),
                "hac_maxlags": 0,
                "note": "pip install patsy",
            }
        )
        return
    import statsmodels.api as sm

    df = pd.read_csv(path)
    hf_cols = (
        "hf_neutral_score",
        "hf_negative_score",
        "hf_very_negative_score",
        "hf_positive_score",
        "hf_very_positive_score",
    )
    if not all(c in df.columns for c in hf_cols):
        return
    w = df.loc[df[list(hf_cols)].notna().all(axis=1)].copy()
    w["hf_net_4_lag1"] = (
        pd.to_numeric(w["hf_very_positive_score_lag1"], errors="coerce")
        + pd.to_numeric(w["hf_positive_score_lag1"], errors="coerce")
        - pd.to_numeric(w["hf_very_negative_score_lag1"], errors="coerce")
        - pd.to_numeric(w["hf_negative_score_lag1"], errors="coerce")
    )
    w = w[w["ret_1d"].notna() & w["hf_net_4_lag1"].notna()].copy()
    if len(w) < 30:
        return
    w["ticker"] = w["ticker"].astype("category")

    y, X = dmatrices(
        "ret_1d ~ hf_net_4_lag1 + C(ticker)", w, return_type="dataframe"
    )
    nl = nw_maxlags(len(y))
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nl})
    lo = res.params.index.get_loc("hf_net_4_lag1")
    rows.append(
        {
            "dataset": "stacked_NVDA_AMD_INTC",
            "spec": "A_bivariate_plus_ticker_dummies",
            "n": int(len(y)),
            "r2": float(res.rsquared),
            "coef_sentiment": float(res.params.iloc[lo]),
            "se_sentiment": float(res.bse.iloc[lo]),
            "t_sentiment": float(res.tvalues.iloc[lo]),
            "pvalue_sentiment": float(res.pvalues.iloc[lo]),
            "hac_maxlags": nl,
            "note": "HAC on sorted rows; panel clustering optional follow-up",
        }
    )

    if all(c in w.columns for c in ["soxx_ret_1d", "spy_ret_1d", "vix_ret_1d"]):
        w2 = w.dropna(subset=["soxx_ret_1d", "spy_ret_1d", "vix_ret_1d"])
        if len(w2) < 30:
            return
        y2, X2 = dmatrices(
            "ret_1d ~ hf_net_4_lag1 + C(ticker) + soxx_ret_1d + spy_ret_1d + vix_ret_1d",
            w2,
            return_type="dataframe",
        )
        nl2 = nw_maxlags(len(y2))
        res2 = sm.OLS(y2, X2).fit(cov_type="HAC", cov_kwds={"maxlags": nl2})
        lo2 = res2.params.index.get_loc("hf_net_4_lag1")
        rows.append(
            {
                "dataset": "stacked_NVDA_AMD_INTC",
                "spec": "B_plus_controls_and_ticker_dummies",
                "n": int(len(y2)),
                "r2": float(res2.rsquared),
                "coef_sentiment": float(res2.params.iloc[lo2]),
                "se_sentiment": float(res2.bse.iloc[lo2]),
                "t_sentiment": float(res2.tvalues.iloc[lo2]),
                "pvalue_sentiment": float(res2.pvalues.iloc[lo2]),
                "hac_maxlags": nl2,
                "note": "",
            }
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-1 HAC OLS for HuggingFace net sentiment.")
    p.add_argument(
        "--prep-dir",
        type=Path,
        default=PREP_DEFAULT,
        help="Folder with merged_*_market_sentiment_regression_hf.csv and merged_pooled_*.csv",
    )
    p.add_argument(
        "--universe-hf",
        type=Path,
        default=None,
        help="merged_universe_pooled_regression_hf.csv (default: beside prep-dir parent)",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write results CSV here (default: analysis_output/hf_ols_tier1_results.csv)",
    )
    p.add_argument(
        "--stacked",
        type=Path,
        default=None,
        help="Stacked panel CSV (default: <prep-dir>/firm_merged_regression_NVDA_AMD_INTC_stacked.csv if exists)",
    )
    args = p.parse_args()

    PREP = args.prep_dir.resolve()
    OUT_DIR = args.out_csv.parent if args.out_csv else OUT_DIR_DEFAULT
    out_csv = args.out_csv if args.out_csv is not None else OUT_DIR / "hf_ols_tier1_results.csv"
    uni_hf = (
        args.universe_hf
        if args.universe_hf is not None
        else SCRIPT_DIR / "merged_universe_pooled_regression_hf.csv"
    )
    stacked_path = args.stacked
    if stacked_path is None:
        cand = PREP / "firm_merged_regression_NVDA_AMD_INTC_stacked.csv"
        stacked_path = (
            cand
            if cand.exists()
            else SCRIPT_DIR / "merged_historical_data_share" / "firm_merged_regression_NVDA_AMD_INTC_stacked.csv"
        )

    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError:
        print("Install: pip install statsmodels", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for t in ("NVDA", "AMD", "INTC"):
        fp = PREP / f"merged_{t}_market_sentiment_regression_hf.csv"
        if fp.exists():
            run_firm(fp, f"firm_{t}_hf", rows)

    run_universe(uni_hf, rows)
    for bench in ("SOXX", "SPY"):
        run_pooled(bench, rows, PREP)

    run_stacked(stacked_path, rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
