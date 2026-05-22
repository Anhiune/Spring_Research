#!/usr/bin/env python3
"""
Direction / up-down forecasts from **lag-1** sentiment information only,
plus simple **trend** summaries over the previous 5â€“15 sessions (all known by
the time you predict from row date t).

Targets
-------
  â€¢ next_day    : sign(ret_1d[t+1])           â€” tomorrow up vs down
  â€¢ fwd_5d      : sign( sum_{k=1..5} ret[t+k] )
  â€¢ fwd_10d     : sign( sum_{k=1..10} ret[t+k] )
  â€¢ fwd_15d     : sign( sum_{k=1..15} ret[t+k] )

Features (per sentiment net, aligned to **no same-day outcome leakage**)
-----------------------------------------------------------------------
  â€¢ level_lag1  : sentiment.shift(1)  â€” yesterdayâ€™s net (same as your lag-1 spec)
  â€¢ trend_5     : sentiment.shift(1) - sentiment.shift(6)
                  â€” 5-session change ending yesterday (mood slope)
  â€¢ trend_10    : sentiment.shift(1) - sentiment.shift(11)
  â€¢ trend_15    : sentiment.shift(1) - sentiment.shift(16)

Models: **logistic regression** â€” in-sample statsmodels Logit (pseudo-RÂ², LLF),
plus **time-honest** metrics on the same specification:

  â€¢ **time_split_*:** fit sklearn LogisticRegression on the **first** train_frac
    of rows (chronological); accuracy / AUC on the **held-out tail** only.
  â€¢ **walkforward_*:** **expanding** training window â€” for each t from min_train..n-1,
    refit on rows [0:t), predict row t (true sequential forecast). Mean accuracy
    and AUC on those one-step predictions.

Walk-forward uses **sklearn** (fast refits). Install: `pip install scikit-learn`.

Output: analysis_output/sentiment_direction_forecast_results.csv

Usage (from analysis/):
  python run_sentiment_direction_forecast.py
  python run_sentiment_direction_forecast.py --train-frac 0.75 --wf-min-train 100
  python run_sentiment_direction_forecast.py --no-walkforward   # split only
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PREP = SCRIPT_DIR / "reddit_prepared"
OUT_CSV = SCRIPT_DIR / "analysis_output" / "sentiment_direction_forecast_results.csv"
TICKERS = ("NVDA", "AMD", "INTC")


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


def forward_return_sum(r: pd.Series, h: int) -> pd.Series:
    """Sum of ret[t+1]..ret[t+h] aligned at index t."""
    parts = [r.shift(-k) for k in range(1, h + 1)]
    return pd.concat(parts, axis=1).sum(axis=1)


def build_features(s: pd.Series) -> pd.DataFrame:
    s = pd.to_numeric(s, errors="coerce")
    return pd.DataFrame(
        {
            "level_lag1": s.shift(1),
            "trend_5": s.shift(1) - s.shift(6),
            "trend_10": s.shift(1) - s.shift(11),
            "trend_15": s.shift(1) - s.shift(16),
        }
    )


def majority_acc(y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    mode = int(y.mean() >= 0.5)
    return float((y == mode).mean())


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _sklearn_logit_split(
    Xv: np.ndarray,
    yv: np.ndarray,
    train_frac: float,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    n = len(yv)
    cut = int(np.floor(n * train_frac))
    out = {
        "time_split_accuracy": float("nan"),
        "time_split_baseline_accuracy": float("nan"),
        "time_split_acc_minus_baseline": float("nan"),
        "time_split_auc": float("nan"),
        "time_split_n_train": float("nan"),
        "time_split_n_test": float("nan"),
    }
    if cut < 40 or n - cut < 15:
        return out
    X_tr, y_tr = Xv[:cut], yv[:cut]
    X_te, y_te = Xv[cut:], yv[cut:]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return out
    clf = LogisticRegression(max_iter=300, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clf.fit(X_tr, y_tr)
        except Exception:
            return out
    p_te = clf.predict_proba(X_te)[:, 1]
    pred = (p_te >= 0.5).astype(int)
    acc = float((pred == y_te).mean())
    base = majority_acc(y_te.astype(float))
    out.update(
        {
            "time_split_accuracy": acc,
            "time_split_baseline_accuracy": base,
            "time_split_acc_minus_baseline": acc - base,
            "time_split_auc": safe_auc(y_te.astype(int), p_te),
            "time_split_n_train": float(cut),
            "time_split_n_test": float(len(y_te)),
        }
    )
    return out


def _sklearn_logit_walkforward(
    Xv: np.ndarray,
    yv: np.ndarray,
    min_train: int,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    n = len(yv)
    out = {
        "walkforward_accuracy": float("nan"),
        "walkforward_baseline_accuracy": float("nan"),
        "walkforward_acc_minus_baseline": float("nan"),
        "walkforward_auc": float("nan"),
        "walkforward_n_predictions": float("nan"),
    }
    if n <= min_train + 15:
        return out
    preds: list[int] = []
    probs: list[float] = []
    trues: list[int] = []
    for i in range(min_train, n):
        X_tr, y_tr = Xv[:i], yv[:i]
        if len(np.unique(y_tr)) < 2:
            continue
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf.fit(X_tr, y_tr)
            except Exception:
                continue
        pr = clf.predict_proba(Xv[i : i + 1])[:, 1]
        p = float(np.asarray(pr, dtype=float).ravel()[0])
        probs.append(p)
        preds.append(1 if p >= 0.5 else 0)
        trues.append(int(yv[i]))
    if len(preds) < 20:
        return out
    pr = np.array(preds)
    te = np.array(trues)
    pb = np.array(probs)
    acc = float((pr == te).mean())
    base = majority_acc(te.astype(float))
    out.update(
        {
            "walkforward_accuracy": acc,
            "walkforward_baseline_accuracy": base,
            "walkforward_acc_minus_baseline": acc - base,
            "walkforward_auc": safe_auc(te, pb),
            "walkforward_n_predictions": float(len(preds)),
        }
    )
    return out


def fit_one(
    y: pd.Series,
    X: pd.DataFrame,
    model_label: str,
    train_frac: float,
    walkforward: bool,
    wf_min_train: int,
) -> dict | None:
    import statsmodels.api as sm

    dfm = pd.concat([y.rename("y"), X], axis=1).dropna()
    dfm = dfm.reset_index(drop=True)
    if len(dfm) < 60:
        return None
    yv = dfm["y"].to_numpy(dtype=float)
    if len(np.unique(yv)) < 2:
        return None
    Xmat = dfm[X.columns].to_numpy(dtype=float)
    Xd = sm.add_constant(Xmat, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(yv, Xd).fit(disp=False, maxiter=100)
        except Exception:
            return None
    p_hat = res.predict(Xd)
    pred = (p_hat >= 0.5).astype(int)
    acc = float((pred == yv).mean())
    base = majority_acc(yv)
    row = {
        "model": model_label,
        "n": len(yv),
        "accuracy_insample": acc,
        "baseline_accuracy_insample": base,
        "acc_minus_baseline_insample": acc - base,
        "pseudo_r2_mcfadden": float(res.prsquared),
        "llf": float(res.llf),
        "auc_insample": safe_auc(yv.astype(int), p_hat),
    }
    for k, v in {
        "time_split_accuracy": float("nan"),
        "time_split_baseline_accuracy": float("nan"),
        "time_split_acc_minus_baseline": float("nan"),
        "time_split_auc": float("nan"),
        "time_split_n_train": float("nan"),
        "time_split_n_test": float("nan"),
        "walkforward_accuracy": float("nan"),
        "walkforward_baseline_accuracy": float("nan"),
        "walkforward_acc_minus_baseline": float("nan"),
        "walkforward_auc": float("nan"),
        "walkforward_n_predictions": float("nan"),
    }.items():
        row.setdefault(k, v)
    try:
        row.update(_sklearn_logit_split(Xmat, yv.astype(int), train_frac))
    except ImportError:
        pass
    if walkforward:
        try:
            row.update(_sklearn_logit_walkforward(Xmat, yv.astype(int), wf_min_train))
        except ImportError:
            pass
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=list(TICKERS))
    ap.add_argument(
        "--train-frac",
        type=float,
        default=0.75,
        help="Early fraction for time-split train (rest = test)",
    )
    ap.add_argument(
        "--wf-min-train",
        type=int,
        default=0,
        help="Min training rows before first walk-forward pred (0 = max(80, 35%% of n))",
    )
    ap.add_argument(
        "--no-walkforward",
        action="store_true",
        help="Skip expanding-window walk-forward (faster)",
    )
    args = ap.parse_args()

    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError:
        print("Install: pip install statsmodels", file=sys.stderr)
        sys.exit(1)

    sent_map = [
        ("nrc_net_sentiment", "NRC_net"),
        ("hf_net_4", "HF_net_4"),
        ("fb_net_sentiment", "FB_net"),
    ]

    horizons = [
        ("next_day", 1),
        ("fwd_5d", 5),
        ("fwd_10d", 10),
        ("fwd_15d", 15),
    ]

    rows = []
    for tkr in args.tickers:
        try:
            d = load_firm(tkr)
        except FileNotFoundError as e:
            print(f"Skip {tkr}: {e}")
            continue
        r = d["ret_1d"]

        for h_label, h in horizons:
            if h == 1:
                y_raw = r.shift(-1)
            else:
                y_raw = forward_return_sum(r, h)
            y = (y_raw > 0).astype(float)

            for col, slab in sent_map:
                if col not in d.columns:
                    continue
                s = d[col] if col != "hf_net_4" else d["hf_net_4"]
                Fe = build_features(s)
                dfm0 = pd.concat([y.rename("y"), Fe], axis=1).dropna()
                wf_min = args.wf_min_train or max(80, int(0.35 * len(dfm0)))

                kw = dict(
                    train_frac=args.train_frac,
                    walkforward=not args.no_walkforward,
                    wf_min_train=wf_min,
                )

                # (1) level only
                r1 = fit_one(y, Fe[["level_lag1"]], "logit_level_lag1", **kw)
                if r1:
                    rows.append({"ticker": tkr, "sentiment": slab, "target": h_label, **r1})

                # (2) level + trend_5 â€” "mood drift" + level
                r2 = fit_one(y, Fe[["level_lag1", "trend_5"]], "logit_level_plus_trend5", **kw)
                if r2:
                    rows.append({"ticker": tkr, "sentiment": slab, "target": h_label, **r2})

                # (3) level + trend_10
                r3 = fit_one(y, Fe[["level_lag1", "trend_10"]], "logit_level_plus_trend10", **kw)
                if r3:
                    rows.append({"ticker": tkr, "sentiment": slab, "target": h_label, **r3})

                # (4) level + trend_15 â€” closest to your 5â€“15 day story in one shot
                r4 = fit_one(y, Fe[["level_lag1", "trend_15"]], "logit_level_plus_trend15", **kw)
                if r4:
                    rows.append({"ticker": tkr, "sentiment": slab, "target": h_label, **r4})

    if not rows:
        print("No results â€” check data / installs (sklearn optional for AUC).")
        return

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(out)} rows)")
    key = "walkforward_acc_minus_baseline"
    if key in out.columns:
        g = out.sort_values(key, ascending=False, na_position="last").groupby(
            ["ticker", "target"], as_index=False
        ).head(3)
        show = [
            "ticker",
            "target",
            "sentiment",
            "model",
            "time_split_acc_minus_baseline",
            "walkforward_acc_minus_baseline",
            "time_split_auc",
            "walkforward_auc",
        ]
        show = [c for c in show if c in g.columns]
        print("\nTop walk-forward acc lift vs majority (3 per tickerÃ—target):")
        pd.set_option("display.width", 220)
        print(g[show].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()

