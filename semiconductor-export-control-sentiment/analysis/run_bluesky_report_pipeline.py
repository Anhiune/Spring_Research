#!/usr/bin/env python3
"""
One-shot pipeline: Bluesky → same artifacts as the Reddit Quarto report.

**Row-level path** (firm/broad from keywords): needs
  bluesky_data/bluesky_all_cleaned_with_sentiment.csv
  → prepare_bluesky_dual_layers.py + prepare_pooled_daily_sentiment.py

**Aggregate daily path** (pooled * _mean columns, one series for all tickers): needs
  bluesky_data/bluesky_all_cleaned_daily_sentiment.csv
  → prepare_bluesky_from_aggregate_daily.py

If both exist, the row-level scored file takes precedence.

Usage:
  python run_bluesky_report_pipeline.py
  python run_bluesky_report_pipeline.py --skip-granger
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Full Bluesky → report pipeline.")
    p.add_argument(
        "--bluesky-csv",
        type=Path,
        default=REPO_ROOT / "bluesky_data" / "bluesky_all_cleaned_with_sentiment.csv",
        help="Row-level scored CSV (optional if --daily-csv exists)",
    )
    p.add_argument(
        "--daily-csv",
        type=Path,
        default=REPO_ROOT / "bluesky_data" / "bluesky_all_cleaned_daily_sentiment.csv",
        help="Pooled daily aggregate with *_mean columns",
    )
    p.add_argument("--skip-granger", action="store_true")
    args = p.parse_args()

    scored = args.bluesky_csv.resolve()
    daily = args.daily_csv.resolve()

    PREP = SCRIPT_DIR / "bluesky_prepared"
    AN = SCRIPT_DIR / "analysis_output_bluesky"
    FIG_DIAG = AN / "figures" / "sentiment_diagnostics"
    FIG_TREND = AN / "figures" / "sentiment_return_trends"
    UNI_MERGE = SCRIPT_DIR / "merged_universe_pooled_sentiment_bluesky.csv"
    UNI_REG = SCRIPT_DIR / "merged_universe_pooled_regression_bluesky.csv"
    UNI_REG_HF = SCRIPT_DIR / "merged_universe_pooled_regression_hf_bluesky.csv"
    POOLED_CAL = PREP / "pooled_daily_on_trading_calendar.csv"

    py = sys.executable

    if scored.exists():
        print("Using row-level scored Bluesky CSV -> dual layers + pooled calendar.")
        run(
            [
                py,
                str(SCRIPT_DIR / "prepare_bluesky_dual_layers.py"),
                "--bluesky-csv",
                str(scored),
                "-o",
                str(PREP),
            ]
        )
        run(
            [
                py,
                str(SCRIPT_DIR / "prepare_pooled_daily_sentiment.py"),
                "--prep-dir",
                str(PREP),
                "-o",
                str(PREP),
            ]
        )
    elif daily.exists():
        print(
            "Using pooled daily Bluesky CSV -> shared aggregate sentiment on NVDA/AMD/INTC "
            "(for firm-specific series, add row-level *_with_sentiment.csv)."
        )
        run(
            [
                py,
                str(SCRIPT_DIR / "prepare_bluesky_from_aggregate_daily.py"),
                "--input",
                str(daily),
                "-o",
                str(PREP),
            ]
        )
    else:
        print(
            "Need either:\n"
            f"  {scored}\n"
            "or\n"
            f"  {daily}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not POOLED_CAL.exists():
        print(f"Missing {POOLED_CAL} after prep step.", file=sys.stderr)
        sys.exit(1)

    run(
        [
            py,
            str(SCRIPT_DIR / "build_universe_trend_daily.py"),
            "--merge-pooled-sentiment",
            str(POOLED_CAL),
            "--merged-universe-out",
            str(UNI_MERGE),
        ]
    )
    run(
        [
            py,
            str(SCRIPT_DIR / "make_regression_ready_panels.py"),
            "--prep-dir",
            str(PREP),
            "--universe-merged-in",
            str(UNI_MERGE),
            "--universe-regression-out",
            str(UNI_REG),
            "--universe-regression-hf-out",
            str(UNI_REG_HF),
        ]
    )

    FIG_DIAG.mkdir(parents=True, exist_ok=True)
    FIG_TREND.mkdir(parents=True, exist_ok=True)

    run(
        [
            py,
            str(SCRIPT_DIR / "plot_sentiment_diagnostics.py"),
            "--prep-dir",
            str(PREP),
            "-o",
            str(FIG_DIAG),
        ]
    )
    run(
        [
            py,
            str(SCRIPT_DIR / "plot_correlation_sentiment_vs_fwd_return.py"),
            "--prep-dir",
            str(PREP),
            "--out-csv",
            str(AN / "correlation_sentiment_vs_fwd_return.csv"),
            "--fig-dir",
            str(FIG_DIAG),
        ]
    )
    run(
        [
            py,
            str(SCRIPT_DIR / "plot_sentiment_vs_return_trends.py"),
            "--prep-dir",
            str(PREP),
            "-o",
            str(FIG_TREND),
        ]
    )
    run(
        [
            py,
            str(SCRIPT_DIR / "run_hf_ols_tier1.py"),
            "--prep-dir",
            str(PREP),
            "--universe-hf",
            str(UNI_REG_HF),
            "--out-csv",
            str(AN / "hf_ols_tier1_results.csv"),
        ]
    )
    run(
        [
            py,
            str(SCRIPT_DIR / "report_make_summary_tex.py"),
            "--hf-results",
            str(AN / "hf_ols_tier1_results.csv"),
            "--tex-dir",
            str(AN / "report_tables"),
        ]
    )

    if not args.skip_granger:
        for t in ("NVDA", "AMD", "INTC"):
            run(
                [
                    py,
                    str(SCRIPT_DIR / "run_granger_firm.py"),
                    "--ticker",
                    t,
                    "--prep-dir",
                    str(PREP),
                    "--out-dir",
                    str(AN),
                ]
            )
        run(
            [
                py,
                str(SCRIPT_DIR / "run_granger_overall.py"),
                "--input",
                str(UNI_REG),
                "--out-dir",
                str(AN),
            ]
        )

    print(
        "\nDone. Render report:\n"
        "  cd historical data\n"
        "  quarto render UROP_bluesky_sentiment_analysis_report.qmd"
    )


if __name__ == "__main__":
    main()
