#!/usr/bin/env python3
"""
Rebuild the packaged analysis outputs from the prepared daily panels included in this package.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HIST = ROOT / "analysis"
PY = sys.executable
TICKERS = ("NVDA", "AMD", "INTC")
REPORT = "semiconductor_export_control_sentiment_report.qmd"
REPORT_TEX = ROOT / "semiconductor_export_control_sentiment_report.tex"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    workdir = cwd if cwd is not None else ROOT
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(workdir), check=True)


def run_reddit(skip_granger: bool) -> None:
    prep = HIST / "reddit_prepared"
    out = HIST / "analysis_output"
    diag = out / "figures" / "sentiment_diagnostics"
    trends = out / "figures" / "sentiment_return_trends"

    run(
        [
            PY,
            str(HIST / "build_universe_trend_daily.py"),
            "--merge-pooled-sentiment",
            str(prep / "pooled_daily_on_trading_calendar.csv"),
            "--merged-universe-out",
            str(HIST / "merged_universe_pooled_sentiment.csv"),
        ]
    )
    run(
        [
            PY,
            str(HIST / "make_regression_ready_panels.py"),
            "--prep-dir",
            str(prep),
            "--universe-merged-in",
            str(HIST / "merged_universe_pooled_sentiment.csv"),
            "--universe-regression-out",
            str(HIST / "merged_universe_pooled_regression.csv"),
            "--universe-regression-hf-out",
            str(HIST / "merged_universe_pooled_regression_hf.csv"),
        ]
    )
    run(
        [
            PY,
            str(HIST / "plot_sentiment_diagnostics.py"),
            "--prep-dir",
            str(prep),
            "-o",
            str(diag),
        ]
    )
    run(
        [
            PY,
            str(HIST / "plot_correlation_sentiment_vs_fwd_return.py"),
            "--prep-dir",
            str(prep),
            "--out-csv",
            str(out / "correlation_sentiment_vs_fwd_return.csv"),
            "--fig-dir",
            str(diag),
        ]
    )
    run(
        [
            PY,
            str(HIST / "plot_sentiment_vs_return_trends.py"),
            "--prep-dir",
            str(prep),
            "-o",
            str(trends),
        ]
    )
    run(
        [
            PY,
            str(HIST / "run_hf_ols_tier1.py"),
            "--prep-dir",
            str(prep),
            "--universe-hf",
            str(HIST / "merged_universe_pooled_regression_hf.csv"),
            "--out-csv",
            str(out / "hf_ols_tier1_results.csv"),
        ]
    )
    run(
        [
            PY,
            str(HIST / "report_make_summary_tex.py"),
            "--hf-results",
            str(out / "hf_ols_tier1_results.csv"),
            "--tex-dir",
            str(out / "report_tables"),
        ]
    )
    run([PY, str(HIST / "run_sentiment_roll15_compare.py")])

    if not skip_granger:
        for ticker in TICKERS:
            run(
                [
                    PY,
                    str(HIST / "run_granger_firm.py"),
                    "--ticker",
                    ticker,
                    "--prep-dir",
                    str(prep),
                    "--out-dir",
                    str(out),
                ]
            )
        run(
            [
                PY,
                str(HIST / "run_granger_overall.py"),
                "--input",
                str(HIST / "merged_universe_pooled_regression.csv"),
                "--out-dir",
                str(out),
            ]
        )


def run_bluesky(skip_granger: bool) -> None:
    prep = HIST / "bluesky_prepared"
    out = HIST / "analysis_output_bluesky"
    diag = out / "figures" / "sentiment_diagnostics"
    trends = out / "figures" / "sentiment_return_trends"

    run(
        [
            PY,
            str(HIST / "build_universe_trend_daily.py"),
            "--merge-pooled-sentiment",
            str(prep / "pooled_daily_on_trading_calendar.csv"),
            "--merged-universe-out",
            str(HIST / "merged_universe_pooled_sentiment_bluesky.csv"),
        ]
    )
    run(
        [
            PY,
            str(HIST / "make_regression_ready_panels.py"),
            "--prep-dir",
            str(prep),
            "--universe-merged-in",
            str(HIST / "merged_universe_pooled_sentiment_bluesky.csv"),
            "--universe-regression-out",
            str(HIST / "merged_universe_pooled_regression_bluesky.csv"),
            "--universe-regression-hf-out",
            str(HIST / "merged_universe_pooled_regression_hf_bluesky.csv"),
        ]
    )
    run(
        [
            PY,
            str(HIST / "plot_sentiment_diagnostics.py"),
            "--prep-dir",
            str(prep),
            "-o",
            str(diag),
        ]
    )
    run(
        [
            PY,
            str(HIST / "plot_correlation_sentiment_vs_fwd_return.py"),
            "--prep-dir",
            str(prep),
            "--out-csv",
            str(out / "correlation_sentiment_vs_fwd_return.csv"),
            "--fig-dir",
            str(diag),
        ]
    )
    run(
        [
            PY,
            str(HIST / "plot_sentiment_vs_return_trends.py"),
            "--prep-dir",
            str(prep),
            "-o",
            str(trends),
        ]
    )
    run(
        [
            PY,
            str(HIST / "run_hf_ols_tier1.py"),
            "--prep-dir",
            str(prep),
            "--universe-hf",
            str(HIST / "merged_universe_pooled_regression_hf_bluesky.csv"),
            "--out-csv",
            str(out / "hf_ols_tier1_results.csv"),
        ]
    )
    run(
        [
            PY,
            str(HIST / "report_make_summary_tex.py"),
            "--hf-results",
            str(out / "hf_ols_tier1_results.csv"),
            "--tex-dir",
            str(out / "report_tables"),
        ]
    )

    if not skip_granger:
        for ticker in TICKERS:
            run(
                [
                    PY,
                    str(HIST / "run_granger_firm.py"),
                    "--ticker",
                    ticker,
                    "--prep-dir",
                    str(prep),
                    "--out-dir",
                    str(out),
                ]
            )
        run(
            [
                PY,
                str(HIST / "run_granger_overall.py"),
                "--input",
                str(HIST / "merged_universe_pooled_regression_bluesky.csv"),
                "--out-dir",
                str(out),
            ]
        )


def run_cross_platform() -> None:
    run([PY, str(HIST / "make_reddit_vs_bluesky_montage.py")])
    run([PY, str(HIST / "plot_platform_compare_clean.py")])


def render_report() -> None:
    quarto = shutil.which("quarto")
    if not quarto:
        raise SystemExit("Quarto is not installed or not on PATH.")
    run([quarto, "render", REPORT, "--to", "html", "--embed-resources"], cwd=ROOT)
    for target in ("docx", "pdf"):
        run([quarto, "render", REPORT, "--to", target], cwd=ROOT)
    if REPORT_TEX.exists():
        REPORT_TEX.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the packaged analysis and optional report.")
    parser.add_argument("--skip-granger", action="store_true", help="Skip Granger and VAR outputs.")
    parser.add_argument("--render", action="store_true", help="Render the main Quarto report after the analysis run.")
    args = parser.parse_args()

    run_reddit(skip_granger=args.skip_granger)
    run_bluesky(skip_granger=args.skip_granger)
    run_cross_platform()

    if args.render:
        render_report()

    print("Done.")


if __name__ == "__main__":
    main()
