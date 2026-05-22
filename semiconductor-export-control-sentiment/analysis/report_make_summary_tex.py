#!/usr/bin/env python3
"""Build LaTeX booktabs tables into analysis_output/report_tables/ for Overleaf or PDF.
Called from Quarto or: python report_make_summary_tex.py"""
from __future__ import annotations

from pathlib import Path

import argparse
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "analysis_output"
TEX_DIR = OUT / "report_tables"


def tex_cell(s: str) -> str:
    """Minimal escaping for table body cells."""
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def latex_coef_cell(r: pd.Series) -> str:
    """Multi-line cell: coef + stars, SE, then p/n/R2 (journal-style density)."""
    p = float(r.pvalue_sentiment)
    if p < 0.01:
        stars = r"^{***}"
    elif p < 0.05:
        stars = r"^{**}"
    elif p < 0.1:
        stars = r"^{*}"
    else:
        stars = ""
    line1 = rf"$\hat{{\beta}}={r.coef_sentiment:.4f}{stars}$"
    line2 = rf"$({r.se_sentiment:.4f})$"
    line3 = rf"{{\scriptsize $p={p:.3f}$; $n={int(r.n)}$; $R^2={r.r2:.4f}$}}"
    return rf"\shortstack{{{line1}\\{line2}\\{line3}}}"


def firm_hf_wide(hf: pd.DataFrame) -> pd.DataFrame:
    tickers = ["NVDA", "AMD", "INTC"]
    rows = []
    for spec_key, spec_label in [
        ("A_bivariate", r"OLS HAC: $ret_t \sim$ HF net lag 1 (only)"),
        ("B_plus_soxx_spy_vix", r"OLS HAC: $+\ \mathrm{SOXX},\mathrm{SPY},\mathrm{VIX}$"),
    ]:
        cells = {"Specification": spec_label}
        for t in tickers:
            sub = hf[(hf["dataset"] == f"firm_{t}_hf") & (hf["spec"] == spec_key)]
            if sub.empty:
                cells[t] = "—"
            else:
                cells[t] = latex_coef_cell(sub.iloc[0])
        rows.append(cells)
    return pd.DataFrame(rows)


def other_blocks(hf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, sp, label in [
        ("universe_pooled_hf", "A_bivariate", "Universe ew $ret$ $\\sim$ HF net (bivariate)"),
        ("universe_pooled_hf", "B_plus_soxx_spy_vix", "Universe + benchmarks"),
        ("pooled_broad_x_SOXX_return", "A_bivariate", "Pooled broad $\\times$ SOXX bench return"),
        ("pooled_broad_x_SOXX_return", "B_plus_other_benchmark_controls", "+ other bench (excl. SOXX own)"),
        ("pooled_broad_x_SPY_return", "A_bivariate", "Pooled broad $\\times$ SPY"),
        ("pooled_broad_x_SPY_return", "B_plus_other_benchmark_controls", "+ other bench"),
        ("stacked_NVDA_AMD_INTC", "A_bivariate_plus_ticker_dummies", "Stacked + ticker FE"),
        ("stacked_NVDA_AMD_INTC", "B_plus_controls_and_ticker_dummies", "+ FE + benchmarks"),
    ]:
        sub = hf[(hf["dataset"] == ds) & (hf["spec"] == sp)]
        if sub.empty:
            rows.append({"Block": label, "Result": "—"})
        else:
            rows.append(
                {
                    "Block": label,
                    "Result": latex_coef_cell(sub.iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def df_to_booktabs(df: pd.DataFrame, caption: str, label: str, col_width: str) -> str:
    ncol = len(df.columns)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{tex_cell(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + f"p{{{col_width}}}" * (ncol - 1) + "}",
        r"\toprule",
    ]
    cols = list(df.columns)
    lines.append(" & ".join(tex_cell(c) for c in cols) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_cell(v) for v in row.values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hf-results",
        type=Path,
        default=BASE / "analysis_output" / "hf_ols_tier1_results.csv",
        help="Input CSV from run_hf_ols_tier1.py",
    )
    p.add_argument(
        "--tex-dir",
        type=Path,
        default=BASE / "analysis_output" / "report_tables",
        help="Directory for summary_hf_ols_master.tex",
    )
    args = p.parse_args()

    TEX_DIR = args.tex_dir
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    p_in = args.hf_results
    if not p_in.exists():
        print(f"Missing {p_in}")
        return
    hf = pd.read_csv(p_in)
    t1 = firm_hf_wide(hf)
    t2 = other_blocks(hf)
    out = TEX_DIR / "summary_hf_ols_master.tex"
    body = [
        "% Auto-generated — \\input in LaTeX (needs booktabs; \\shortstack is standard)",
        "% Stars: * p<0.10, ** p<0.05, *** p<0.01",
        df_to_booktabs(
            t1,
            "Firm-level OLS with Newey--West HAC (HF net four-class, lag 1).",
            "tab:ols-firm-hf",
            "5.6cm",
        ),
        df_to_booktabs(
            t2,
            "Other HF OLS blocks: universe, pooled benchmarks, stacked panel.",
            "tab:ols-other-hf",
            "11cm",
        ),
    ]
    out.write_text("\n".join(body), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
