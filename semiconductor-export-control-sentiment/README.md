# Semiconductor Export-Control Sentiment

This package is the polished final research bundle for the Spring 2026 UROP project on cross-platform semiconductor sentiment and short-run equity-return analysis.

It includes the final report, the final poster, the prepared analysis inputs, the generated analysis outputs, and the scripts needed to rebuild the packaged results from the prepared-data stage forward.

## Final Deliverables

- `semiconductor_export_control_sentiment_report.qmd`: Quarto source for the final report
- `semiconductor_export_control_sentiment_report.pdf`: Final report in PDF format
- `semiconductor_export_control_sentiment_report.docx`: Final report in Word format
- `semiconductor_export_control_sentiment_report.html`: Final report in standalone HTML format
- `semiconductor_export_control_sentiment_poster.pdf`: Final poster in PDF format
- `semiconductor_export_control_sentiment_poster.html`: Poster export in HTML format

## Package Layout

- `analysis/`: Prepared panels, market data, analysis scripts, and generated figures/tables
- `reddit_scraper/`: Reddit collection utilities and configuration templates
- `bluesky_scraper/`: Bluesky collection utilities and configuration templates
- `references.bib`: Bibliography used by the report
- `report_reference.docx`: Word reference document used during Quarto rendering
- `requirements.txt`: Python dependencies for the packaged analysis workflow
- `rebuild_analysis.py`: One-command rebuild script for the packaged analysis outputs

## Software Requirements

- Python 3.10 or newer
- Quarto, if the report will be re-rendered

Install Python packages with:

```powershell
pip install -r requirements.txt
```

## Rebuild the Analysis

To rebuild the packaged analysis artifacts from the included prepared panels:

```powershell
python rebuild_analysis.py
```

To rebuild the analysis and then render the report:

```powershell
python rebuild_analysis.py --render
```

## Render the Report Only

If the packaged analysis outputs are already present and only the report needs to be rendered, run the desired target directly:

```powershell
quarto render semiconductor_export_control_sentiment_report.qmd --to html --embed-resources
quarto render semiconductor_export_control_sentiment_report.qmd --to docx
quarto render semiconductor_export_control_sentiment_report.qmd --to pdf
```

## Notes

- The prepared datasets used by the report are already included in `analysis/`.
- The package is organized around reproducibility from prepared data onward; raw collection is not required to rebuild the included results.
- Scraper credentials are not included. Use the example configuration files in the scraper folders to create local credential files when needed.
