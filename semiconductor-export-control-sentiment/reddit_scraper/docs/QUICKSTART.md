# Quick Start - Advanced Reddit Scraper

## 1. Environment Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 2. Configuration
- **Keywords**:
  - `config/keywords_general.txt`: Add policy terms (e.g., `bis`, `sanctions`).
  - `config/keywords_ontopic.txt`: Add company names/tickers (e.g., `nvda`, `amd`).
- **Subreddits**:
  - `config/subreddits.txt`: Add targets (e.g., `semiconductors`, `stocks`).

## 3. Run
```bash
python scrape_reddit_pullpush.py
```

## 4. View Results
Results go to `output/` in two formats:
- **CSV**: Best for data analysis/R/Stata.
- **Excel**: Best for manual review (sheets grouped by company).

## Troubleshooting
- **API (502/504)**: PullPush is a community service; if it's down, wait and retry.
- **No Results**: Verify your general keywords and on-topic keywords match common terms in those subreddits.
