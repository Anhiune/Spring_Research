# Bluesky Scraper

This utility collects Bluesky posts for the semiconductor export-control sentiment project using the AT Protocol Python SDK (`atproto`).

## Included Files

- `scrape_bluesky.py`: Single-range scraper for one date window
- `scrape_bluesky_batch.py`: Month-by-month batch scraper for longer historical pulls
- `config/keywords.txt`: Project keyword list used by the scraper
- `config/credentials.example.ini`: Safe credential template

## Setup

1. Install the dependency:

```bash
pip install -r requirements.txt
```

2. Create `config/credentials.ini` from the example template and fill in your Bluesky handle and app password.

3. Review or edit `config/keywords.txt` as needed.

## Usage

```bash
python scrape_bluesky.py
python scrape_bluesky.py --since 2025-01-01 --until 2025-03-01 --limit 1000
python scrape_bluesky_batch.py --start 2024-01 --end 2024-12
```

## Notes

- `config/credentials.ini` is intentionally ignored by git and is not included in this package.
- The batch scraper writes monthly files by default and is the better option for larger historical pulls.
- Output folders are created automatically when the scraper runs.
