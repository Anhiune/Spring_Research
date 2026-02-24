# Reddit Scraper — PRAW + spaCy

A Python tool that scrapes Reddit submissions and comments using **PRAW** (Python Reddit API Wrapper), filters sentences with **spaCy NLP**, and exports results to CSV files. Supports both **one-time** and **continuous loop** operation.

## Features

- **PRAW Integration**: Authenticated Reddit API access via OAuth2.
- **spaCy NLP**: Sentence segmentation + noun/proper-noun keyword matching with lemmatization.
- **Continuous Mode**: Runs in a loop with configurable interval between scraping rounds.
- **Bot Filtering**: Automatically skips bot-generated content.
- **Deduplication**: Tracks processed submission & comment IDs across runs.
- **Incremental Saves**: Writes CSV every 1 000 rows to avoid data loss.
- **Graceful Shutdown**: Press Ctrl+C to finish the current item and save before exiting.
- **Robust Retries**: Configurable retry counts for API and comment-loading failures.

## Prerequisites

- Python 3.9+
- A Reddit account
- Reddit API credentials (see below)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set Up Reddit API Credentials

1. Go to <https://www.reddit.com/prefs/apps>
2. Click **"Create App"** or **"Create Another App"**
3. Fill in:
   - **name**: any name (e.g. `my_scraper`)
   - **type**: select **script**
   - **redirect uri**: `http://localhost:8080`
4. Note down the **client_id** (under the app name) and **client_secret**.

### 3. Create `praw.ini`

Copy the example template to your user config directory:

| OS | Location |
|----|----------|
| Windows | `%APPDATA%\praw.ini` |
| Linux / macOS | `~/.config/praw.ini` |

Content (fill in your values):

```ini
[bot1]
client_id=YOUR_CLIENT_ID
client_secret=YOUR_CLIENT_SECRET
password=YOUR_REDDIT_PASSWORD
username=YOUR_REDDIT_USERNAME
```

A template is provided at `config/praw_example.ini`.

### 4. Configure Subreddits & Keywords

Edit the files in the `config/` directory:

- **`config/subreddits.txt`** — one subreddit per line (lines starting with `#` are ignored).
- **`config/keywords.txt`** — keywords for filtering (lines starting with `#` are ignored).

## Usage

### Continuous loop (default)

```bash
python scrape_reddit.py --username YOUR_USERNAME
```

Runs every 30 minutes by default. Adjust with `--interval`:

```bash
python scrape_reddit.py --username YOUR_USERNAME --interval 60
```

### One-time run

```bash
python scrape_reddit.py --username YOUR_USERNAME --once
```

### Custom post limit

```bash
python scrape_reddit.py --username YOUR_USERNAME --limit 100
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--username` | `your_reddit_username_here` | Reddit username for user-agent |
| `--limit` | `500` | Posts to fetch per subreddit |
| `--once` | off | Run one round then exit |
| `--interval` | `30` | Minutes between rounds (continuous mode) |

## Output

CSV files are saved to `output/reddit_YYYYMMDD_HHMMSS.csv`.

| Column | Description |
|--------|-------------|
| `submission_id` | Reddit post ID |
| `comment_id` | Comment ID (`None` for titles/selftext) |
| `timestamp` | Unix timestamp |
| `subreddit` | Source subreddit |
| `text_type` | `title`, `selftext`, or `comment` |
| `text` | Filtered sentence |

## How Filtering Works

1. Text is split into sentences using spaCy's sentence boundary detector.
2. Each sentence is checked for keywords — **only nouns and proper nouns** are matched.
3. Both the token's surface form and its lemma are compared (case-insensitive).
4. Sentences longer than 100 words are skipped.
5. Bot content is detected by markers like "I am a bot".

## File Structure

```
reddit_scraper/
├── config/
│   ├── keywords.txt          # Keywords for filtering
│   ├── subreddits.txt         # Target subreddits
│   └── praw_example.ini       # Credential template
├── data/
│   ├── submission_ids_recorded.txt
│   └── comment_ids_recorded.txt
├── output/                    # CSV output files
├── scrape_reddit.py           # Main scraper (PRAW)
├── scrape_reddit_pullpush.py  # Alternative (PullPush API)
├── requirements.txt
├── .gitignore
└── README.md
```

## Verification

1. **Dry run (PRAW)**: `python scrape_reddit.py --username YOU --once --limit 10`
2. **Dry run (PullPush)**: `python scrape_reddit_pullpush.py --once --after 7d --max-pages 1`
3. Check `output/` for the generated CSV.
4. Check `scrape.log` / `scrape_pullpush.log` for any errors.
5. Re-run — the ID tracker should skip already-processed posts.

---

## Option B: PullPush.io API (No Credentials Needed)

The `scrape_reddit_pullpush.py` script queries [PullPush.io](https://pullpush.io/) — a free API that mirrors all of Reddit's historical data. **No Reddit API credentials required.**

### Advantages over PRAW

| | PRAW (`scrape_reddit.py`) | PullPush (`scrape_reddit_pullpush.py`) |
|---|---|---|
| Credentials | Reddit OAuth app required | None |
| Historical data | Recent ~1000 posts only | Full history back to 2005 |
| Rate limits | Reddit API limits | Generous (1 req/sec is fine) |
| Comments | Full thread via `replace_more()` | Keyword-searched comments |
| Real-time | Yes (live API) | Slight delay (mirrors) |

### Usage

```bash
# One-time: last 30 days of submissions only
python scrape_reddit_pullpush.py --once --after 30d

# One-time: with comments too
python scrape_reddit_pullpush.py --once --after 30d --comments

# All historical data (no date filter)
python scrape_reddit_pullpush.py --once

# Continuous loop every 60 minutes
python scrape_reddit_pullpush.py --interval 60

# Limit API pages per (subreddit, keyword) pair
python scrape_reddit_pullpush.py --once --max-pages 5
```

### All PullPush options

| Flag | Default | Description |
|------|---------|-------------|
| `--after` | none | Only results after this date (epoch or `30d`) |
| `--before` | none | Only results before this date |
| `--comments` | off | Also search comments (more API calls) |
| `--max-pages` | 50 | Max pages per (subreddit, keyword) pair |
| `--once` | off | Run once then exit |
| `--interval` | 30 | Minutes between rounds (continuous mode) |

### Search Terms vs Filter Keywords

The scraper uses **two keyword lists**:

1. **`config/search_terms.txt`** — Sent to the PullPush API as the `q` parameter. Keep this focused (10–20 broad terms) to avoid excessive API calls.
2. **`config/keywords.txt`** — Used locally by spaCy to filter sentences from the retrieved posts. Can be a large, detailed list.

If `search_terms.txt` doesn't exist, the script falls back to using all keywords from `keywords.txt` as search queries.

---
*Developed for UROP Research Spring 2026*
