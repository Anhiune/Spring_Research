# Bluesky Scraper

Scrapes Bluesky posts matching Tesla-related keywords using the AT Protocol Python SDK (`atproto`).

---

## Setup

1. **Install dependency**
   ```bash
   pip install atproto
   ```

2. **Credentials** are stored in `config/credentials.ini` (already filled in):
   ```ini
   [bluesky]
   handle       = anhiune.bsky.social
   app_password = <your app password>
   ```
   > ⚠️ `credentials.ini` is gitignored — it will NOT be committed to GitHub.

3. **Edit keywords** in `config/keywords.txt` — one keyword per line.

---

## Usage

```bash
# Scrape the last 7 days (default)
python scrape_bluesky.py

# Custom date range
python scrape_bluesky.py --since 2025-01-01 --until 2025-03-01

# Increase post limit per keyword
python scrape_bluesky.py --since 2025-01-01 --until 2025-03-01 --limit 1000
```

---

## Output

Results are saved to `output/bluesky_YYYYMMDD_HHMMSS.csv` with columns:

| Column               | Description                     |
|----------------------|---------------------------------|
| `uri`                | Unique post identifier          |
| `cid`                | Content ID                      |
| `author_handle`      | Bluesky handle of the author    |
| `author_display_name`| Display name of the author      |
| `text`               | Post text                       |
| `created_at`         | Timestamp (UTC)                 |
| `like_count`         | Number of likes                 |
| `repost_count`       | Number of reposts               |
| `reply_count`        | Number of replies               |
| `quote_count`        | Number of quote-posts           |

---

## Notes

- The Bluesky search API returns at most **100 posts per request**; the scraper paginates automatically.
- Posts are **deduplicated** across keywords by `uri`.
- A 0.5-second delay is added between requests to respect rate limits.
- App passwords are generated at: **Settings → Privacy and Security → App Passwords** on Bluesky.
