"""
Bluesky Post Scraper
====================
Searches Bluesky (via AT Protocol) for posts matching keywords
and saves results to a timestamped CSV in the output/ directory.

Usage:
    python scrape_bluesky.py [OPTIONS]

Options:
    --since   Start date  (YYYY-MM-DD). Default: 7 days ago.
    --until   End date    (YYYY-MM-DD). Default: today.
    --limit   Max posts per keyword.   Default: 500.
    --output  Output directory.        Default: output/

Example:
    python scrape_bluesky.py --since 2025-01-01 --until 2025-03-01 --limit 1000
"""

import argparse
import configparser
import csv
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from atproto import Client
from atproto.exceptions import AtProtocolError

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
OUTPUT_DIR = BASE_DIR / "output"

CREDENTIALS_FILE = CONFIG_DIR / "credentials.ini"
KEYWORDS_FILE    = CONFIG_DIR / "keywords.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_credentials() -> tuple[str, str]:
    """Read handle and app_password from config/credentials.ini."""
    cfg = configparser.ConfigParser()
    cfg.read(CREDENTIALS_FILE)
    handle   = cfg["bluesky"]["handle"].strip()
    password = cfg["bluesky"]["app_password"].strip()
    return handle, password


def load_keywords() -> list[str]:
    """Read non-empty, non-comment lines from config/keywords.txt."""
    with open(KEYWORDS_FILE, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


def post_to_row(post) -> dict:
    """Flatten a Bluesky PostView into a flat dict for CSV output."""
    record = post.record
    return {
        "uri":          post.uri,
        "cid":          post.cid,
        "author_handle":      post.author.handle,
        "author_display_name": post.author.display_name or "",
        "text":         getattr(record, "text", ""),
        "created_at":   getattr(record, "created_at", ""),
        "like_count":   post.like_count   or 0,
        "repost_count": post.repost_count or 0,
        "reply_count":  post.reply_count  or 0,
        "quote_count":  post.quote_count  or 0,
    }


CSV_FIELDS = [
    "uri", "cid", "author_handle", "author_display_name",
    "text", "created_at",
    "like_count", "repost_count", "reply_count", "quote_count",
]


# ── Core scrape function ───────────────────────────────────────────────────────

def scrape_keyword(
    client: Client,
    keyword: str,
    since: str,
    until: str,
    max_posts: int,
    seen_uris: set,
) -> list[dict]:
    """
    Paginate through Bluesky search results for `keyword`.
    Returns a list of row dicts for posts not already in seen_uris.
    """
    rows   = []
    cursor = None

    print(f"  Searching: '{keyword}'", end="", flush=True)

    while len(rows) < max_posts:
        try:
            params = {
                "q":     keyword,
                "limit": min(100, max_posts - len(rows)),  # API max is 100
                "since": since,
                "until": until,
            }
            if cursor:
                params["cursor"] = cursor

            response = client.app.bsky.feed.search_posts(params)
        except AtProtocolError as exc:
            print(f"\n    [!] API error for '{keyword}': {exc}")
            break

        posts = response.posts
        if not posts:
            break

        for post in posts:
            if post.uri not in seen_uris:
                seen_uris.add(post.uri)
                rows.append(post_to_row(post))

        print(".", end="", flush=True)

        cursor = getattr(response, "cursor", None)
        if not cursor:
            break

        time.sleep(0.5)   # polite rate-limiting

    print(f" → {len(rows)} new posts")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    today = datetime.now(timezone.utc).date()
    week_ago = today - timedelta(days=7)

    parser = argparse.ArgumentParser(description="Scrape Bluesky posts by keyword.")
    parser.add_argument("--since",  default=str(week_ago), help="Start date YYYY-MM-DD")
    parser.add_argument("--until",  default=str(today),    help="End date   YYYY-MM-DD")
    parser.add_argument("--limit",  type=int, default=500, help="Max posts per keyword")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    return parser.parse_args()


def main():
    args     = parse_args()
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ISO-8601 datetime strings required by the API
    since_dt = f"{args.since}T00:00:00Z"
    until_dt = f"{args.until}T23:59:59Z"

    # ── Credentials & login ────────────────────────────────────────────────
    handle, password = load_credentials()
    client = Client()
    print(f"Logging in as {handle} …")
    client.login(handle, password)
    print("Logged in successfully.\n")

    # ── Keywords ───────────────────────────────────────────────────────────
    keywords = load_keywords()
    print(f"Keywords ({len(keywords)}): {', '.join(keywords)}\n")
    print(f"Date range: {args.since} → {args.until}")
    print(f"Max posts per keyword: {args.limit}\n")

    # ── Scrape ─────────────────────────────────────────────────────────────
    all_rows  = []
    seen_uris: set = set()

    for kw in keywords:
        rows = scrape_keyword(client, kw, since_dt, until_dt, args.limit, seen_uris)
        all_rows.extend(rows)

    # ── Save CSV ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file  = out_dir / f"bluesky_{timestamp}.csv"

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} unique posts saved to:\n  {out_file}")


if __name__ == "__main__":
    main()
