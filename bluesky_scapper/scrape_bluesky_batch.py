"""
Bluesky Batch Scraper — Month-by-Month
=======================================
Loops through every calendar month from START_YEAR/START_MONTH to today,
searches all keywords from config/keywords.txt, deduplicates by URI,
and saves one CSV per month into the output folder (default: ../bluesky_data/).

Usage:
    python scrape_bluesky_batch.py [OPTIONS]

Options:
    --start     First month to scrape  (YYYY-MM). Default: 2023-01
    --end       Last  month to scrape  (YYYY-MM). Default: current month
    --limit     Max posts per keyword per month.   Default: 500
    --output    Output directory.                  Default: ../bluesky_data
    --resume    Skip months whose CSV already exists (default: True)

Examples:
    # Full historical run (2023-01 → today)
    python scrape_bluesky_batch.py

    # Just 2024
    python scrape_bluesky_batch.py --start 2024-01 --end 2024-12

    # Re-scrape everything from scratch (ignore existing files)
    python scrape_bluesky_batch.py --no-resume
"""

import argparse
import configparser
import csv
import os
import re
import time
from calendar import monthrange
from datetime import datetime, timezone
from pathlib import Path

from atproto import Client
from atproto.exceptions import AtProtocolError

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
CONFIG_DIR       = BASE_DIR / "config"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.ini"
KEYWORDS_FILE    = CONFIG_DIR / "keywords.txt"
DEFAULT_OUTPUT   = BASE_DIR.parent / "bluesky_data"

CSV_FIELDS = [
    "uri", "cid", "author_handle", "author_display_name",
    "text", "created_at",
    "like_count", "repost_count", "reply_count", "quote_count",
    "keyword",          # which keyword matched this post
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_credentials() -> tuple[str, str]:
    cfg = configparser.ConfigParser()
    cfg.read(CREDENTIALS_FILE)
    return cfg["bluesky"]["handle"].strip(), cfg["bluesky"]["app_password"].strip()


def load_keywords() -> list[str]:
    with open(KEYWORDS_FILE, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


def wait_for_rate_limit(exc: AtProtocolError, default_wait: int = 65):
    """
    When a 429 RateLimitExceeded is returned, parse the ratelimit-reset
    timestamp from the exception message and sleep until the window resets.
    """
    match = re.search(r"'ratelimit-reset':\s*'(\d+)'", str(exc))
    if match:
        reset_ts = int(match.group(1))
        wait_sec = max(5, reset_ts - int(time.time())) + 5   # +5s buffer
    else:
        wait_sec = default_wait
    print(f" [Rate limited -- waiting {wait_sec}s]", flush=True)
    time.sleep(wait_sec)


def relogin(client: Client, handle: str, password: str):
    """Re-authenticate to refresh the session token."""
    print("  [Re-logging in to refresh session...]", flush=True)
    try:
        client.login(handle, password)
        print("  [Re-login OK]", flush=True)
    except Exception as e:
        print(f"  [Re-login failed: {e} -- waiting 30s]", flush=True)
        time.sleep(30)


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Return list of (year, month) tuples from start='YYYY-MM' to end='YYYY-MM' inclusive."""
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]),   int(end[5:7])
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def post_to_row(post, keyword: str) -> dict:
    record = post.record
    return {
        "uri":                post.uri,
        "cid":                post.cid,
        "author_handle":      post.author.handle,
        "author_display_name": post.author.display_name or "",
        "text":               getattr(record, "text", ""),
        "created_at":         getattr(record, "created_at", ""),
        "like_count":         post.like_count   or 0,
        "repost_count":       post.repost_count or 0,
        "reply_count":        post.reply_count  or 0,
        "quote_count":        post.quote_count  or 0,
        "keyword":            keyword,
    }


def scrape_month(
    client: Client,
    handle: str,
    password: str,
    keywords: list[str],
    year: int,
    month: int,
    max_posts: int,
) -> list[dict]:
    """
    Search all keywords for a single calendar month.
    Returns deduplicated rows (by URI).
    """
    last_day   = monthrange(year, month)[1]
    since_dt   = f"{year:04d}-{month:02d}-01T00:00:00Z"
    until_dt   = f"{year:04d}-{month:02d}-{last_day:02d}T23:59:59Z"
    seen_uris: set[str] = set()
    all_rows: list[dict] = []

    for kw in keywords:
        cursor    = None
        kw_count  = 0
        retries   = 0
        max_retry = 5
        print(f"    [{kw}]", end="", flush=True)

        while kw_count < max_posts and retries < max_retry:
            try:
                params: dict = {
                    "q":     kw,
                    "limit": min(100, max_posts - kw_count),
                    "since": since_dt,
                    "until": until_dt,
                }
                if cursor:
                    params["cursor"] = cursor

                response = client.app.bsky.feed.search_posts(params)
                retries  = 0   # reset on success

            except AtProtocolError as exc:
                err_str = str(exc)
                retries += 1
                if "RateLimitExceeded" in err_str or "429" in err_str:
                    wait_for_rate_limit(exc)
                    continue   # retry same cursor
                else:
                    # Empty / network / server error: re-login then back off
                    backoff = 15 * retries
                    print(f" [error ({retries}/{max_retry}) -- re-login+wait {backoff}s]",
                          end="", flush=True)
                    relogin(client, handle, password)
                    time.sleep(backoff)
                    print(f"    [{kw}]", end="", flush=True)
                    continue
            except Exception as exc:
                retries += 1
                backoff = 15 * retries
                print(f" [unexpected error ({retries}/{max_retry}): {exc} -- wait {backoff}s]",
                      end="", flush=True)
                time.sleep(backoff)
                continue

            posts = response.posts
            if not posts:
                break

            for post in posts:
                if post.uri not in seen_uris:
                    seen_uris.add(post.uri)
                    all_rows.append(post_to_row(post, kw))
                    kw_count += 1

            print(".", end="", flush=True)

            cursor = getattr(response, "cursor", None)
            if not cursor:
                break

            time.sleep(1.0)  # stay well under 3000 req/5min rate limit

        if retries >= max_retry:
            print(f" [SKIPPED after {max_retry} retries]", end="")
        print(f" {kw_count}")

    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    now       = datetime.now(timezone.utc)
    this_month = f"{now.year:04d}-{now.month:02d}"

    parser = argparse.ArgumentParser(
        description="Scrape Bluesky posts month-by-month for semiconductor/export-control research."
    )
    parser.add_argument("--start",  default="2023-01",  help="First month YYYY-MM  (default: 2023-01)")
    parser.add_argument("--end",    default=this_month, help="Last  month YYYY-MM  (default: current month)")
    parser.add_argument("--limit",  type=int, default=500, help="Max posts per keyword per month (default: 500)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Re-scrape months that already have a CSV")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main():
    args     = parse_args()
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    handle, password = load_credentials()
    client = Client()
    print(f"Logging in as {handle} …")
    client.login(handle, password)
    print("Login successful.\n")

    keywords = load_keywords()
    months   = month_range(args.start, args.end)

    print(f"Keywords  : {len(keywords)}")
    print(f"Months    : {args.start} to {args.end}  ({len(months)} months)")
    print(f"Limit     : {args.limit} posts / keyword / month")
    print(f"Output    : {out_dir}")
    print(f"Resume    : {args.resume}\n")
    print("=" * 60)

    total_posts  = 0
    total_months = 0

    for year, month in months:
        label    = f"{year:04d}-{month:02d}"
        out_file = out_dir / f"bluesky_{label}.csv"

        # ── Resume: skip if already done ──────────────────────────────
        if args.resume and out_file.exists():
            size = out_file.stat().st_size
            print(f"[{label}] SKIP — {out_file.name} already exists ({size:,} bytes)")
            continue

        print(f"\n[{label}] Scraping …")
        rows = scrape_month(client, handle, password, keywords, year, month, args.limit)

        # ── Save CSV ──────────────────────────────────────────────────
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        total_posts  += len(rows)
        total_months += 1
        print(f"  --> Saved {len(rows):,} posts to {out_file.name}")

    print("\n" + "=" * 60)
    print(f"Done. {total_posts:,} posts across {total_months} month(s) saved to {out_dir}\n")


if __name__ == "__main__":
    main()
