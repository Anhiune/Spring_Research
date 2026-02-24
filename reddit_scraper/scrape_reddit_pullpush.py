'''
Reddit Scraper — PullPush.io API
Fetches historical Reddit submissions and comments via the free PullPush API.
No Reddit API credentials required.

Features:
  - Query by keyword × subreddit with automatic pagination
  - Date-range support (--after / --before epoch or shorthand like 30d)
  - spaCy NLP sentence segmentation + noun/proper-noun keyword filtering
  - Bot detection, deduplication via ID tracking
  - Incremental CSV saves (every 1000 rows)
  - Continuous-loop or one-shot mode
  - Graceful Ctrl+C shutdown

Developed for UROP Research Spring 2026
'''

import argparse
import calendar
import os
import signal
import sys
import logging
from datetime import datetime
from time import sleep, time

import pandas as pd
import requests
import spacy
import re

# ──────────────────────────────────────────────
# Paths (relative to this script)
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_IDS_FILE = os.path.join(SCRIPT_DIR, "data", "submission_ids_recorded.txt")
COMMENT_IDS_FILE = os.path.join(SCRIPT_DIR, "data", "comment_ids_recorded.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# ──────────────────────────────────────────────
# Defaults (overridden by CLI args)
# ──────────────────────────────────────────────
API_BASE = "https://api.pullpush.io/reddit/search"
MAX_PAGE_SIZE = 100          # PullPush hard limit per request
MAX_PAGES = 50               # max pages per (subreddit, keyword) combo
REQUEST_DELAY = 3.0          # seconds between API calls (avoid 429 rate limits)
TIMEOUT = 30                 # HTTP timeout
MAX_SENTENCE_LENGTH = 100    # words
SAVE_THRESHOLD = 1000
LOOP_INTERVAL_MINUTES = 30

# ──────────────────────────────────────────────
# Runtime state
# ──────────────────────────────────────────────
nlp = None
KEYWORDS = []
subreddits = []
search_terms = []   # keywords used as API search queries
submission_ids_seen = set()
comment_ids_seen = set()
rows = []
COLUMNS = ['submission_id', 'comment_id', 'timestamp', 'subreddit', 'text_type', 'text']
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    logging.info("Shutdown signal received — finishing current work…")
    _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def setup_logging():
    log_path = os.path.join(SCRIPT_DIR, "scrape_pullpush.log")
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )

# ──────────────────────────────────────────────
# File helpers
# ──────────────────────────────────────────────

def load_lines(filepath, description="items"):
    """Load non-blank, non-comment lines from a text file."""
    items = []
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                val = line.strip()
                if val and not val.startswith("#"):
                    items.append(val)
        logging.info(f"Loaded {len(items)} {description} from {filepath}")
    else:
        logging.warning(f"File not found: {filepath}")
    return items


def load_id_set(filepath, description="IDs"):
    ids = set()
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    ids.add(s)
        logging.info(f"Loaded {len(ids)} previously scraped {description}")
    return ids

# ──────────────────────────────────────────────
# Init
# ──────────────────────────────────────────────

def init(args):
    global nlp, KEYWORDS, subreddits, search_terms
    global submission_ids_seen, comment_ids_seen

    logging.info("Loading spaCy model…")
    nlp = spacy.load("en_core_web_sm")
    nlp.enable_pipe("senter")
    logging.info("spaCy model loaded")

    KEYWORDS = [k.lower() for k in load_lines(
        os.path.join(SCRIPT_DIR, "config", "keywords.txt"), "filter keywords")]
    subreddits = load_lines(
        os.path.join(SCRIPT_DIR, "config", "subreddits.txt"), "subreddits")

    # Search terms can be a separate file or fall back to keywords
    search_file = os.path.join(SCRIPT_DIR, "config", "search_terms.txt")
    if os.path.exists(search_file):
        search_terms = load_lines(search_file, "search terms")
    else:
        # Use all keywords as search queries
        search_terms = list(KEYWORDS)
        logging.info(f"Using {len(search_terms)} keywords as API search terms "
                     "(create config/search_terms.txt to override)")

    submission_ids_seen = load_id_set(SUBMISSION_IDS_FILE, "submission IDs")
    comment_ids_seen = load_id_set(COMMENT_IDS_FILE, "comment IDs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "data"), exist_ok=True)

# ──────────────────────────────────────────────
# NLP helpers
# ──────────────────────────────────────────────

BOT_MARKERS = ["i am a bot", "i'm a bot", "this action was performed automatically"]

def _is_bot(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in BOT_MARKERS)


def sentence_segment(text: str):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def filter_for_keywords(sentences: list[str]) -> list[str]:
    """Keep sentences containing ≥1 keyword (as noun/proper noun, lemma-aware)."""
    kw_set = set(KEYWORDS)
    kept = []
    for sent in sentences:
        if len(sent.split()) > MAX_SENTENCE_LENGTH:
            continue
        doc = nlp(sent)
        for tok in doc:
            if tok.pos_ in ("NOUN", "PROPN"):
                if tok.text.lower() in kw_set or tok.lemma_.lower() in kw_set:
                    kept.append(sent)
                    break
    return kept


def segment_and_filter(text: str) -> list[str]:
    if not text or text == "[removed]" or text == "[deleted]":
        return []
    return filter_for_keywords(sentence_segment(text))

# ──────────────────────────────────────────────
# PullPush API
# ──────────────────────────────────────────────

_SHORTHAND_RE = re.compile(r'^(\d+)([smhd])$', re.IGNORECASE)
_UNIT_SECONDS = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}

def _parse_date_param(value):
    """Convert a date parameter to an epoch integer.

    Accepts:
      - None  → None
      - int / numeric string → epoch directly
      - shorthand like '30d', '12h' → epoch = now - delta
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = _SHORTHAND_RE.match(s)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        return int(time()) - n * _UNIT_SECONDS[unit]
    try:
        return int(float(s))
    except ValueError:
        logging.warning(f"Unrecognised date value '{value}' — ignoring")
        return None


def _api_get(endpoint: str, params: dict) -> dict | None:
    """Make a GET request to PullPush with retry."""
    url = f"{API_BASE}/{endpoint}/"
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = 5 * (attempt + 1)
                logging.warning(f"Rate limited (429). Waiting {wait}s…")
                sleep(wait)
            else:
                logging.warning(f"API error (attempt {attempt+1}): {e}")
                sleep(2 ** attempt)
        except Exception as e:
            logging.warning(f"API error (attempt {attempt+1}): {e}")
            sleep(2 ** attempt)
    return None


def fetch_submissions(subreddit: str, query: str,
                      after=None, before=None, max_pages=MAX_PAGES) -> list[dict]:
    """Paginate through submission results for one (subreddit, query) pair."""
    params = {
        'subreddit': subreddit,
        'q': query,
        'size': MAX_PAGE_SIZE,
        'sort': 'desc',
        'sort_type': 'created_utc',
    }
    if after is not None:
        params['after'] = after
    if before is not None:
        params['before'] = before

    all_items = []
    for page in range(max_pages):
        if _shutdown:
            break
        data = _api_get("submission", params)
        if data is None:
            break
        items = data.get('data', [])
        if not items:
            break
        all_items.extend(items)
        # Pagination: move the window backwards
        oldest = min(item['created_utc'] for item in items)
        params['before'] = oldest
        sleep(REQUEST_DELAY)
    return all_items


def fetch_comments(subreddit: str = None, query: str = None,
                   link_id: str = None,
                   after=None, before=None, max_pages=10) -> list[dict]:
    """Fetch comments — either by subreddit+query OR by link_id (submission)."""
    params = {'size': MAX_PAGE_SIZE, 'sort': 'desc', 'sort_type': 'created_utc'}
    if subreddit:
        params['subreddit'] = subreddit
    if query:
        params['q'] = query
    if link_id:
        params['link_id'] = link_id
    if after is not None:
        params['after'] = after
    if before is not None:
        params['before'] = before

    all_items = []
    for page in range(max_pages):
        if _shutdown:
            break
        data = _api_get("comment", params)
        if data is None:
            break
        items = data.get('data', [])
        if not items:
            break
        all_items.extend(items)
        oldest = min(item['created_utc'] for item in items)
        params['before'] = oldest
        sleep(REQUEST_DELAY)
    return all_items

# ──────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────

def process_submission_data(sub: dict):
    """Extract filtered sentences from a submission dict."""
    sid = sub.get('id')
    if not sid or sid in submission_ids_seen:
        return
    selftext = sub.get('selftext', '') or ''
    title = sub.get('title', '') or ''
    if _is_bot(selftext):
        submission_ids_seen.add(sid)
        return

    subreddit_name = sub.get('subreddit', 'unknown')
    ts = sub.get('created_utc', 0)

    for sent in segment_and_filter(title):
        rows.append({
            'submission_id': sid, 'comment_id': None,
            'timestamp': ts, 'subreddit': subreddit_name,
            'text_type': 'title', 'text': sent.strip(),
        })
    for sent in segment_and_filter(selftext):
        rows.append({
            'submission_id': sid, 'comment_id': None,
            'timestamp': ts, 'subreddit': subreddit_name,
            'text_type': 'selftext', 'text': sent.strip(),
        })
    submission_ids_seen.add(sid)


def process_comment_data(comment: dict):
    """Extract filtered sentences from a comment dict."""
    cid = comment.get('id')
    if not cid or cid in comment_ids_seen:
        return
    body = comment.get('body', '') or ''
    if _is_bot(body):
        comment_ids_seen.add(cid)
        return

    sid = comment.get('link_id', '').replace('t3_', '')
    subreddit_name = comment.get('subreddit', 'unknown')
    ts = comment.get('created_utc', 0)

    for sent in segment_and_filter(body):
        rows.append({
            'submission_id': sid, 'comment_id': cid,
            'timestamp': ts, 'subreddit': subreddit_name,
            'text_type': 'comment', 'text': sent.strip(),
        })
    comment_ids_seen.add(cid)

# ──────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────

def save_rows():
    global rows
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COLUMNS)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUTPUT_DIR, f"reddit_pullpush_{ts_str}.csv")
    df.to_csv(out, index=False)
    logging.info(f"Saved {len(rows)} rows → {out}")

    with open(SUBMISSION_IDS_FILE, "w", encoding="utf-8") as f:
        f.writelines(f"{i}\n" for i in submission_ids_seen)
    with open(COMMENT_IDS_FILE, "w", encoding="utf-8") as f:
        f.writelines(f"{i}\n" for i in comment_ids_seen)
    logging.info("Updated ID tracking files")
    rows = []

# ──────────────────────────────────────────────
# Scraping round
# ──────────────────────────────────────────────

def scrape_round(after_val, before_val, do_comments, max_pages):
    """One full pass over all subreddits × search terms for a given date window."""
    global _shutdown

    for sub_name in subreddits:
        if _shutdown:
            break
        logging.info(f"═══ r/{sub_name} ═══")

        for term in search_terms:
            if _shutdown:
                break
            logging.info(f"  Searching submissions for '{term}'…")
            subs = fetch_submissions(sub_name, term,
                                     after=after_val, before=before_val,
                                     max_pages=max_pages)
            logging.info(f"  Retrieved {len(subs)} submissions")
            for s in subs:
                process_submission_data(s)

            if do_comments:
                logging.info(f"  Searching comments for '{term}'…")
                cmts = fetch_comments(subreddit=sub_name, query=term,
                                      after=after_val, before=before_val,
                                      max_pages=max_pages)
                logging.info(f"  Retrieved {len(cmts)} comments")
                for c in cmts:
                    process_comment_data(c)

            if len(rows) >= SAVE_THRESHOLD:
                save_rows()

        logging.info(f"═══ Done: r/{sub_name} ═══")

    save_rows()

# ──────────────────────────────────────────────
# Month-by-month helpers
# ──────────────────────────────────────────────

def _generate_months(start_str: str, end_str: str):
    """Yield (year, month) tuples from 'YYYY-MM' start to 'YYYY-MM' end inclusive."""
    sy, sm = map(int, start_str.split("-"))
    ey, em = map(int, end_str.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _month_epoch_range(year: int, month: int):
    """Return (after_epoch, before_epoch) for a given year-month."""
    start_dt = datetime(year, month, 1)
    # first day of next month
    if month == 12:
        end_dt = datetime(year + 1, 1, 1)
    else:
        end_dt = datetime(year, month + 1, 1)
    return int(start_dt.timestamp()), int(end_dt.timestamp())

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    global _shutdown

    p = argparse.ArgumentParser(
        description="Reddit Scraper — PullPush.io API (no credentials needed)")

    # ── Date range: month-by-month batch mode ──
    p.add_argument("--start", default=None,
                   help="Start month in YYYY-MM format (e.g. 2023-01). "
                        "Enables month-by-month batch mode.")
    p.add_argument("--end", default=None,
                   help="End month in YYYY-MM format (e.g. 2026-02). "
                        "Defaults to current month if --start is given.")

    # ── Date range: epoch / shorthand mode ──
    p.add_argument("--after", default=None,
                   help="Only results after this date. Epoch timestamp or "
                        "shorthand like '30d'. Ignored if --start is used.")
    p.add_argument("--before", default=None,
                   help="Only results before this date. Same format as --after. "
                        "Ignored if --start is used.")

    # ── Other options ──
    p.add_argument("--comments", action="store_true",
                   help="Also fetch comments (slower, more API calls). "
                        "Default: submissions only.")
    p.add_argument("--max-pages", type=int, default=MAX_PAGES,
                   help=f"Max API pages per (subreddit, keyword) pair "
                        f"(default: {MAX_PAGES}; each page = {MAX_PAGE_SIZE} results)")
    p.add_argument("--once", action="store_true",
                   help="Run once then exit (instead of continuous loop). "
                        "Implied when using --start.")
    p.add_argument("--interval", type=int, default=LOOP_INTERVAL_MINUTES,
                   help=f"Minutes between rounds in continuous mode "
                        f"(default: {LOOP_INTERVAL_MINUTES})")
    args = p.parse_args()

    setup_logging()
    init(args)

    # ──────────────────────────────────────────
    # Mode A: Month-by-month batch
    # ──────────────────────────────────────────
    if args.start:
        end_month = args.end or datetime.now().strftime("%Y-%m")
        months = list(_generate_months(args.start, end_month))

        logging.info("=" * 60)
        logging.info("PullPush Reddit Scraper — BATCH MODE (month-by-month)")
        logging.info(f"  Range        : {args.start} → {end_month} ({len(months)} months)")
        logging.info(f"  Subreddits   : {len(subreddits)}")
        logging.info(f"  Search terms : {len(search_terms)}")
        logging.info(f"  Filter kw    : {len(KEYWORDS)}")
        logging.info(f"  Comments     : {'yes' if args.comments else 'no'}")
        logging.info(f"  Max pages    : {args.max_pages}")
        logging.info("=" * 60)

        for i, (year, month) in enumerate(months, 1):
            if _shutdown:
                break
            label = f"{year}-{month:02d}"
            after_ep, before_ep = _month_epoch_range(year, month)
            logging.info(f"")
            logging.info(f"{'─'*50}")
            logging.info(f"  Month {i}/{len(months)}: {label}")
            logging.info(f"  Epoch window: {after_ep} → {before_ep}")
            logging.info(f"{'─'*50}")

            scrape_round(after_ep, before_ep, args.comments, args.max_pages)

            logging.info(f"  ✓ Finished {label}")

        save_rows()
        logging.info("Batch scraping complete.")
        return

    # ──────────────────────────────────────────
    # Mode B: Single run / continuous loop
    # ──────────────────────────────────────────
    after_val = _parse_date_param(args.after)
    before_val = _parse_date_param(args.before)

    logging.info("=" * 60)
    logging.info("PullPush Reddit Scraper")
    logging.info(f"  Mode         : {'one-time' if args.once else 'continuous loop'}")
    logging.info(f"  Subreddits   : {len(subreddits)}")
    logging.info(f"  Search terms : {len(search_terms)}")
    logging.info(f"  Filter kw    : {len(KEYWORDS)}")
    logging.info(f"  Comments     : {'yes' if args.comments else 'no'}")
    logging.info(f"  After        : {args.after or 'all time'}")
    logging.info(f"  Before       : {args.before or 'now'}")
    logging.info(f"  Max pages    : {args.max_pages}")
    if not args.once:
        logging.info(f"  Interval     : {args.interval} min")
    logging.info("=" * 60)

    round_num = 0
    while True:
        round_num += 1
        logging.info(f"▶ Round {round_num} — {datetime.now().isoformat()}")
        scrape_round(after_val, before_val, args.comments, args.max_pages)
        logging.info(f"▶ Round {round_num} complete")

        if args.once or _shutdown:
            break

        logging.info(f"Sleeping {args.interval} minutes…")
        for _ in range(args.interval * 60):
            if _shutdown:
                break
            sleep(1)

    save_rows()
    logging.info("Scraper shut down.")


if __name__ == "__main__":
    main()
