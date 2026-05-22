'''
Reddit Scraper - Continuous Loop Implementation
Scrapes Reddit submissions and comments from specified subreddits,
filters sentences based on keywords using spaCy NLP, and saves to CSV files.

Features:
  - PRAW Reddit API integration
  - spaCy NLP for sentence segmentation and keyword filtering (nouns/proper nouns)
  - Bot detection and filtering
  - Duplicate prevention via ID tracking
  - Incremental CSV saving (every 1000 rows)
  - Continuous loop mode with configurable interval
  - Robust error handling and retry logic
  - Comprehensive logging

Based on: simple-reddit-scraping by Elizabeth Hoefer
Customized for: UROP Research Spring 2026
'''

import argparse
import pandas as pd
import praw
import spacy
import logging
import signal
import sys
from datetime import datetime
from time import sleep
import os

# ──────────────────────────────────────────────
# Globals (set during init)
# ──────────────────────────────────────────────
nlp = None
reddit = None
KEYWORDS = []
subreddits = []
submission_ids_already_scraped = set()
comment_ids_already_scraped = set()
rows = []
COLUMN_LABELS = ['submission_id', 'comment_id', 'timestamp', 'subreddit', 'text_type', 'text']

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_IDS_FILE = os.path.join(SCRIPT_DIR, "data", "submission_ids_recorded.txt")
COMMENT_IDS_FILE = os.path.join(SCRIPT_DIR, "data", "comment_ids_recorded.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Configuration defaults
MAX_SENTENCE_LENGTH = 100       # max words per sentence to consider
POSTS_PER_SUBREDDIT = 500       # how many new posts to fetch per subreddit
SAVE_THRESHOLD = 1000           # save CSV every N rows
LOOP_INTERVAL_MINUTES = 30      # minutes between scraping rounds in continuous mode
REPLACE_MORE_RETRIES = 50       # retries for loading "more comments"
API_RETRIES = 5                 # retries for API calls
RETRY_DELAY = 5                 # seconds between retries

# Graceful shutdown flag
_shutdown = False


def _signal_handler(sig, frame):
    """Handle Ctrl+C gracefully — save remaining data before exiting."""
    global _shutdown
    logging.info("Shutdown signal received. Finishing current work and saving…")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def setup_logging():
    log_file = os.path.join(SCRIPT_DIR, "scrape.log")
    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.INFO)

    handler_file = logging.FileHandler(log_file)
    handler_file.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler_stream, handler_file],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for logger_name in ("praw", "prawcore"):
        lgr = logging.getLogger(logger_name)
        lgr.setLevel(logging.INFO)
        lgr.addHandler(handler_stream)
        lgr.addHandler(handler_file)

# ──────────────────────────────────────────────
# Initialisation helpers
# ──────────────────────────────────────────────

def load_list_from_file(filepath, description="items"):
    """Load a list from a text file, ignoring comments (#) and blank lines."""
    items = []
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                item = line.strip()
                if item and not item.startswith("#"):
                    items.append(item)
        logging.info(f"Loaded {len(items)} {description} from {filepath}")
    else:
        logging.warning(f"File not found: {filepath}")
    return items


def load_id_set(filepath, description="IDs"):
    """Load a set of IDs from a text file."""
    ids = set()
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    ids.add(stripped)
        logging.info(f"Loaded {len(ids)} previously scraped {description}")
    return ids


def init(username: str):
    """Initialise Reddit client, spaCy model, keywords, subreddits, and ID trackers."""
    global nlp, reddit, KEYWORDS, subreddits
    global submission_ids_already_scraped, comment_ids_already_scraped

    # Reddit
    reddit = praw.Reddit(
        site_name="bot1",
        user_agent=f"windows:reddit_scraper:v2.0 (by /u/{username})",
    )
    logging.info("Reddit API client initialised")

    # spaCy
    logging.info("Loading spaCy model…")
    nlp = spacy.load("en_core_web_sm")
    nlp.enable_pipe("senter")
    logging.info("spaCy model loaded")

    # Keywords & subreddits
    KEYWORDS = load_list_from_file(
        os.path.join(SCRIPT_DIR, "config", "keywords.txt"), "keywords"
    )
    subreddits = load_list_from_file(
        os.path.join(SCRIPT_DIR, "config", "subreddits.txt"), "subreddits"
    )

    # ID tracking
    submission_ids_already_scraped = load_id_set(SUBMISSION_IDS_FILE, "submission IDs")
    comment_ids_already_scraped = load_id_set(COMMENT_IDS_FILE, "comment IDs")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "data"), exist_ok=True)

# ──────────────────────────────────────────────
# NLP helpers
# ──────────────────────────────────────────────

def sentence_segment(text):
    """Segment text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def filter_sentences_for_keywords(sentences, keywords):
    """Keep sentences containing at least one keyword (as noun / proper noun)."""
    keywords_lower = {k.lower() for k in keywords}
    filtered = []
    for sent in sentences:
        if len(sent.split()) > MAX_SENTENCE_LENGTH:
            continue
        doc = nlp(sent)
        for token in doc:
            if token.pos_ in ("PROPN", "NOUN"):
                if token.text.lower() in keywords_lower or token.lemma_.lower() in keywords_lower:
                    filtered.append(sent)
                    break
    return filtered


def segment_and_filter(text, keywords):
    """Segment text then filter by keywords."""
    return filter_sentences_for_keywords(sentence_segment(text), keywords)

# ──────────────────────────────────────────────
# Saving helpers
# ──────────────────────────────────────────────

def save_rows():
    """Save accumulated rows to a timestamped CSV and update ID tracking files."""
    global rows
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COLUMN_LABELS)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"reddit_{timestamp_str}.csv")
    df.to_csv(output_file, index=False)
    logging.info(f"Saved {len(rows)} rows → {output_file}")

    # Persist ID trackers
    with open(SUBMISSION_IDS_FILE, "w", encoding="utf-8") as f:
        for sid in submission_ids_already_scraped:
            f.write(f"{sid}\n")
    with open(COMMENT_IDS_FILE, "w", encoding="utf-8") as f:
        for cid in comment_ids_already_scraped:
            f.write(f"{cid}\n")
    logging.info("Updated recorded ID files")

    rows = []  # reset

# ──────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────

BOT_MARKERS = ["I am a bot", "I'm a bot", "this action was performed automatically"]


def _is_bot_text(text: str) -> bool:
    text_lower = text.lower()
    return any(marker.lower() in text_lower for marker in BOT_MARKERS)


def process_submission(submission):
    """Process a single submission: title, selftext, and comments."""
    global _shutdown
    if _shutdown:
        return

    logging.info(f"Processing submission {submission.id} in r/{submission.subreddit.display_name}")

    # ---- Title & selftext ----
    if submission.id not in submission_ids_already_scraped:
        if submission.is_self:
            if _is_bot_text(submission.selftext):
                logging.info(f"Skipping bot post {submission.id}")
                submission_ids_already_scraped.add(submission.id)
                return

            for sent in segment_and_filter(submission.title, KEYWORDS):
                rows.append({
                    'submission_id': submission.id,
                    'comment_id': None,
                    'timestamp': submission.created_utc,
                    'subreddit': submission.subreddit.display_name,
                    'text_type': 'title',
                    'text': sent.strip(),
                })

            for sent in segment_and_filter(submission.selftext, KEYWORDS):
                rows.append({
                    'submission_id': submission.id,
                    'comment_id': None,
                    'timestamp': submission.created_utc,
                    'subreddit': submission.subreddit.display_name,
                    'text_type': 'selftext',
                    'text': sent.strip(),
                })
        else:
            # Link post — still process the title
            for sent in segment_and_filter(submission.title, KEYWORDS):
                rows.append({
                    'submission_id': submission.id,
                    'comment_id': None,
                    'timestamp': submission.created_utc,
                    'subreddit': submission.subreddit.display_name,
                    'text_type': 'title',
                    'text': sent.strip(),
                })

        submission_ids_already_scraped.add(submission.id)
    else:
        logging.debug(f"Submission {submission.id} already scraped — checking for new comments")

    # ---- Comments ----
    for attempt in range(REPLACE_MORE_RETRIES):
        try:
            submission.comments.replace_more(limit=None)
            break
        except Exception as e:
            logging.error(f"replace_more() error (attempt {attempt+1}): {e}")
            sleep(RETRY_DELAY)

    for comment in submission.comments.list():
        if _shutdown:
            return
        if comment.id in comment_ids_already_scraped:
            continue
        try:
            body = comment.body
        except AttributeError:
            comment_ids_already_scraped.add(comment.id)
            continue

        if _is_bot_text(body):
            comment_ids_already_scraped.add(comment.id)
            continue

        for sent in segment_and_filter(body, KEYWORDS):
            rows.append({
                'submission_id': submission.id,
                'comment_id': comment.id,
                'timestamp': comment.created_utc,
                'subreddit': submission.subreddit.display_name,
                'text_type': 'comment',
                'text': sent.strip(),
            })

        comment_ids_already_scraped.add(comment.id)

    logging.info(f"Finished submission {submission.id} | buffer={len(rows)} rows")

# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def scrape_round(limit: int):
    """Run one full round: iterate every subreddit and save incrementally."""
    global _shutdown
    for subreddit_name in subreddits:
        if _shutdown:
            break
        logging.info(f"── Subreddit: r/{subreddit_name} (limit={limit}) ──")
        try:
            sub = reddit.subreddit(subreddit_name)
            submissions = None
            for attempt in range(API_RETRIES):
                try:
                    submissions = list(sub.new(limit=limit))
                    break
                except Exception as e:
                    logging.error(f"Error fetching r/{subreddit_name} (attempt {attempt+1}): {e}")
                    sleep(RETRY_DELAY)

            if submissions is None:
                logging.error(f"Giving up on r/{subreddit_name} after {API_RETRIES} attempts")
                continue

            logging.info(f"Fetched {len(submissions)} submissions from r/{subreddit_name}")

            for submission in submissions:
                if _shutdown:
                    break
                process_submission(submission)

                # Incremental save
                if len(rows) >= SAVE_THRESHOLD:
                    save_rows()

        except Exception as e:
            logging.error(f"Unexpected error with r/{subreddit_name}: {e}", exc_info=True)

        logging.info(f"── Done: r/{subreddit_name} ──")

    # Save whatever remains after the round
    save_rows()


def main():
    global _shutdown

    parser = argparse.ArgumentParser(description="Reddit Scraper — PRAW + spaCy")
    parser.add_argument("--username", default="your_reddit_username_here",
                        help="Your Reddit username (used in user-agent string)")
    parser.add_argument("--limit", type=int, default=POSTS_PER_SUBREDDIT,
                        help=f"Posts to fetch per subreddit (default: {POSTS_PER_SUBREDDIT})")
    parser.add_argument("--once", action="store_true",
                        help="Run once then exit (instead of continuous loop)")
    parser.add_argument("--interval", type=int, default=LOOP_INTERVAL_MINUTES,
                        help=f"Minutes between rounds in continuous mode (default: {LOOP_INTERVAL_MINUTES})")
    args = parser.parse_args()

    setup_logging()
    init(args.username)

    logging.info("=" * 60)
    logging.info("Reddit Scraper started")
    logging.info(f"  Mode       : {'one-time' if args.once else 'continuous loop'}")
    logging.info(f"  Subreddits : {len(subreddits)}")
    logging.info(f"  Keywords   : {len(KEYWORDS)}")
    logging.info(f"  Limit      : {args.limit} posts/subreddit")
    if not args.once:
        logging.info(f"  Interval   : {args.interval} minutes")
    logging.info("=" * 60)

    round_number = 0
    while True:
        round_number += 1
        logging.info(f"▶ Round {round_number} starting at {datetime.now().isoformat()}")
        scrape_round(limit=args.limit)
        logging.info(f"▶ Round {round_number} complete")

        if args.once or _shutdown:
            break

        logging.info(f"Sleeping {args.interval} minutes until next round…")
        for _ in range(args.interval * 60):
            if _shutdown:
                break
            sleep(1)

    # Final cleanup save
    save_rows()
    logging.info("Scraper shut down cleanly.")


if __name__ == "__main__":
    main()
