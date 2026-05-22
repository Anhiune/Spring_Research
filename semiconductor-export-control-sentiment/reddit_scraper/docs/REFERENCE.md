# Reddit Scraping - How the Original Repository Works

This document explains how the `simple-reddit-scraping` repository implements Reddit scraping. Use this as a reference for understanding the approach used in your custom implementation.

## Overview

The repository scrapes Reddit using PRAW (Python Reddit API Wrapper) to collect posts and comments, then uses spaCy for NLP-based sentence filtering based on keywords.

---

## Key Libraries

1. **PRAW** - Reddit API wrapper for authentication and data retrieval
2. **spaCy** - NLP for sentence segmentation and part-of-speech tagging
3. **pandas** - Data manipulation and CSV export

---

## Workflow

1. **Authentication**: Uses `praw.ini` for secure credential storage
2. **Fetching**: Gets 500 most recent posts per subreddit
3. **Segmentation**: Breaks text into sentences using spaCy
4. **Filtering**: Only keeps sentences with keywords (nouns/proper nouns only)
5. **Bot Detection**: Skips content containing "I am a bot"
6. **Duplicate Prevention**: Tracks processed IDs in text files
7. **Export**: Saves to timestamped CSV files every 1000 rows

---

## Data Structure

Each CSV row contains:
- `submission_id` - Reddit post ID
- `comment_id` - Comment ID (null for titles/selftext)
- `timestamp` - Unix timestamp
- `subreddit` - Subreddit name
- `text_type` - 'title', 'selftext', or 'comment'
- `text` - Filtered sentence

---

## NLP Filtering Logic

The keyword filtering is smart:
- Only considers **nouns** and **proper nouns** (not verbs, adjectives, etc.)
- Uses **lemmatization** (e.g., "robots" matches keyword "robot")
- Filters at **sentence level** (not post level) for precision
- Skips sentences over 100 words

---

## Error Handling

- 5 retry attempts for API calls
- 50 retry attempts for comment loading
- 5-second delays between retries
- Comprehensive logging to `scrape.log`

---

## Continuous Operation

Designed to run continuously in background:
- Incremental saving prevents data loss
- ID tracking enables resuming after interruptions
- Can run in `screen` or `tmux` sessions

For full implementation details, see the original source files in the `simple-reddit-scraping` directory.
