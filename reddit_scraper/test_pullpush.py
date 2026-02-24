'''
Test script for PullPush API scraper
Tests basic functionality with minimal data
'''

import pandas as pd
import requests
import spacy
import logging
from datetime import datetime
import os

# Simple logging setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Test configuration
PULLPUSH_API_BASE = "https://api.pullpush.io/reddit/search"
TEST_SUBREDDIT = "science"
TEST_KEYWORD = "research"
DAYS_BACK = 1  # Just test last day

logging.info("Testing PullPush API connection...")

# Test 1: Fetch a few submissions
url = f"{PULLPUSH_API_BASE}/submission/"
params = {
    'subreddit': TEST_SUBREDDIT,
    'q': TEST_KEYWORD,
    'size': 5,
    'after': f'{DAYS_BACK}d',
    'sort': 'desc'
}

try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    submissions = data.get('data', [])
    
    logging.info(f"✓ Successfully fetched {len(submissions)} submissions")
    
    if submissions:
        sample = submissions[0]
        logging.info(f"  Sample submission ID: {sample.get('id')}")
        logging.info(f"  Title: {sample.get('title', '')[:50]}...")
        logging.info(f"  Subreddit: {sample.get('subreddit')}")
        logging.info(f"  Timestamp: {sample.get('created_utc')}")
    
except Exception as e:
    logging.error(f"✗ API test failed: {e}")
    exit(1)

# Test 2: Check spaCy
try:
    logging.info("Testing spaCy NLP model...")
    nlp = spacy.load("en_core_web_sm")
    nlp.enable_pipe("senter")
    
    test_text = "This is a research study. Data analysis is important."
    doc = nlp(test_text)
    sentences = [sent.text for sent in doc.sents]
    
    logging.info(f"✓ spaCy working - found {len(sentences)} sentences")
    
except Exception as e:
    logging.error(f"✗ spaCy test failed: {e}")
    logging.error("  Run: python -m spacy download en_core_web_sm")
    exit(1)

# Test 3: Check directories
logging.info("Checking directory structure...")
required_dirs = ['data', 'output', 'config']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        logging.info(f"✓ {dir_name}/ exists")
    else:
        logging.warning(f"✗ {dir_name}/ missing - creating it")
        os.makedirs(dir_name, exist_ok=True)

logging.info("\n" + "="*50)
logging.info("All tests passed! ✓")
logging.info("The scraper is ready to use.")
logging.info("Run: python scrape_reddit_pullpush.py")
logging.info("="*50)
