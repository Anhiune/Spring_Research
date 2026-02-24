'''
Simple script to scrape Reddit submissions and comments from specified subreddits, 
filter sentences based on keywords, and save the results to CSV files.
Author: Elizabeth Hoefer
'''

import pandas as pd
import praw
import spacy
import logging
from datetime import datetime
from time import sleep

#### Logging setup
handler_stream = logging.StreamHandler()
handler_stream.setLevel(logging.INFO)

handler_file = logging.FileHandler("scrape.log")
handler_file.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler_stream, handler_file],
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

for logger_name in ("praw", "prawcore"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler_stream)
    logger.addHandler(handler_file)

#### Preliminaries
YOUR_REDDIT_USERNAME = "your_reddit_username_here"
MAX_SENTENCE_LENGTH = 100  # maximum length of a sentence to consider (number of words)

reddit = praw.Reddit(
    site_name="bot1", # sensitive info (ie reddit password) should be in $HOME/.config/praw.ini
    user_agent=f"mac:scraping:v0.0 (by /u/{YOUR_REDDIT_USERNAME})",
)
nlp = spacy.load("en_core_web_sm")
nlp.enable_pipe("senter")

KEYWORDS = [] 
with open("keywords_for_filtering.txt") as f:
    for line in f:
        KEYWORDS.append(line.strip())

submission_ids_already_scraped = []
comment_ids_already_scraped = []
with open("submission_ids_recorded.txt") as f:
    for line in f:
        submission_ids_already_scraped.append(line.strip())
submission_ids_already_scraped = list(set(submission_ids_already_scraped)) # deduplicate

with open("comment_ids_recorded.txt") as f:
    for line in f:
        comment_ids_already_scraped.append(line.strip())
comment_ids_already_scraped = list(set(comment_ids_already_scraped)) # deduplicate

rows = [] # prep for DataFrame
# submission_id: praw/reddit submission id (submission.id)
# comment_id (optional): praw/reddit comment id (comment.id)
# timestamp: unix timestamp from praw (submission.created_utc)
# subreddit: e.g. 'learnpython' 
# text_type: 'title', 'selftext', or 'comment'
# text: the actual text
column_labels = ['submission_id', 'comment_id', 'timestamp', 'subreddit', 'text_type', 'text']

subreddits = []
with open("top_20_reddit_communities.txt") as f:
    for line in f:
        subreddits.append(line.strip())

#### Helpers

def sentence_segment(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def filter_sentences_for_keywords(sentences, keywords):
    filtered_sentences = []
    for sent in sentences:
        if len(sent.split()) > MAX_SENTENCE_LENGTH:
            continue # skip long sentences
        doc = nlp(sent)
        for token in doc:
            if token.pos_ == "PROPN" or token.pos_ == "NOUN": # remove this if-statement if you also want to consider words that are not nouns
                if token.text.lower() in keywords or token.lemma_.lower() in keywords:
                    filtered_sentences.append(sent)
                    break
    return filtered_sentences

def segment_and_filter(text, keywords):
    sentences = sentence_segment(text)
    filtered = filter_sentences_for_keywords(sentences, keywords)
    return filtered

def process_submission(submission):
    logging.info(f"Processing submission {submission.id}")

    # check id
    if submission.id in submission_ids_already_scraped:
        logging.info(f"Skipping already scraped submission {submission.id}")
        # return
    else:
        if submission.is_self: # is_self means it's a text post, not a link
            # get sentences for title, body, and comments
            if 'I am a bot' in submission.selftext:
                logging.info(f"Skipping bot post {submission.id}")
                submission_ids_already_scraped.append(submission.id)
                return
            title_sentences = segment_and_filter(submission.title, KEYWORDS)
            body_sentences = segment_and_filter(submission.selftext, KEYWORDS)
            if title_sentences:
                for sent in title_sentences:
                    rows.append(
                        {
                            'submission_id': submission.id, 'comment_id': None,
                            'timestamp': submission.created_utc, 'subreddit': submission.subreddit.display_name,
                            'text_type': 'title', 'text': sent.strip()
                        }
                    )
            if body_sentences:
                for sent in body_sentences:
                    rows.append(
                        {
                            'submission_id': submission.id, 'comment_id': None,
                            'timestamp': submission.created_utc, 'subreddit': submission.subreddit.display_name,
                            'text_type': 'selftext', 'text': sent.strip()
                        }
                    )
            submission_ids_already_scraped.append(submission.id)
    # try to get comments even if submission was already scraped, in case new comments were added
    for i in range(50): # try 50 times
        try:
            submission.comments.replace_more()
            break
        except Exception as e:
            logging.error(f"Error with replace_more(): {e}. Sleeping 5 seconds and retrying.")
            sleep(5)

    for comment in submission.comments:
        if comment.id in comment_ids_already_scraped:
            logging.info(f"Skipping already scraped comment {comment.id}")
            continue
        if 'I am a bot' in comment.body:
            logging.info(f"Skipping bot comment {comment.id}")
            comment_ids_already_scraped.append(comment.id)
            continue
        comment_sentences = segment_and_filter(comment.body, KEYWORDS)
        if comment_sentences:
            for sent in comment_sentences:
                rows.append(
                    {
                        'submission_id': submission.id, 'comment_id': comment.id,
                        'timestamp': comment.created_utc, 'subreddit': submission.subreddit.display_name,
                        'text_type': 'comment', 'text': sent.strip()
                    }
                )
        comment_ids_already_scraped.append(comment.id)
    logging.info(f"Finished processing submission {submission.id}")
    return

#### Main loop
for subreddit_name in subreddits:
    logging.info(f"Starting subreddit {subreddit_name}")
    subreddit = reddit.subreddit(subreddit_name)
    for i in range(5): # try 5 times
        try:
            submissions = subreddit.new(limit=500)
            break
        except Exception as e:
            logging.error(f'Error getting new submissions for {subreddit_name}: {e}')
            logging.info('Sleeping 5 seconds and trying again.')
            sleep(5)
    for submission in submissions:
        process_submission(submission)
    
    # check length of rows and save if >= 1000
    if len(rows) >= 1000:
        df = pd.DataFrame(rows, columns=column_labels)
        # get current timestamp for filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}.csv"

        df.to_csv(f"reddit_{timestamp_str}.csv", index=False) # save locally with name indicating time right now (so there will be many csvs with different timestamps)
        logging.info(f"Saved {len(rows)} rows to reddit_{timestamp_str}.csv")
        # update recorded ids files
        with open("submission_ids_recorded.txt", "w") as f_sub:
            for sub_id in set(submission_ids_already_scraped):
                f_sub.write(f"{sub_id}\n")
        with open("comment_ids_recorded.txt", "w") as f_com:
            for com_id in set(comment_ids_already_scraped):
                f_com.write(f"{com_id}\n")
        logging.info(f"Updated recorded IDs files.")

        rows = [] # reset rows

    logging.info(f"Finished subreddit {subreddit_name}")