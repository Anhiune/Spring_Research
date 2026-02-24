# Reddit scraping

This is a simple script to scrape text from Reddit. Sample output is in `test_data.csv`.

## Requirements

In a new virtual environment, install the needed packages by running the following:

```
pip install pandas
pip install spacy
pip install praw
python -m spacy download en_core_web_sm
```

## Configure subreddits and keywords

First, determine which subreddits to scrape from, and make a list of keywords. The keywords will determine which sentences the script keeps. The subreddits are in the file `top_20_reddit_communities.txt` (you can rename it depending on whether you actually want to use those communities or not) and the keywords are in `keywords_for_filtering.txt`.

## Configure Reddit API credentials

The script uses PRAW (Quickstart guide [here](https://praw.readthedocs.io/en/stable/getting_started/quick_start.html)), which requires you to provide your Reddit username (unfortunately required - you'll have to make an account if you don't already have one), as well as API access tokens (a client ID and a client secret). The quick start guide describes these. The code in `scrape.py` is currently configured to store the client ID and client secret in a separate file for more security, `~/.config/praw.ini`, but you can follow the method in the quick start guide and input it directly into the script itself if you prefer.

To sum up, the steps to configure these credentials are:

1. Create a Reddit account if you don't already have one, at [reddit.com](https://www.reddit.com).

2. Generate a client ID and client secret, following the "First steps" section [here](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps).

3. Put your client ID and client secret either into a `praw.ini` file (see the format in the `example_praw.ini` in this repo) or into the script itself (simpler but less secure).

## Run the script

Ideally, you should run the script in a separate `screen` or `tmux` session so that it can run continuously in the background. If you have trouble with this or am not sure what I'm talking about, let me know. You can just run:

```
python scrape.py
```