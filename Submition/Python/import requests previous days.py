import requests
import pandas as pd
import time
from datetime import datetime, timezone

# === Step 1: Set Up Authentication ===
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAPid3gEAAAAAS0zZ2SYinxzXXbIT8%2FA8xtLK3OE%3Dh1nU5qX2jfcYNVTEjNexSNyngyelVorQz3W3ripPnU6zNBFaMn'  # Replace with your token

headers = {
    'Authorization': f'Bearer {BEARER_TOKEN}',
    'User-Agent': 'TeslaSentimentFetcher'
}

# === Step 2: Set Parameters for August 13 Only ===
search_url = "https://api.twitter.com/2/tweets/search/recent"
query = '(Tesla OR "Elon Musk" OR "Tesla stock" OR "Tesla economy" OR "Tesla government") lang:en -is:retweet'

start_time = "2025-08-20T08:00:00Z"
end_time   = "2025-08-20T10:00:00Z"
max_results = 100  # max allowed per page

def fetch_aug13_tweets(min_likes=0, max_pages=2):
    tweets = []
    next_token = None
    total_raw = 0

    for page in range(max_pages):
        params = {
            'query': query,
            'max_results': max_results,
            'start_time': start_time,
            'end_time': end_time,
            'tweet.fields': 'created_at,public_metrics,lang,author_id'
        }
        if next_token:
            params['next_token'] = next_token

        response = requests.get(search_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            break

        json_data = response.json()
        data = json_data.get('data', [])
        total_raw += len(data)
        print(f"📄 Page {page+1}: Retrieved {len(data)} tweets")

        for tweet in data:
            metrics = tweet['public_metrics']
            if metrics['like_count'] >= min_likes:
                tweets.append({
                    'date': tweet['created_at'][:10],
                    'tweet_id': tweet['id'],
                    'author_id': tweet['author_id'],
                    'text': tweet['text'][:100],
                    'likes': metrics['like_count'],
                    'retweets': metrics['retweet_count'],
                    'replies': metrics['reply_count'],
                    'quotes': metrics['quote_count'],
                })

        next_token = json_data.get('meta', {}).get('next_token')
        if not next_token:
            break
        time.sleep(1.2)

    print(f"\n🧮 Total raw tweets: {total_raw} | Saved after filter: {len(tweets)}")
    if tweets:
        print(f"🔎 Sample tweet: {tweets[0]['text']}...")

    return tweets

# === Save to Excel ===
def save_to_excel(df):
    filename = f"tesla_aug20_2025.xlsx"
    try:
        df.to_excel(filename, index=False)
        print(f"\n✅ Saved {len(df)} tweets to: {filename}")
    except ModuleNotFoundError:
        print("❌ Please install openpyxl: pip install openpyxl")

# === Run ===
if __name__ == "__main__":
    df = pd.DataFrame(fetch_aug13_tweets())
    save_to_excel(df)
