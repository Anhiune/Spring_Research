import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

# === Step 1: Set Your Bearer Token ===
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAPid3gEAAAAA5%2F%2F0Br%2B8FAqJcWHK3%2BXX8nxfYac%3DVU56pVY05p9VOFx7kRBtsgtcVWGGOuSSV23b1g2VOBMmRwVM4l'  # ← Replace with your real token

headers = {
    'Authorization': f'Bearer {BEARER_TOKEN}',
    'User-Agent': 'TeslaSentimentFetcher'
}

# === Step 2: Define Parameters ===
search_url = "https://api.twitter.com/2/tweets/search/recent"
query = '(Tesla OR "Elon Musk" OR "Tesla stock" OR "Tesla economy" OR "Tesla government") lang:en -is:retweet'
max_results = 100

# === Step 3: Fetch Tweets for Each Day with Debug Logging ===
def fetch_tweets_for_day(start_time, end_time, min_likes=0, max_pages=5):
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
                    'text': tweet['text'][:100],  # truncate for preview
                    'likes': metrics['like_count'],
                    'retweets': metrics['retweet_count'],
                    'replies': metrics['reply_count'],
                    'quotes': metrics['quote_count'],
                })

        next_token = json_data.get('meta', {}).get('next_token')
        if not next_token:
            break
        time.sleep(1.2)  # avoid rate limits

    print(f"🧮 Total raw tweets: {total_raw} | Saved after filter: {len(tweets)}")
    if tweets:
        print(f"🔎 Sample tweet: {tweets[0]['text'][:100]}...")

    return tweets

# === Step 4: Loop Over Days ===
def collect_7_days_of_tweets():
    all_data = []
    for i in range(7):
        end_day = datetime.now(timezone.utc) - timedelta(days=i)
        start_day = end_day - timedelta(days=1)

        if (datetime.now(timezone.utc) - end_day).total_seconds() < 600:
            print(f"⏭ Skipping {end_day.date()} – too close to now")
            continue

        end_time = end_day.isoformat(timespec='seconds').replace('+00:00', 'Z')
        start_time = start_day.isoformat(timespec='seconds').replace('+00:00', 'Z')
        print(f"\n📅 Fetching tweets for: {start_day.date()}")

        day_tweets = fetch_tweets_for_day(start_time, end_time, min_likes=0)
        all_data.extend(day_tweets)

    return pd.DataFrame(all_data)

# === Step 5: Save to Excel ===
def save_to_excel(df):
    today_str = datetime.today().strftime('%Y-%m-%d')
    filename = f"tesla_7days_debugged_{today_str}.xlsx"
    try:
        df.to_excel(filename, index=False)
        print(f"\n✅ Saved {len(df)} tweets to: {filename}")
    except ModuleNotFoundError:
        print("❌ Please install openpyxl: pip install openpyxl")

# === Run Everything ===
if __name__ == "__main__":
    df_all = collect_7_days_of_tweets()
    save_to_excel(df_all)
