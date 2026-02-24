import pandas as pd

# 1. Load your data (adjust file path as needed).
df = pd.read_csv(r"C:\Users\hoang\Documents\Summer Research\python (locally download file)\tesla_tweets_with_sentiment.csv")

# 2. Convert 'date' column to datetime.
#    `errors='coerce'` converts invalid strings to NaT (missing value):contentReference[oaicite:0]{index=0}.
df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

# 3. Drop rows where 'date' could not be parsed.
df = df[df['date'].notnull()]

# 4. Define which columns hold sentiment scores.
emotion_cols = [
    'anger', 'anticipation', 'disgust', 'fear',
    'joy', 'sadness', 'surprise', 'trust',
    'positive', 'negative'
]

# 5. Group by date and compute the mean sentiment for each day.
daily_sentiment = df.groupby('date')[emotion_cols].mean().reset_index()

# 6. Save the daily sentiment time series.
output_path = r"C:\Users\hoang\Documents\Summer Research\python (locally download file)\daily_tesla_sentiment.csv"
daily_sentiment.to_csv(output_path, index=False)

print("✅ Daily sentiment time series created and saved to:", output_path)
