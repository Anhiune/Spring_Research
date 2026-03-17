#!/usr/bin/env python3
"""
example_sentiment_analysis.py – Example analysis workflows using sentiment_scoring outputs
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def example_1_compare_methods():
    """Compare NRC and HuggingFace sentiment on same texts."""
    print("=" * 70)
    print("Example 1: Compare NRC vs HuggingFace Sentiments")
    print("=" * 70)

    # Load scored data
    df = pd.read_csv('bluesky_all_with_sentiment.csv')

    # Select subset with both methods
    sample = df[['text_clean', 'nrc_net_sentiment', 'nrc_dominant_emotion',
                 'hf_dominant_sentiment']].head(10)

    print("\nSide-by-side comparison:")
    print(sample.to_string(index=False))

    # Calculate agreement
    nrc_positive = df['nrc_net_sentiment'] > 0
    hf_positive = df['hf_dominant_sentiment'].isin(['positive', 'very_positive'])
    agreement = (nrc_positive == hf_positive).mean()

    print(f"\nMethodAgreement Rate: {agreement:.1%}")
    print(f"Both positive: {(nrc_positive & hf_positive).sum()} texts")
    print(f"Both negative: {(~nrc_positive & ~hf_positive).sum()} texts")
    print(f"Disagreement: {(nrc_positive != hf_positive).sum()} texts")


def example_2_temporal_trends():
    """Analyze sentiment trends over time."""
    print("\n" + "=" * 70)
    print("Example 2: Temporal Sentiment Trends")
    print("=" * 70)

    # Load daily aggregation
    daily = pd.read_csv('bluesky_all_daily_sentiment.csv')
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')

    print(f"\nDate range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"Total days: {len(daily)}")

    # Summary statistics
    print("\nNRC Net Sentiment Statistics:")
    print(f"  Mean: {daily['nrc_net_sentiment_mean'].mean():.2f}")
    print(f"  Std Dev: {daily['nrc_net_sentiment_mean'].std():.2f}")
    print(f"  Min: {daily['nrc_net_sentiment_mean'].min():.2f}")
    print(f"  Max: {daily['nrc_net_sentiment_mean'].max():.2f}")

    print("\nHuggingFace Positive Score Statistics:")
    print(f"  Mean: {daily['hf_positive_score_mean'].mean():.2f}")
    print(f"  Std Dev: {daily['hf_positive_score_mean'].std():.2f}")
    print(f"  Min: {daily['hf_positive_score_mean'].min():.2f}")
    print(f"  Max: {daily['hf_positive_score_mean'].max():.2f}")

    # Best/worst days
    print("\nMost Negative Days (NRC):")
    worst = daily.nsmallest(3, 'nrc_net_sentiment_mean')[['date', 'nrc_net_sentiment_mean']]
    for _, row in worst.iterrows():
        print(f"  {row['date'].date()}: {row['nrc_net_sentiment_mean']:.2f}")

    print("\nMost Positive Days (NRC):")
    best = daily.nlargest(3, 'nrc_net_sentiment_mean')[['date', 'nrc_net_sentiment_mean']]
    for _, row in best.iterrows():
        print(f"  {row['date'].date()}: {row['nrc_net_sentiment_mean']:.2f}")


def example_3_emotion_distribution():
    """Analyze distribution of emotions."""
    print("\n" + "=" * 70)
    print("Example 3: Emotion Distribution Analysis")
    print("=" * 70)

    df = pd.read_csv('bluesky_all_with_sentiment.csv')

    emotions = ['nrc_anger', 'nrc_anticipation', 'nrc_disgust', 'nrc_fear',
                'nrc_joy', 'nrc_sadness', 'nrc_surprise', 'nrc_trust']

    print("\nEmotion Prevalence (% of texts with non-zero score):")
    for emotion in emotions:
        prevalence = (df[emotion] > 0).mean()
        mean_count = df[emotion].mean()
        print(f"  {emotion.replace('nrc_', '').title():15s}: {prevalence:.1%} " +
              f"(avg count: {mean_count:.2f})")

    # Dominant emotions
    print("\nMost Common Dominant Emotions:")
    dominant_counts = df['nrc_dominant_emotion'].value_counts()
    for emotion, count in dominant_counts.head(8).items():
        print(f"  {emotion.title():15s}: {count:5d} texts ({count/len(df):.1%})")


def example_4_keyword_analysis():
    """Analyze sentiment by keyword (if available)."""
    print("\n" + "=" * 70)
    print("Example 4: Sentiment by Keyword")
    print("=" * 70)

    df = pd.read_csv('bluesky_all_with_sentiment.csv')

    if 'keyword' in df.columns:
        print("\nSentiment Statistics by Keyword:")
        keyword_sentiment = df.groupby('keyword').agg({
            'nrc_net_sentiment': ['mean', 'count'],
            'hf_positive_score': 'mean'
        }).round(2)
        keyword_sentiment.columns = ['NRC_Net_Sentiment', 'Count', 'HF_Positive_Score']
        keyword_sentiment = keyword_sentiment.sort_values('NRC_Net_Sentiment', ascending=False)

        print(keyword_sentiment.to_string())
    else:
        print("Keyword column not found in data.")


def example_5_text_filtering():
    """Find texts with specific sentiment profiles."""
    print("\n" + "=" * 70)
    print("Example 5: Text Filtering by Sentiment")
    print("=" * 70)

    df = pd.read_csv('bluesky_all_with_sentiment.csv')

    # Highly emotional texts (high absolute net sentiment)
    high_emotion = df[df['nrc_net_sentiment'].abs() > 2].head(5)
    print(f"\nHighly Emotional Texts ({len(high_emotion)} total):")
    for idx, row in high_emotion.iterrows():
        text = row['text_clean'][:60] + '...' if len(row['text_clean']) > 60 else row['text_clean']
        print(f"  [{row['nrc_net_sentiment']:+.0f}] {text}")

    # Mixed sentiment (disagreement between methods)
    df['nrc_pos'] = df['nrc_net_sentiment'] > 0
    df['hf_pos'] = df['hf_dominant_sentiment'].isin(['positive', 'very_positive'])
    mixed = df[df['nrc_pos'] != df['hf_pos']].head(5)
    print(f"\nMixed Sentiment Texts (Methods Disagree, {len(mixed)} total):")
    for idx, row in mixed.iterrows():
        text = row['text_clean'][:60] + '...' if len(row['text_clean']) > 60 else row['text_clean']
        print(f"  [NRC: {row['nrc_net_sentiment']:+.0f}, HF: {row['hf_dominant_sentiment']}] {text}")


if __name__ == '__main__':
    # Run all examples
    try:
        example_1_compare_methods()
    except FileNotFoundError:
        print("Could not find sentiment scored files. Run sentiment_scoring.py first.")

    try:
        example_2_temporal_trends()
    except Exception as e:
        print(f"Temporal trends error: {e}")

    try:
        example_3_emotion_distribution()
    except Exception as e:
        print(f"Emotion distribution error: {e}")

    try:
        example_4_keyword_analysis()
    except Exception as e:
        print(f"Keyword analysis error: {e}")

    try:
        example_5_text_filtering()
    except Exception as e:
        print(f"Text filtering error: {e}")

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
