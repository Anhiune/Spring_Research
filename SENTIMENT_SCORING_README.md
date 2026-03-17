#!/usr/bin/env python3
"""
README for sentiment_scoring.py - Dual-Backend Sentiment Analysis Pipeline

Overview
========
sentiment_scoring.py provides a complete sentiment analysis pipeline for cleaned social media data
(from clean_sentiment_data.py). It simultaneously scores text using two complementary backends:

1. NRC Lexicon (Bollen Method) - Traditional emotion lexicon providing 10 emotion dimensions
2. HuggingFace Multilingual Transformer - Modern neural approach supporting 23+ languages

Quick Start
===========
# Score Bluesky data with both backends
python sentiment_scoring.py bluesky_all_cleaned.csv --text-col text_clean --date-col created_at

# Score Reddit data with both backends
python sentiment_scoring.py reddit_pullpush_2023-01_cleaned.csv --text-col text_clean --date-col timestamp

# Only NRC scoring (faster, no GPU needed)
python sentiment_scoring.py data.csv --skip-hf

# Only HuggingFace scoring (modern transformer only)
python sentiment_scoring.py data.csv --skip-nrc

Installation
============
pip install pandas nltk transformers torch

# For GPU acceleration (optional, requires CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Download NLTK data (automatic on first run):
python -c "import nltk; nltk.download('punkt')"

# NRC Lexicon Setup
The script automatically finds the NRC lexicon at:
  - c:/Users/hoang/Downloads/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

Input Requirements
==================
Your cleaned CSV (from clean_sentiment_data.py) must contain:

Required columns:
  - text_clean        : Cleaned text to score
  - lang              : Language code (ISO 639-1, e.g. 'en', 'de', 'fr')
  - text_en           : English translation (needed for NRC on non-English text)
  - created_at OR timestamp : For daily aggregation

Optional columns:
  - any metadata (will be preserved in output)

Output Files
============
For input "mydata_cleaned.csv", produces:

1. mydata_cleaned_with_sentiment.csv
   - All original columns + new sentiment columns
   - NRC columns: nrc_anger, nrc_joy, ..., nrc_net_sentiment, nrc_dominant_emotion (12 cols)
   - HF columns: hf_very_negative_score, hf_negative_score, ..., hf_dominant_sentiment (6 cols)
   - Rows: Same as input

2. mydata_cleaned_daily_sentiment.csv
   - Date-aggregated sentiment metrics
   - Columns: date + mean of all numeric sentiment scores
   - Rows: One per unique date

3. mydata_cleaned_sentiment_report.json
   - Processing metadata (rows processed, time elapsed, backends used)
   - Column names and counts

Sentiment Backends Explained
============================

NRC Lexicon (Emotion Dimensions):
---------------------------------
Produces 10 emotion dimensions based on word-level emotion associations:
  - Primary emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust
  - Valence: positive, negative
  - Derived: net_sentiment (positive_count - negative_count)
  - Dominant emotion: which emotion has highest count

Score interpretation: Raw word counts (higher = more of that emotion detected)
Pros: Fast, interpretable, works in all languages (via translation)
Cons: Lexicon-based, misses context

HuggingFace Model (5-Class Sentiment):
--------------------------------------
Modern transformer-based classification supporting 23 languages natively:
  - Classes: very_negative, negative, neutral, positive, very_positive
  - Scores: Confidence scores for each class (0-1)
  - Dominant: Which class has highest confidence

Score interpretation: Confidence probabilities (sum to ~1.0)
Pros: Context-aware, multilingual, state-of-the-art
Cons: Slower, requires more compute, less interpretable

Command Line Options
====================
python sentiment_scoring.py INPUT_CSV [options]

INPUT_CSV                 Path to cleaned CSV file

Options:
  --text-col TEXT_COL     Column with text to score (default: text_clean)
  --date-col DATE_COL     Column with date/timestamp (auto-detected if not specified)
  --lang-col LANG_COL     Column with language code (default: lang)
  --output-dir DIR        Output directory (default: current dir)
  --batch-size N          HF batch size, higher=faster but more RAM (default: 32)
  --nrc-path PATH         NRC lexicon file path
  --skip-nrc              Disable NRC scoring
  --skip-hf               Disable HuggingFace scoring

Performance Notes
=================
Dataset Size | NRC Only | Both Backends | Notes
-------------|----------|---------------|------
100 rows     | <1s      | 2-3s          | Fast, good for testing
1,000 rows   | 1-2s     | 10-15s        |
10,000 rows  | 2-5s     | 30-60s        | HF model inference dominates
100K rows    | 20-30s   | 5-10 min      | CPU-bound; GPU ~100x faster

Tips for Large Datasets:
- Use --skip-hf if only NRC needed (much faster)
- NRC scoring is parallelizable; HF benefits from GPU
- Batch size 32-64 is optimal for most systems
- 135K rows takes ~2-3 hours on CPU, ~30 min on GPU

Example: Using Both Scores
===========================
After scoring, compare sentiment methods:

import pandas as pd

df = pd.read_csv('data_cleaned_with_sentiment.csv')

# Compare NRC vs HF on same text
comparison = df[['text_clean', 'nrc_net_sentiment', 'hf_dominant_sentiment']].head(10)
print(comparison)

# Check disagreement between methods
nrc_positive = df['nrc_net_sentiment'] > 0
hf_positive = df['hf_dominant_sentiment'].isin(['positive', 'very_positive'])
disagreement = nrc_positive != hf_positive
print(f"Methods disagree on {disagreement.sum()} texts ({100*disagreement.mean():.1f}%)")

# Calibrate HF probabilities
neutral_mask = df['hf_neutral_score'] > 0.4
df.loc[neutral_mask, 'hf_dominant_sentiment'] = 'neutral'

Troubleshooting
===============

Issue: "NRC lexicon not found"
Solution: Ensure the file exists at c:/Users/hoang/Downloads/NRC-Emotion-Lexicon/...
          Or manually download from: https://saifmohammad.com/WebDocs/NRC-Emotion-Lexicon.zip

Issue: Date parsing errors
Solution: Script auto-detects date format. If errors occur, use --date-col explicitly.
          Supported formats: ISO8601, Unix timestamps, various datetime strings.

Issue: Out of memory (HF model)
Solution: Reduce batch size: --batch-size 8
          Or: Use --skip-hf to use NRC only

Issue: Very slow on CPU
Solution: This is normal for 100K+ rows with HF. Either:
          - Wait (2-3 hours typical)
          - Use --skip-hf (10x faster)
          - Use GPU if available (100x faster)

Testing
=======
Create a small test sample:
python -c "import pandas as pd; df = pd.read_csv('bluesky_all_cleaned.csv'); \
           df.iloc[:100].to_csv('test_sample.csv', index=False)"

python sentiment_scoring.py test_sample.csv --text-col text_clean --date-col created_at

Verify outputs:
python -c "import pandas as pd; df = pd.read_csv('test_sample_with_sentiment.csv'); \
           print(df[['text_clean', 'nrc_anger', 'hf_neutral_score']].head())"

Version & Attribution
=====================
sentiment_scoring.py v1.0.0
Created for UROP Spring 2026 Research Project
Based on: Bollen et al. (2011) NRC Emotion Lexicon, HuggingFace tabularisai/multilingual-sentiment-analysis

References
==========
- NRC Lexicon: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
- HuggingFace Model: https://huggingface.co/tabularisai/multilingual-sentiment-analysis
- NLTK: https://www.nltk.org/
- TransformersLib: https://huggingface.co/docs/transformers/
