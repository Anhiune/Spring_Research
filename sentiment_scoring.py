#!/usr/bin/env python3
"""
sentiment_scoring.py – Dual-backend sentiment analysis for cleaned social media data.

Architecture
============
1. Input Layer       – read cleaned CSV from clean_sentiment_data.py
2. Dual Scorers      – NRC Lexicon + HuggingFace transformer
3. Processing        – batch-based scoring with language awareness
4. Aggregation       – daily time series generation
5. Output Layer      – per-row + daily CSV + JSON reports

Usage Examples
==============
  # Basic: score cleaned Bluesky data with both backends
  python sentiment_scoring.py bluesky_all_cleaned.csv

  # Specify output directory
  python sentiment_scoring.py reddit_cleaned.csv --output-dir ./results

  # Only NRC scoring (skip HF)
  python sentiment_scoring.py data.csv --skip-hf

  # Only HF scoring (skip NRC)
  python sentiment_scoring.py data.csv --skip-nrc

  # Advanced: custom text column, batch size
  python sentiment_scoring.py data.csv --text-col text_clean --batch-size 64

Author : UROP Research Team
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import sys
import time
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Enable tqdm with pandas
tqdm.pandas(desc="Progress")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORING_VERSION = "1.0.0"
SEED = 42
NRC_DOWNLOAD_URL = "https://saifmohammad.com/WebDocs/NRC-Emotion-Lexicon.zip"
NRC_LEXICON_FILENAME = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# Emotion dimensions (NRC & Bollen)
NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
    "positive", "negative"
]

# HF sentiment classes
HF_CLASSES = ["very_negative", "negative", "neutral", "positive", "very_positive"]

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sentiment_scoring")

# Regex patterns
RE_WHITESPACE = re.compile(r"\s+")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy int/float types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# NRC Lexicon Management
# ---------------------------------------------------------------------------

class NRCLexicon:
    """Manage NRC emotion lexicon with caching and auto-download."""

    def __init__(self, lexicon_path: str = "nrc_lexicon.txt"):
        self.lexicon_path = pathlib.Path(lexicon_path)
        self.emotions: Dict[str, List[str]] = defaultdict(list)
        self._load_or_download()

    def _download_lexicon(self) -> None:
        """Download and extract NRC lexicon from Harvard source."""
        log.info("Downloading NRC Emotion Lexicon from %s…", NRC_DOWNLOAD_URL)
        try:
            zip_path = pathlib.Path("nrc_lexicon.zip")
            urllib.request.urlretrieve(NRC_DOWNLOAD_URL, str(zip_path))

            # Extract the lexicon file
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find and extract the wordlevel lexicon file
                for name in zf.namelist():
                    if "Wordlevel" in name and ".txt" in name:
                        with zf.open(name) as source:
                            with open(self.lexicon_path, 'wb') as target:
                                target.write(source.read())
                        log.info("Extracted %s to %s", name, self.lexicon_path)
                        break

            zip_path.unlink()  # Clean up zip
            log.info("NRC lexicon download complete.")
        except Exception as exc:
            log.error("Failed to download NRC lexicon: %s", exc)
            raise

    def _load_or_download(self) -> None:
        """Load lexicon from disk or download if missing."""
        if not self.lexicon_path.exists():
            # Try default location first
            default_paths = [
                pathlib.Path("c:/Users/hoang/Downloads/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
                pathlib.Path("~/Downloads/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt").expanduser(),
            ]
            for default_path in default_paths:
                if default_path.exists():
                    self.lexicon_path = default_path
                    log.info("Found NRC lexicon at %s", default_path)
                    break
            else:
                log.info("NRC lexicon not found at %s. Downloading…", self.lexicon_path)
                try:
                    self._download_lexicon()
                except Exception as exc:
                    log.error("Download failed: %s", exc)
                    raise

        log.info("Loading NRC lexicon from %s…", self.lexicon_path)
        with open(self.lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    word, emotion = parts[0], parts[1]
                    association = int(parts[2])
                    if association == 1 and emotion in NRC_EMOTIONS:
                        self.emotions[word].append(emotion)

        log.info("Loaded %d words from NRC lexicon.", len(self.emotions))

    def get_emotions(self, word: str) -> List[str]:
        """Get emotion associations for a word."""
        return self.emotions.get(word.lower(), [])


# ---------------------------------------------------------------------------
# Abstract Scorer Base Class
# ---------------------------------------------------------------------------

class SentimentScorer(ABC):
    """Abstract base class for sentiment scorers."""

    @abstractmethod
    def score_batch(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Score a batch of texts. Return list of score dicts."""
        pass

    @abstractmethod
    def get_output_columns(self) -> List[str]:
        """Return column names for output."""
        pass


# ---------------------------------------------------------------------------
# NRC Lexicon Scorer
# ---------------------------------------------------------------------------

class NRCLexiconScorer(SentimentScorer):
    """Score sentiment using NRC emotion lexicon (Bollen method)."""

    def __init__(self, lexicon_path: str = "nrc_lexicon.txt"):
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            self.word_tokenize = word_tokenize

            # Ensure tokenizer data is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        except ImportError:
            log.error("NLTK not installed. Install via: pip install nltk")
            raise

        self.lexicon = NRCLexicon(lexicon_path)

    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize text and filter to alpha characters."""
        if not text or not isinstance(text, str):
            return []
        try:
            tokens = self.word_tokenize(text.lower())
            return [t for t in tokens if t.isalpha()]
        except Exception:
            return []

    def score_batch(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Score batch of texts using NRC lexicon."""
        results = []
        for text in texts:
            tokens = self._tokenize_and_clean(text)
            emotion_counts = {emotion: 0 for emotion in NRC_EMOTIONS}

            for token in tokens:
                emotions = self.lexicon.get_emotions(token)
                for emotion in emotions:
                    emotion_counts[emotion] += 1

            # Calculate derived metrics
            net_sentiment = emotion_counts.get("positive", 0) - emotion_counts.get("negative", 0)
            dominant_emotion = max(NRC_EMOTIONS,
                                 key=lambda e: emotion_counts.get(e, 0))

            score_dict = {f"nrc_{emotion}": emotion_counts[emotion]
                         for emotion in NRC_EMOTIONS}
            score_dict["nrc_net_sentiment"] = net_sentiment
            score_dict["nrc_dominant_emotion"] = dominant_emotion

            results.append(score_dict)

        return results

    def get_output_columns(self) -> List[str]:
        """Return NRC output column names."""
        cols = [f"nrc_{emotion}" for emotion in NRC_EMOTIONS]
        cols.extend(["nrc_net_sentiment", "nrc_dominant_emotion"])
        return cols


# ---------------------------------------------------------------------------
# HuggingFace Transformer Scorer
# ---------------------------------------------------------------------------

class HuggingFaceScorer(SentimentScorer):
    """Score sentiment using HuggingFace multilingual transformer."""

    def __init__(self, model_name: str = "tabularisai/multilingual-sentiment-analysis",
                 batch_size: int = 32):
        try:
            from transformers import pipeline
            self.pipeline = pipeline
        except ImportError:
            log.error("transformers not installed. Install via: pip install transformers torch")
            raise

        self.model_name = model_name
        self.batch_size = batch_size
        self.pipe = None
        self._load_model()

    def _load_model(self) -> None:
        """Load HuggingFace pipeline."""
        log.info("Loading HuggingFace model: %s", self.model_name)
        try:
            # Auto-detect device (GPU if available)
            device = 0 if self._has_gpu() else -1
            self.pipe = self.pipeline(
                "text-classification",
                model=self.model_name,
                device=device,
                top_k=None
            )
            log.info("Model loaded successfully.")
        except Exception as exc:
            log.error("Failed to load model: %s", exc)
            raise

    @staticmethod
    def _has_gpu() -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def score_batch(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Score batch of texts using HuggingFace transformer."""
        # Filter out empty texts
        valid_texts = [t if t and isinstance(t, str) else "" for t in texts]

        try:
            predictions = self.pipe(valid_texts, batch_size=self.batch_size)
        except Exception as exc:
            log.warning("Batch prediction failed: %s. Processing individually.", exc)
            predictions = []
            for text in valid_texts:
                try:
                    result = self.pipe(text)
                    predictions.append(result)
                except Exception:
                    predictions.append([{"label": "neutral", "score": 1.0}])

        results = []
        for pred in predictions:
            # pred is a list of dicts: [{"label": "positive", "score": 0.95}, ...]
            score_dict = {}

            # Extract scores for each class
            for item in pred:
                label = item.get("label", "").lower()
                score = float(item.get("score", 0.0))

                # Normalize label to match our class names
                normalized_label = label.replace(" ", "_")
                if normalized_label in HF_CLASSES:
                    score_dict[f"hf_{normalized_label}_score"] = round(score, 4)

            # Ensure all classes have scores (fill missing with 0)
            for class_name in HF_CLASSES:
                if f"hf_{class_name}_score" not in score_dict:
                    score_dict[f"hf_{class_name}_score"] = 0.0

            # Calculate dominant sentiment
            scores = {k: v for k, v in score_dict.items() if k.endswith("_score")}
            if scores:
                dominant = max(scores, key=scores.get).replace("hf_", "").replace("_score", "")
                score_dict["hf_dominant_sentiment"] = dominant
            else:
                score_dict["hf_dominant_sentiment"] = "neutral"

            results.append(score_dict)

        return results

    def get_output_columns(self) -> List[str]:
        """Return HuggingFace output column names."""
        cols = [f"hf_{class_name}_score" for class_name in HF_CLASSES]
        cols.append("hf_dominant_sentiment")
        return cols


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class SentimentPipeline:
    """Orchestrate sentiment scoring pipeline."""

    def __init__(self, input_path: str, text_col: str = "text_clean",
                 date_col: str = None, lang_col: str = "lang",
                 output_dir: str = "./", skip_nrc: bool = False,
                 skip_hf: bool = False, batch_size: int = 32,
                 nrc_lexicon_path: str = "nrc_lexicon.txt"):
        self.input_path = pathlib.Path(input_path)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.text_col = text_col
        self.lang_col = lang_col
        self.batch_size = batch_size
        self.skip_nrc = skip_nrc
        self.skip_hf = skip_hf

        # Auto-detect date column
        self.date_col = date_col or self._detect_date_column()

        # Initialize scorers
        self.nrc_scorer = None if skip_nrc else NRCLexiconScorer(nrc_lexicon_path)
        self.hf_scorer = None if skip_hf else HuggingFaceScorer(batch_size=batch_size)

        log.info("Pipeline initialized: input=%s, skip_nrc=%s, skip_hf=%s",
                self.input_path.name, skip_nrc, skip_hf)

    def _detect_date_column(self) -> str:
        """Auto-detect date/timestamp column."""
        df_sample = pd.read_csv(self.input_path, nrows=1)
        for col in df_sample.columns:
            if col.lower() in ["created_at", "timestamp", "date", "datetime"]:
                return col
        return None

    def run(self) -> None:
        """Execute full pipeline."""
        t0 = time.time()

        # Read input
        log.info("Reading input CSV…")
        df = pd.read_csv(self.input_path)
        df_in = df.copy()

        # Validate required columns
        if self.text_col not in df.columns:
            log.error("Column '%s' not found. Available: %s",
                     self.text_col, list(df.columns))
            sys.exit(1)

        # Add sentiment scores
        log.info("Applying sentiment scoring to %d rows…", len(df))
        df = self._apply_scoring(df)

        # Save per-row results
        output_stem = self.input_path.stem
        row_output = self.output_dir / f"{output_stem}_with_sentiment.csv"
        df.to_csv(row_output, index=False)
        log.info("Per-row scores → %s", row_output)

        # Generate daily aggregation
        if self.date_col:
            log.info("Generating daily aggregation…")
            daily_df = self._aggregate_daily(df)
            daily_output = self.output_dir / f"{output_stem}_daily_sentiment.csv"
            daily_df.to_csv(daily_output, index=False)
            log.info("Daily aggregation → %s", daily_output)
        else:
            log.warning("No date column found. Skipping daily aggregation.")
            daily_df = None

        # Generate report
        elapsed = time.time() - t0
        report = self._build_report(df_in, df, daily_df, elapsed)
        report_path = self.output_dir / f"{output_stem}_sentiment_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        log.info("Report → %s", report_path)

        log.info("Done in %.1fs.", elapsed)

    def _apply_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply NRC and/or HF scoring to dataframe."""
        texts = df[self.text_col].fillna("").astype(str).tolist()

        # Process in batches
        all_scores = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Scoring"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            # Get scores from each backend
            nrc_scores = []
            hf_scores = []

            if self.nrc_scorer:
                nrc_scores = self.nrc_scorer.score_batch(batch_texts)

            if self.hf_scorer:
                hf_scores = self.hf_scorer.score_batch(batch_texts)

            # Merge scores
            for i in range(len(batch_texts)):
                merged = {}
                if nrc_scores:
                    merged.update(nrc_scores[i])
                if hf_scores:
                    merged.update(hf_scores[i])
                all_scores.append(merged)

        # Add scores to dataframe
        score_df = pd.DataFrame(all_scores)
        df = pd.concat([df, score_df], axis=1)
        return df

    def _aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment scores by day."""
        if self.date_col not in df.columns:
            return None

        # Parse date with flexible format handling
        df["date"] = pd.to_datetime(df[self.date_col], format="mixed", utc=True).dt.date

        # Get sentiment columns
        sentiment_cols = [col for col in df.columns
                         if col.startswith("nrc_") or col.startswith("hf_")]
        # Filter to numeric columns (exclude dominant_emotion/sentiment which are strings)
        numeric_cols = [col for col in sentiment_cols
                       if col not in ["nrc_dominant_emotion", "hf_dominant_sentiment"]]

        # Group by date and calculate mean for numeric columns
        daily = df.groupby("date").agg({
            col: "mean" for col in numeric_cols
        }).reset_index()
        daily.columns = ["date"] + [f"{col}_mean" for col in numeric_cols]

        return daily

    def _build_report(self, df_in: pd.DataFrame, df_out: pd.DataFrame,
                      daily_df: Optional[pd.DataFrame], elapsed: float) -> Dict[str, Any]:
        """Generate JSON report."""
        sentiment_cols = [col for col in df_out.columns
                         if col.startswith("nrc_") or col.startswith("hf_")]

        return {
            "scoring_version": SCORING_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_file": str(self.input_path),
            "text_column": self.text_col,
            "date_column": self.date_col,
            "rows_input": len(df_in),
            "rows_output": len(df_out),
            "sentiment_columns": sentiment_cols,
            "settings": {
                "skip_nrc": self.skip_nrc,
                "skip_hf": self.skip_hf,
                "batch_size": self.batch_size,
            },
            "elapsed_seconds": round(elapsed, 2),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Apply dual sentiment analysis (NRC + HuggingFace) to cleaned social media data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("input", help="Path to cleaned CSV file (from clean_sentiment_data.py)")
    p.add_argument("--text-col", default="text_clean",
                   help="Column name containing text (default: text_clean)")
    p.add_argument("--date-col", default=None,
                   help="Column name for date/timestamp (auto-detected if not specified)")
    p.add_argument("--lang-col", default="lang",
                   help="Column name for language code (default: lang)")
    p.add_argument("--output-dir", default="./",
                   help="Output directory (default: current directory)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for HF model (default: 32)")
    p.add_argument("--nrc-path", default="nrc_lexicon.txt",
                   help="Path to NRC lexicon file (auto-download if missing)")
    p.add_argument("--skip-nrc", action="store_true",
                   help="Skip NRC lexicon scoring")
    p.add_argument("--skip-hf", action="store_true",
                   help="Skip HuggingFace transformer scoring")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    try:
        pipeline = SentimentPipeline(
            input_path=args.input,
            text_col=args.text_col,
            date_col=args.date_col,
            lang_col=args.lang_col,
            output_dir=args.output_dir,
            skip_nrc=args.skip_nrc,
            skip_hf=args.skip_hf,
            batch_size=args.batch_size,
            nrc_lexicon_path=args.nrc_path,
        )
        pipeline.run()
    except Exception as exc:
        log.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
