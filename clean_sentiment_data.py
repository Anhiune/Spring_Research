#!/usr/bin/env python3
"""
clean_sentiment_data.py – Production-quality text-cleaning pipeline for
sentiment analysis.

Architecture
------------
1. IO layer          – read / write CSV, XLSX, JSONL, Parquet
2. Normalisation     – ftfy + NFKC + whitespace
3. Boilerplate       – regex-based junk removal (configurable JSON list)
4. Entity masking    – URLs, emails, phones, @mentions, hashtags, numbers
5. Punctuation       – keep sentiment-bearing !? ; compress extreme repeats
6. Emoji / emoticon  – demojize + emoticon tokens
7. Negation scope    – optional bigram joining for classic models
8. Stopword / lemma  – optional, for classic models only
9. Deduplication     – exact + optional MinHash/LSH near-dupe
10. Quality filters  – min-tokens, max-chars, bot detection
11. Language detect  – fasttext or langdetect
12. Translation      – openai / gcp / hf / none  (cached in SQLite)
13. ASCII folding    – optional unidecode transliteration
14. Reporting        – JSON report + console log

Usage examples
--------------
  # Minimal – clean a Reddit CSV
  python clean_sentiment_data.py reddit_pullpush_2023-01.csv

  # Specify text column, mask numbers, drop dupes
  python clean_sentiment_data.py data.csv --text-col body --num-mode mask --drop-dupes

  # Classic-model pipeline (stopword removal + negation scope)
  python clean_sentiment_data.py data.csv --model-type classic --negation-scope

  # Translate non-English rows via OpenAI
  python clean_sentiment_data.py data.csv --translate-backend openai

  # Output as parquet
  python clean_sentiment_data.py data.csv --out-format parquet

Author : UROP Research Team
Version: 1.0.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ftfy
import emoji
import pandas as pd


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
# Constants
# ---------------------------------------------------------------------------
CLEANING_VERSION = "1.0.0"
SEED = 42

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("clean_sentiment")

# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------
RE_URL = re.compile(
    r"https?://\S+|www\.\S+", re.IGNORECASE
)
RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
)
RE_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}"
)
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
RE_WHITESPACE = re.compile(r"\s+")
RE_REPEAT_PUNCT = re.compile(r"([!?.])\1{3,}")   # 4+ repeats → max 3
RE_CAMELCASE = re.compile(r"(?<=[a-z])(?=[A-Z])")

# Negation cues
NEGATION_CUES = {"not", "no", "never", "hardly", "barely", "scarcely",
                 "n't", "nt", "neither", "nor", "nobody", "nothing",
                 "nowhere", "cannot"}

# Common emoticons → tokens
EMOTICON_MAP = {
    ":)":  ":smile:",   ":-)": ":smile:",   "=)":  ":smile:",
    ":(":  ":sad:",     ":-(": ":sad:",      "=(":  ":sad:",
    ";)":  ":wink:",    ";-)": ":wink:",
    ":D":  ":grin:",    ":-D": ":grin:",
    ":'(": ":cry:",     ":'-(": ":cry:",
    ":P":  ":tongue:",  ":-P": ":tongue:",
    "<3":  ":heart:",
    ":/":  ":unsure:",  ":-/": ":unsure:",
    ":O":  ":surprise:", ":-O": ":surprise:",
    "XD":  ":laughing:", "xD": ":laughing:",
}
# Sort by longest first so multi-char emoticons match before subsets
EMOTICON_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(EMOTICON_MAP, key=len, reverse=True))
)

# Bot patterns for Reddit / Bluesky
BOT_PATTERNS = [
    re.compile(r"I am a bot", re.IGNORECASE),
    re.compile(r"I'm a bot", re.IGNORECASE),
    re.compile(r"this action was performed automatically", re.IGNORECASE),
    re.compile(r"beep boop", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_input(path: str, **kwargs) -> pd.DataFrame:
    """Auto-detect file type and read into DataFrame."""
    p = pathlib.Path(path)
    ext = p.suffix.lower()
    readers = {
        ".csv":     pd.read_csv,
        ".xlsx":    pd.read_excel,
        ".jsonl":   lambda f, **kw: pd.read_json(f, lines=True, **kw),
        ".parquet": pd.read_parquet,
    }
    reader = readers.get(ext)
    if reader is None:
        raise ValueError(f"Unsupported file type: {ext}. "
                         f"Supported: {', '.join(readers)}")
    log.info("Reading %s (%s)", p.name, ext)
    return reader(str(p), **kwargs)


def write_output(df: pd.DataFrame, stem: str, out_format: str) -> str:
    """Write DataFrame to disk; return output path."""
    out_path = f"{stem}_cleaned.{out_format}"
    if out_format == "csv":
        df.to_csv(out_path, index=False)
    elif out_format == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {out_format}")
    log.info("Saved cleaned data → %s  (%d rows)", out_path, len(df))
    return out_path


# ---------------------------------------------------------------------------
# Core transforms  (each is a pure function: str → str)
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Fix mojibake via ftfy, apply NFKC, collapse whitespace."""
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text


def remove_boilerplate(text: str,
                       patterns: Optional[List[re.Pattern]] = None) -> str:
    """Remove boilerplate lines matched by compiled regex patterns."""
    if not patterns:
        return text
    for pat in patterns:
        text = pat.sub("", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def split_camelcase(word: str) -> str:
    """Split CamelCase into separate words: 'LoveThis' → 'Love This'."""
    return RE_CAMELCASE.sub(" ", word)


def mask_entities(text: str, num_mode: str = "keep") -> str:
    """Replace URLs, emails, phones, @mentions with placeholders.

    Hashtags: remove # and split CamelCase.
    Numbers: keep or mask depending on *num_mode*.
    """
    text = RE_URL.sub("<URL>", text)
    text = RE_EMAIL.sub("<EMAIL>", text)
    text = RE_PHONE.sub("<PHONE>", text)
    text = RE_MENTION.sub("<USER>", text)
    # Hashtags — keep word, split CamelCase
    text = RE_HASHTAG.sub(lambda m: split_camelcase(m.group(1)), text)
    if num_mode == "mask":
        text = RE_NUMBER.sub("<NUM>", text)
    return text


def compress_punctuation(text: str) -> str:
    """Compress 4+ repeated punctuation to 3, keeping intensity."""
    return RE_REPEAT_PUNCT.sub(r"\1\1\1", text)


def handle_emojis(text: str) -> str:
    """Convert Unicode emojis to text tokens; convert emoticons."""
    # Emoticons first (before we alter text further)
    text = EMOTICON_PATTERN.sub(lambda m: EMOTICON_MAP[m.group(0)], text)
    # Unicode emojis
    text = emoji.demojize(text, delimiters=(":", ":"))
    return text


def negation_scope(text: str, window: int = 3) -> str:
    """Join negation cue with the following *window* tokens via underscores.

    Example: "not very good at all" → "not_very_good_at all"
    """
    tokens = text.split()
    result: list[str] = []
    i = 0
    while i < len(tokens):
        tok_lower = tokens[i].lower().rstrip(".,;:!?")
        if tok_lower in NEGATION_CUES or tok_lower.endswith("n't"):
            # Collect negation cue + next `window` tokens
            scope = [tokens[i]]
            for j in range(1, window + 1):
                if i + j < len(tokens):
                    scope.append(tokens[i + j])
            result.append("_".join(scope))
            i += len(scope)
        else:
            result.append(tokens[i])
            i += 1
    return " ".join(result)


def apply_stopwords_lemma(text: str) -> str:
    """Remove stopwords (except negators) and lemmatize for classic models."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
    except ImportError:
        log.warning("nltk not installed — skipping stopword/lemma step.")
        return text

    # Ensure resources are available
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    stop = set(stopwords.words("english")) - NEGATION_CUES - {"not", "no",
                                                                "nor", "never"}
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.lower() not in stop]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Bot detection
# ---------------------------------------------------------------------------

def is_bot_post(text: str) -> bool:
    """Return True if text matches any known bot signature."""
    for pat in BOT_PATTERNS:
        if pat.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def drop_exact_dupes(df: pd.DataFrame, col: str = "text_raw") -> pd.DataFrame:
    """Drop rows with exact duplicate text_raw values."""
    before = len(df)
    df = df.drop_duplicates(subset=[col], keep="first").reset_index(drop=True)
    log.info("Exact dedup: %d → %d rows (removed %d)",
             before, len(df), before - len(df))
    return df


def drop_near_dupes_minhash(df: pd.DataFrame, col: str = "text_clean",
                            threshold: float = 0.85) -> pd.DataFrame:
    """Remove near-duplicate rows using MinHash / LSH.

    Requires the `datasketch` package.  Off by default.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log.warning("datasketch not installed – skipping near-dupe removal.  "
                     "Install via: pip install datasketch")
        return df

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes: dict[int, MinHash] = {}
    for idx, text in df[col].items():
        m = MinHash(num_perm=128)
        for w in str(text).lower().split():
            m.update(w.encode("utf8"))
        minhashes[idx] = m
        try:
            lsh.insert(str(idx), m)
        except ValueError:
            pass  # duplicate hash bucket — already flagged

    keep: set[int] = set()
    for idx in df.index:
        if idx not in keep:
            result = lsh.query(minhashes[idx])
            group = sorted(int(r) for r in result)
            keep.add(group[0])  # keep first occurrence

    before = len(df)
    df = df.loc[df.index.isin(keep)].reset_index(drop=True)
    log.info("Near-dupe removal: %d → %d rows", before, len(df))
    return df


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _detect_lang_fasttext(texts: pd.Series) -> pd.Series:
    """Use fasttext's lid.176.bin for language detection."""
    import fasttext
    import warnings
    warnings.filterwarnings("ignore")
    model_path = os.environ.get(
        "FASTTEXT_LID_MODEL",
        str(pathlib.Path(__file__).parent / "lid.176.bin")
    )
    if not pathlib.Path(model_path).exists():
        raise FileNotFoundError(
            f"FastText model not found at {model_path}. "
            "Download from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin "
            "or set FASTTEXT_LID_MODEL env var."
        )
    model = fasttext.load_model(model_path)

    def _predict(text: str) -> str:
        text = text.replace("\n", " ").strip()
        if not text:
            return "und"
        labels, _ = model.predict(text, k=1)
        return labels[0].replace("__label__", "")

    return texts.apply(_predict)


def _detect_lang_langdetect(texts: pd.Series) -> pd.Series:
    """Fallback language detection via langdetect with progress logging."""
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = SEED

    total = len(texts)
    results = []
    batch_report = max(1, total // 10)  # report every ~10%

    def _safe_detect(text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "und"

    for i, (idx, text) in enumerate(texts.items()):
        results.append(_safe_detect(str(text)))

        if (i + 1) % batch_report == 0:
            log.info("  Language detection: %d / %d (%.0f%%)",
                     i + 1, total, 100 * (i + 1) / total)

    return pd.Series(results, index=texts.index)


def detect_language(texts: pd.Series) -> pd.Series:
    """Detect language; prefer fasttext, fall back to langdetect."""
    try:
        return _detect_lang_fasttext(texts)
    except (ImportError, FileNotFoundError) as exc:
        log.info("FastText unavailable (%s), using langdetect.", exc)
        try:
            return _detect_lang_langdetect(texts)
        except ImportError:
            log.warning("Neither fasttext nor langdetect installed. "
                        "Setting lang='und'.")
            return pd.Series(["und"] * len(texts), index=texts.index)


# ---------------------------------------------------------------------------
# Translation cache (SQLite)
# ---------------------------------------------------------------------------

class TranslationCache:
    """Disk-backed cache keyed by (text, source_lang)."""

    def __init__(self, db_path: str = "translation_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, translation TEXT)"
        )
        self.conn.commit()

    @staticmethod
    def _key(text: str, lang: str) -> str:
        h = hashlib.sha256(f"{lang}||{text}".encode()).hexdigest()
        return h

    def get(self, text: str, lang: str) -> Optional[str]:
        key = self._key(text, lang)
        row = self.conn.execute(
            "SELECT translation FROM cache WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, text: str, lang: str, translation: str) -> None:
        key = self._key(text, lang)
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, translation) VALUES (?, ?)",
            (key, translation),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Translation backends
# ---------------------------------------------------------------------------

def _translate_openai(texts: List[str], langs: List[str],
                      cache: TranslationCache) -> List[str]:
    """Translate via OpenAI Chat API with batching and retries."""
    try:
        import openai
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    client = openai.OpenAI(api_key=api_key)
    results: list[str] = []

    # Process in batches of 20
    batch_size = 20
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        batch_langs = langs[start: start + batch_size]
        batch_results: list[str] = []

        for text, lang in zip(batch_texts, batch_langs):
            cached = cache.get(text, lang)
            if cached is not None:
                batch_results.append(cached)
                continue

            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system",
                             "content": "Translate the following text to English. "
                                        "Output ONLY the translation, nothing else."},
                            {"role": "user", "content": text},
                        ],
                        temperature=0.0,
                        max_tokens=2000,
                    )
                    translation = resp.choices[0].message.content.strip()
                    cache.put(text, lang, translation)
                    batch_results.append(translation)
                    break
                except Exception as exc:
                    wait = 2 ** attempt
                    log.warning("OpenAI attempt %d failed: %s. Retrying in %ds",
                                attempt + 1, exc, wait)
                    time.sleep(wait)
            else:
                log.error("OpenAI translation failed after 5 retries for text: %.60s", text)
                batch_results.append(text)  # fallback to original

        results.extend(batch_results)
    return results


def _translate_gcp(texts: List[str], langs: List[str],
                   cache: TranslationCache) -> List[str]:
    """Translate via Google Cloud Translation v3."""
    try:
        from google.cloud import translate_v3 as translate
    except ImportError:
        raise ImportError("Install google-cloud-translate: "
                          "pip install google-cloud-translate")

    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not creds or not project:
        raise EnvironmentError("Set GOOGLE_APPLICATION_CREDENTIALS and "
                               "GOOGLE_CLOUD_PROJECT env vars.")

    client = translate.TranslationServiceClient()
    parent = f"projects/{project}/locations/global"
    results: list[str] = []

    batch_size = 50
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        batch_langs = langs[start: start + batch_size]
        to_translate: list[tuple[int, str, str]] = []

        for i, (text, lang) in enumerate(zip(batch_texts, batch_langs)):
            cached = cache.get(text, lang)
            if cached is not None:
                results.append(cached)
            else:
                to_translate.append((len(results), text, lang))
                results.append("")  # placeholder

        if to_translate:
            contents = [t[1] for t in to_translate]
            for attempt in range(5):
                try:
                    response = client.translate_text(
                        request={
                            "parent": parent,
                            "contents": contents,
                            "target_language_code": "en",
                            "mime_type": "text/plain",
                        }
                    )
                    for j, trans in enumerate(response.translations):
                        idx, text, lang = to_translate[j]
                        results[idx] = trans.translated_text
                        cache.put(text, lang, trans.translated_text)
                    break
                except Exception as exc:
                    wait = 2 ** attempt
                    log.warning("GCP attempt %d failed: %s. Retrying in %ds",
                                attempt + 1, exc, wait)
                    time.sleep(wait)
            else:
                for idx, text, lang in to_translate:
                    results[idx] = text  # fallback

    return results


def _translate_hf(texts: List[str], langs: List[str],
                  cache: TranslationCache) -> List[str]:
    """Translate via HuggingFace MarianMT (offline)."""
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        raise ImportError(
            "Install transformers + sentencepiece:\n"
            "  pip install transformers sentencepiece torch"
        )

    # Group by source language for efficient model loading
    lang_groups: dict[str, list[tuple[int, str]]] = {}
    results = [""] * len(texts)
    for i, (text, lang) in enumerate(zip(texts, langs)):
        cached = cache.get(text, lang)
        if cached is not None:
            results[i] = cached
        else:
            lang_groups.setdefault(lang, []).append((i, text))

    for lang, items in lang_groups.items():
        model_name = f"Helsinki-NLP/opus-mt-{lang}-en"
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
        except Exception:
            log.warning("MarianMT model %s not found. "
                        "Falling back to original text for lang=%s.", model_name, lang)
            for idx, text in items:
                results[idx] = text
            continue

        batch_size = 16
        for start in range(0, len(items), batch_size):
            batch = items[start: start + batch_size]
            src_texts = [t for _, t in batch]
            encoded = tokenizer(src_texts, return_tensors="pt",
                                padding=True, truncation=True, max_length=512)
            translated = model.generate(**encoded)
            decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
            for (idx, orig), trans in zip(batch, decoded):
                results[idx] = trans
                cache.put(orig, lang, trans)

    return results


def translate_to_english(
    df: pd.DataFrame,
    backend: str,
    cache: TranslationCache,
) -> pd.DataFrame:
    """Add text_en column, translating non-English rows."""
    df["text_en"] = df["text_clean"].copy()
    mask = (df["lang"] != "en") & (df["lang"] != "und")
    non_en = df.loc[mask]

    if non_en.empty:
        log.info("No non-English rows to translate.")
        return df

    log.info("Translating %d non-English rows (backend=%s)…", len(non_en), backend)

    if backend == "none":
        log.info("Translation disabled; text_en = text_clean for non-English rows.")
        return df

    translators = {
        "openai": _translate_openai,
        "gcp": _translate_gcp,
        "hf": _translate_hf,
    }
    translate_fn = translators.get(backend)
    if translate_fn is None:
        log.error("Unknown translation backend: %s", backend)
        return df

    try:
        translated = translate_fn(
            non_en["text_clean"].tolist(),
            non_en["lang"].tolist(),
            cache,
        )
        df.loc[mask, "text_en"] = translated
    except (ImportError, EnvironmentError) as exc:
        log.warning("Translation unavailable: %s. text_en = text_clean.", exc)

    return df


# ---------------------------------------------------------------------------
# ASCII folding
# ---------------------------------------------------------------------------

def ascii_fold(text: str) -> str:
    """Transliterate to ASCII (café → cafe)."""
    try:
        from unidecode import unidecode
        return unidecode(text)
    except ImportError:
        log.warning("unidecode not installed — skipping ASCII fold.")
        return text


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def clean_text(
    text: str,
    *,
    boilerplate_patterns: Optional[List[re.Pattern]] = None,
    num_mode: str = "keep",
    do_negation_scope: bool = False,
    negation_window: int = 3,
    model_type: str = "transformer",
    do_ascii_fold: bool = True,
) -> str:
    """Apply the full cleaning pipeline to a single text string."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_unicode(text)
    text = remove_boilerplate(text, boilerplate_patterns)
    text = mask_entities(text, num_mode=num_mode)
    text = compress_punctuation(text)
    text = handle_emojis(text)

    if do_negation_scope:
        text = negation_scope(text, window=negation_window)

    if model_type == "classic":
        text = apply_stopwords_lemma(text)

    if do_ascii_fold:
        text = ascii_fold(text)

    # Final whitespace cleanup
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Boilerplate config loader
# ---------------------------------------------------------------------------

def load_boilerplate_patterns(config_path: Optional[str]) -> List[re.Pattern]:
    """Load regex patterns from JSON config file.

    Expected format: {"patterns": ["regex1", "regex2", ...]}
    """
    if not config_path or not pathlib.Path(config_path).exists():
        return []
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    patterns = [re.compile(p, re.IGNORECASE) for p in data.get("patterns", [])]
    log.info("Loaded %d boilerplate patterns from %s", len(patterns), config_path)
    return patterns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean text data for sentiment analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Path to input file (CSV/XLSX/JSONL/Parquet)")
    p.add_argument("--text-col", default="text",
                   help="Column name containing text (default: text)")
    p.add_argument("--num-mode", choices=["keep", "mask"], default="keep",
                   help="Number handling: keep or replace with <NUM>")
    p.add_argument("--model-type", choices=["transformer", "classic"],
                   default="transformer",
                   help="Target model type (affects stopword/lemma)")
    p.add_argument("--negation-scope", action="store_true",
                   help="Enable negation-scope joining (for classic models)")
    p.add_argument("--negation-window", type=int, default=3,
                   help="Number of tokens after negation cue to join (default: 3)")
    p.add_argument("--drop-dupes", action="store_true",
                   help="Drop exact-duplicate text rows")
    p.add_argument("--near-dupes", action="store_true",
                   help="Also drop near-duplicate rows via MinHash/LSH")
    p.add_argument("--near-dupe-threshold", type=float, default=0.85,
                   help="Jaccard threshold for near-dupe detection (default: 0.85)")
    p.add_argument("--min-tokens", type=int, default=2,
                   help="Drop rows with fewer tokens (default: 2)")
    p.add_argument("--max-chars", type=int, default=2000,
                   help="Truncate text_clean to this many chars (default: 2000)")
    p.add_argument("--boilerplate-config",
                   help="Path to JSON file with boilerplate regex patterns")
    p.add_argument("--translate-backend",
                   choices=["openai", "gcp", "hf", "none"], default="none",
                   help="Translation backend for non-English text (default: none)")
    p.add_argument("--ascii-fold", action="store_true", default=True,
                   help="Transliterate diacritics to ASCII (default: ON)")
    p.add_argument("--no-ascii-fold", action="store_false", dest="ascii_fold",
                   help="Disable ASCII folding")
    p.add_argument("--out-format", choices=["csv", "parquet"], default="csv",
                   help="Output file format (default: csv)")
    p.add_argument("--output", default=None,
                   help="Output file path (default: <input_stem>_cleaned.<format>)")
    p.add_argument("--cache-db", default="translation_cache.db",
                   help="SQLite cache path for translations")
    p.add_argument("--remove-bots", action="store_true", default=True,
                   help="Remove bot posts/comments (default: ON)")
    p.add_argument("--no-remove-bots", action="store_false", dest="remove_bots",
                   help="Disable bot post removal")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(
    args: argparse.Namespace,
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    bots_removed: int,
    dupes_removed: int,
    near_dupes_removed: int,
    short_removed: int,
    translate_failures: int,
    lang_dist: Dict[str, int],
    elapsed: float,
) -> Dict[str, Any]:
    """Build a JSON-serialisable report dict."""
    return {
        "cleaning_version": CLEANING_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input),
        "text_column": args.text_col,
        "rows_input": len(df_in),
        "rows_output": len(df_out),
        "bots_removed": bots_removed,
        "exact_duplicates_removed": dupes_removed,
        "near_duplicates_removed": near_dupes_removed,
        "short_text_removed": short_removed,
        "translation_backend": args.translate_backend,
        "translation_failures": translate_failures,
        "language_distribution": lang_dist,
        "settings": {
            "num_mode": args.num_mode,
            "model_type": args.model_type,
            "negation_scope": args.negation_scope,
            "min_tokens": args.min_tokens,
            "max_chars": args.max_chars,
            "ascii_fold": args.ascii_fold,
            "drop_dupes": args.drop_dupes,
            "near_dupes": args.near_dupes,
            "remove_bots": args.remove_bots,
        },
        "elapsed_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    t0 = time.time()

    # --- Read ---
    df = read_input(args.input)
    df_in = df.copy()

    if args.text_col not in df.columns:
        log.error("Column '%s' not found. Available: %s",
                  args.text_col, list(df.columns))
        sys.exit(1)

    # --- Preserve raw ---
    df["text_raw"] = df[args.text_col].astype(str)

    # --- Bot removal ---
    bots_removed = 0
    if args.remove_bots:
        bot_mask = df["text_raw"].apply(is_bot_post)
        bots_removed = bot_mask.sum()
        if bots_removed:
            log.info("Removing %d bot posts/comments.", bots_removed)
            df = df[~bot_mask].reset_index(drop=True)

    # --- Load boilerplate patterns ---
    bp_patterns = load_boilerplate_patterns(args.boilerplate_config)

    # --- Clean ---
    log.info("Cleaning %d rows…", len(df))
    df["text_clean"] = df["text_raw"].apply(
        clean_text,
        boilerplate_patterns=bp_patterns,
        num_mode=args.num_mode,
        do_negation_scope=args.negation_scope,
        negation_window=args.negation_window,
        model_type=args.model_type,
        do_ascii_fold=args.ascii_fold,
    )

    # --- Truncation ---
    truncated_mask = df["text_clean"].str.len() > args.max_chars
    df["truncated"] = truncated_mask
    df["text_clean"] = df["text_clean"].str[: args.max_chars]

    # --- Deduplication ---
    dupes_removed = 0
    near_dupes_removed = 0
    if args.drop_dupes:
        before = len(df)
        df = drop_exact_dupes(df, col="text_raw")
        dupes_removed = before - len(df)

    if args.near_dupes:
        before = len(df)
        df = drop_near_dupes_minhash(df, col="text_clean",
                                     threshold=args.near_dupe_threshold)
        near_dupes_removed = before - len(df)

    # --- Quality filter: min tokens ---
    short_removed = 0
    token_counts = df["text_clean"].str.split().str.len().fillna(0)
    short_mask = token_counts < args.min_tokens
    short_removed = short_mask.sum()
    if short_removed:
        log.info("Dropping %d rows with < %d tokens.", short_removed, args.min_tokens)
        df = df[~short_mask].reset_index(drop=True)

    # --- Language detection ---
    log.info("Detecting languages…")
    df["lang"] = detect_language(df["text_clean"])

    # --- Translation ---
    cache = TranslationCache(db_path=args.cache_db)
    translate_failures = 0
    try:
        df = translate_to_english(df, backend=args.translate_backend, cache=cache)
    except Exception as exc:
        log.warning("Translation failed: %s", exc)
        df["text_en"] = df["text_clean"]
        translate_failures = len(df[df["lang"] != "en"])
    finally:
        cache.close()

    # --- Cleaning version ---
    df["cleaning_version"] = CLEANING_VERSION

    # --- Language distribution ---
    lang_dist = {k: int(v) for k, v in df["lang"].value_counts().items()}
    log.info("Language distribution: %s", lang_dist)

    # --- Output ---
    stem = args.output or str(pathlib.Path(args.input).with_suffix(""))
    out_path = write_output(df, stem, args.out_format)

    # --- Report ---
    elapsed = time.time() - t0
    report = build_report(
        args, df_in, df, bots_removed, dupes_removed, near_dupes_removed,
        short_removed, translate_failures, lang_dist, elapsed,
    )
    report_path = f"{stem}_cleaned_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    log.info("Report → %s", report_path)
    log.info("Done in %.1fs.", elapsed)


if __name__ == "__main__":
    main()
