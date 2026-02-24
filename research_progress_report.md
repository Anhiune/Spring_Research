# UROP Research Progress Report
**Topic:** Export-Control / Geopolitical-Risk Sentiment and Semiconductor Stock Risk Premiums  
**Date:** February 24, 2026  
**Author:** Anh Bui — University of St. Thomas

---

## 1. What Has Been Accomplished

### 1.1 Bluesky Scraper — Built and Running

A full automated Bluesky scraper was built from scratch using the `atproto` Python SDK.  
The scraper lives in `bluesky_scapper/` and has two modes:

| Script | Purpose |
|---|---|
| `scrape_bluesky.py` | Ad-hoc scrape for any custom date range and keyword list |
| `scrape_bluesky_batch.py` | **Month-by-month batch scraper** — loops through every calendar month and saves one CSV per month |

Key engineering features:
- **Auto-resume** — skips months whose CSV already exists, so the job is safe to restart after interruptions
- **Rate-limit handling** — detects HTTP 429 responses, parses the `ratelimit-reset` timestamp from the API, and waits exactly until the window resets before retrying
- **Session re-login** — transparently re-authenticates if the session token expires mid-run and backs off with increasing delays on persistent failures
- **Deduplication** — posts are deduplicated by `uri` within each month so the same post is never double-counted even if it matches multiple keywords
- Output to `bluesky_data/` — one CSV per month

### 1.2 Keyword Dictionary — 55 Research-Specific Keywords

Keywords are organized into five thematic groups in `bluesky_scapper/config/keywords.txt`:

| Group | Examples | Count |
|---|---|---|
| Tickers | NVDA, AMD, INTC, QCOM, TSM, ASML, MU, AMAT, LRCX, KLAC, AVGO, MRVL | 12 |
| Company names | NVIDIA, Intel, Qualcomm, TSMC, Micron, Applied Materials, Lam Research, Broadcom | 8 |
| Key products | H100, H20 chip, A100 chip, Blackwell GPU, EUV lithography, HBM memory, CoWoS | 7 |
| Policy/institutions | BIS export, Entity List semiconductor, Foreign Direct Product Rule, ECCN chip | 6 |
| Policy actions | chip export ban, chip restriction China, semiconductor sanction, license requirement | 7 |
| Geopolitical actors | Huawei chip, SMIC semiconductor, YMTC, China semiconductor, Taiwan Strait | 10 |
| Severity/tone | sweeping chip ban, major export restriction, semiconductor loophole, chip waiver | 5 |

### 1.3 Data Collected — Current Snapshot (as of Feb 24, 2026)

The background scrape is still running. What is confirmed saved:

| Period | Files | Posts |
|---|---|---|
| Jan 2023 – Dec 2023 | 12 monthly CSVs | ~4,500 posts |
| Jan 2024 – May 2024 | 5 monthly CSVs | ~4,237 posts |
| **Total** | **17 CSVs** | **~8,737 posts** |

The scraper is continuing in the background through Feb 2026. Estimated final total: **~20,000–30,000 posts** across 38 months.

---

## 2. Data Description

### 2.1 Schema — Each CSV Has 11 Columns

| Column | Type | Description |
|---|---|---|
| `uri` | string | Unique AT Protocol post identifier |
| `cid` | string | Content ID (immutable hash of post content) |
| `author_handle` | string | Bluesky handle (e.g. `user.bsky.social`) |
| `author_display_name` | string | Display name (134 nulls = accounts with no display name set — not missing data) |
| `text` | string | Raw post text (9 nulls — likely deleted posts fetched mid-index) |
| `created_at` | ISO-8601 UTC | Post creation timestamp |
| `like_count` | integer | Likes at time of scrape |
| `repost_count` | integer | Reposts at time of scrape |
| `reply_count` | integer | Replies at time of scrape |
| `quote_count` | integer | Quote-posts at time of scrape |
| `keyword` | string | Which keyword matched this post |

### 2.2 Key Descriptive Statistics

- **Date range covered:** Jan 1, 2023 → May 30, 2024 (so far)
- **Average post text length:** 159 characters (Bluesky has a 300-char limit)
- **Null rate:** `author_display_name` 1.5%, `text` 0.1% — both negligible
- **Top keywords by post volume:**

```
Intel                   1,911 posts   (22%)
NVIDIA                  1,072 posts   (12%)
Micron                    962 posts   (11%)
AMD                       776 posts    (9%)
MU (ticker)               717 posts    (8%)
AMAT                      515 posts    (6%)
TSMC                      515 posts    (6%)
Qualcomm                  513 posts    (6%)
ASML                      388 posts    (4%)
NVDA (ticker)             382 posts    (4%)
China semiconductor        95 posts    (1%)
H100                       94 posts    (1%)
Blackwell GPU              70 posts    (1%)
```

**Notable:** Company name keywords (Intel, NVIDIA) yield far more posts than
policy/geopolitical keywords (China semiconductor, H100). This is expected — it
confirms that Bluesky sentiment is largely company/product-focused, and policy-specific
language is rarer. For modeling, both channels matter: high-volume firm-level posts
provide the baseline; low-volume policy posts are the *treatment*.

### 2.3 Temporal Pattern

Bluesky user base grew significantly in late 2023 (after the Twitter/X exodus accelerated).
Post volume in Aug–Sep 2023 (~1,826 and ~1,361 posts) is 10x higher than early 2023 (~86–178),
which is consistent with platform growth. This means early 2023 data is sparse and
models using pre-Aug 2023 data should apply lower weights or exclude.

---

## 3. Data Cleaning Plan (Critical Step)

Cleaning must happen before any feature engineering or modeling. Work through these
steps in order.

### Step 3.1 — Merge and Standardize All Monthly CSVs

```python
import pandas as pd, glob

files = sorted(glob.glob("bluesky_data/bluesky_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
df["date"] = df["created_at"].dt.date   # trading-day merge key
df.to_parquet("bluesky_data/bluesky_all_raw.parquet", index=False)   # faster for large data
```

### Step 3.2 — Drop Nulls and Deduplicate

Even though the scraper deduplicates within each month, the same post could
theoretically appear across two monthly files at a month boundary.

```python
# Drop null texts (9 rows), deduplicate globally by uri
df = df.dropna(subset=["text"])
df = df.drop_duplicates(subset="uri")

print(f"Clean dataset: {len(df):,} rows")
```

### Step 3.3 — Language Filter (English Only)

Export-control discourse is predominantly English. Non-English posts add noise.

```python
# pip install langdetect
from langdetect import detect

def safe_detect(text):
    try:
        return detect(text)
    except:
        return "unknown"

df["lang"] = df["text"].apply(safe_detect)
df = df[df["lang"] == "en"].copy()
print(f"After language filter: {len(df):,} rows")
```

### Step 3.4 — Bot / Spam Filtering

```python
# 1. Remove accounts posting more than a reasonable daily maximum (likely bots)
daily_post_counts = df.groupby(["author_handle", "date"]).size()
bot_handles = daily_post_counts[daily_post_counts > 30].index.get_level_values("author_handle").unique()
df = df[~df["author_handle"].isin(bot_handles)]

# 2. Remove exact-duplicate post texts (copy-paste spam)
df = df.drop_duplicates(subset=["text", "author_handle"])

# 3. Remove posts that are almost entirely hashtags/tickers (no real content)
df["word_count"] = df["text"].str.split().str.len()
df = df[df["word_count"] >= 5]
```

### Step 3.5 — Relevance Filtering (Critical for Policy Channel)

Not all posts mentioning "Intel" or "AMD" are about export controls.
You need a relevance filter to separate:
- **Tier 1:** Posts that mention BOTH a company AND a policy/geopolitical keyword
  → highest relevance for your research question
- **Tier 2:** Posts mentioning a company but with no policy context
  → useful for general sentiment baseline
- **Tier 3:** Posts matched only on a ticker symbol that might be noise
  (e.g., "intc" as a word in another context)

```python
POLICY_TERMS = [
    "export control", "ban", "sanction", "restriction", "BIS", "entity list",
    "license", "ECCN", "FDP", "Huawei", "SMIC", "China", "PRC", "Taiwan",
    "decoupling", "reshoring", "blacklist", "waiver", "exemption"
]

def has_policy_context(text):
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in POLICY_TERMS)

df["has_policy_context"] = df["text"].apply(has_policy_context)
df["tier"] = df["has_policy_context"].map({True: 1, False: 2})

print("Tier 1 (policy-relevant):", (df["tier"]==1).sum())
print("Tier 2 (company, no policy):", (df["tier"]==2).sum())
```

### Step 3.6 — Text Normalization

```python
import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)             # remove @mentions
    text = re.sub(r"#(\w+)", r"\1", text)        # strip # but keep word
    text = re.sub(r"\s+", " ", text).strip()     # normalize whitespace
    return text

df["text_clean"] = df["text"].apply(clean_text)
```

### Step 3.7 — Assign Trading Date (Calendar → Business Day)

Bluesky posts are timestamped in UTC. You need to map each post to the correct
**trading day** for the event study and forecasting models.

```python
import pandas_market_calendars as mcal   # pip install pandas-market-calendars

nyse = mcal.get_calendar("NYSE")
schedule = nyse.schedule(start_date="2023-01-01", end_date="2026-03-01")
trading_days = pd.DatetimeIndex(schedule.index.date)

def to_trading_day(dt):
    """
    Posts published after 4pm ET or on weekends/holidays
    belong to the NEXT trading day (to avoid look-ahead bias).
    """
    ET_CLOSE = pd.Timestamp("16:00:00")
    dt_et = dt.astimezone("US/Eastern")
    date = dt_et.date()
    time = dt_et.time()

    # If after market close or non-trading day, roll forward
    candidate = pd.Timestamp(date)
    if time >= ET_CLOSE.time() or candidate not in trading_days:
        idx = trading_days.searchsorted(candidate, side="right")
        if idx < len(trading_days):
            return trading_days[idx]
        return pd.NaT
    return date

df["trading_date"] = df["created_at"].apply(to_trading_day)
```

> **This step is the most important for avoiding look-ahead bias.** Sentiment on
> day t must only be used to predict prices from t+1 onward.

### Step 3.8 — Save Clean Dataset

```python
df.to_parquet("bluesky_data/bluesky_all_clean.parquet", index=False)
df.to_csv("bluesky_data/bluesky_all_clean.csv", index=False)
print(f"Final clean dataset: {len(df):,} posts")
```

---

## 4. Sentiment Feature Engineering Plan

After cleaning, build daily sentiment indices from the post text.

### Step 4.1 — Lexicon-Based Scoring (FinBERT or NRC)

Since your existing Tesla model uses the **NRC emotion lexicon** (Bollen method),
use the same approach for continuity — but also add a financial-domain model:

**Option A — NRC Lexicon (consistent with existing work):**
```python
# Same approach as Practice with Bollen method_sentiment_analysis_step3.py
# Produces: positive, negative, anger, fear, anticipation, trust, etc.
```

**Option B — FinBERT (recommended for financial text):**
```python
# pip install transformers torch
from transformers import pipeline
finbert = pipeline("text-classification", model="ProsusAI/finbert")
# Returns: positive / negative / neutral probabilities per post
```

**Recommendation:** Run both. Use FinBERT as primary (it understands financial jargon
like "license denied", "entity list") and NRC as a robustness check.

### Step 4.2 — Daily Aggregation (Per Ticker)

```python
# For each ticker, aggregate sentiment on each trading_date
# Weight by engagement (likes + reposts) to upweight influential posts

def weighted_sentiment(group):
    weight = group["like_count"] + group["repost_count"] + 1  # +1 avoids zero weights
    return (group["sentiment_score"] * weight).sum() / weight.sum()

daily_sentiment = (
    df.groupby(["ticker_matched", "trading_date"])
    .apply(weighted_sentiment)
    .reset_index(name="sentiment_index")
)
```

### Step 4.3 — Policy-Risk Sentiment Index (Your Core Variable)

Separate from the general sentiment, build a **daily policy-risk index** using
only Tier 1 (policy-context) posts. This is the key explanatory variable for H1 and H2.

```python
policy_df = df[df["tier"] == 1]
policy_sentiment = (
    policy_df.groupby("trading_date")["sentiment_score"]
    .agg(["mean", "count", "std"])
    .rename(columns={"mean": "policy_risk_sentiment",
                     "count": "policy_post_count",
                     "std": "policy_sentiment_vol"})
)
```

---

## 5. Empirical Analysis Plan

### Step 5.1 — Pull Stock Prices for the Semiconductor Universe

```python
import yfinance as yf

tickers = ["NVDA", "AMD", "INTC", "QCOM", "TSM", "ASML", "MU", "AMAT",
           "LRCX", "KLAC", "AVGO", "MRVL", "SOXX"]  # SOXX = ETF benchmark

prices = yf.download(tickers, start="2023-01-01", end="2026-03-01", auto_adjust=True)["Close"]
log_returns = prices.pct_change().apply(lambda x: x)   # or log(p_t/p_{t-1})
```

### Step 5.2 — Assign Exposure Scores to Each Firm

Manually score each firm (0–3) based on:
- Revenue exposure to China
- Products on the BIS restricted list
- Supply chain dependency on Taiwan/TSMC

| Ticker | Role | Exposure Score |
|---|---|---|
| NVDA | Fabless (AI chips) | 3 — H100/H20 restrictions hit directly |
| ASML | Equipment | 3 — EUV export ban to China |
| MU | Memory/IDM | 2 — YMTC competition + some China revenue |
| INTC | Fabless/IDM | 2 — China revenue + some entity list exposure |
| QCOM | Fabless | 2 — Huawei license issues |
| AMD | Fabless (GPU) | 2 — MI300 export restrictions |
| AMAT | Equipment | 2 — China is largest market |
| LRCX | Equipment | 2 — China revenue |
| KLAC | Equipment | 1 |
| AVGO | Fabless | 1 |
| MRVL | Fabless | 1 |

### Step 5.3 — Test A: Event Study

1. Build your policy-event calendar (BIS rule dates, Entity List additions, major
   Congressional actions).
2. For each event, compute Cumulative Abnormal Returns (CARs) in windows
   [-1, +1], [-1, +3], [-3, +3] days using the market model.
3. Test whether pre-event policy-risk sentiment (averaged over days [-10, -1])
   predicts the size of CARs. Run:

```r
# In R — consistent with your existing ARIMAX code style
lm(CAR ~ pre_event_policy_sentiment * exposure_score + log_mktcap + beta, data=event_df)
```

### Step 5.4 — Test B: ARIMAX Volatility Forecasting

Extend your existing **ARIMAX model** from the Submition folder to the new context:

**Current model (Tesla, from Submition):**
```
adj_return ~ AR(2) + L1_return + L2_return + L1_sentiment + L2_sentiment
           + log_adj_close + log_volume + log_p_s_ttm + log_eps_ttm
```

**New model (Semiconductor, volatility target):**
```
realized_vol_{t+5} ~ AR(2) + L1_realized_vol + L2_realized_vol
                   + L1_policy_risk_sentiment + L2_policy_risk_sentiment
                   + L1_return + log_volume + vix
                   + exposure_score × L1_policy_risk_sentiment
```

Where `realized_vol_{t+5}` = standard deviation of log returns over the next 5 trading days.

The interaction term `exposure_score × L1_policy_risk_sentiment` directly tests H3
(heterogeneous effects).

### Step 5.5 — Test C: Granger Causality (Lead/Lag)

Consistent with your existing `granger test.csv` and Granger test R code:

```r
# Using your existing structure from "Data file for Granger test.csv"
library(vars)
VARselect(cbind(policy_risk_sentiment, realized_vol), lag.max=10)
var_model <- VAR(cbind(policy_risk_sentiment, realized_vol), p=2)
causality(var_model, cause="policy_risk_sentiment")
```

---

## 6. Future Extensions to the Existing ARIMAX Model (Submition Folder)

The existing model in `Submition/` uses Tesla tweets as its sentiment source.
Here are concrete extensions ranked by priority:

### Priority 1 — Replace/Supplement the Sentiment Source

The current `sentiment_index` comes from Tesla tweets via the Twitter v2 API.
You can now augment it with Bluesky data (for 2023 onward), using the same NRC
lexicon method already implemented in `Practice with Bollen method_sentiment_analysis_step3.py`.
This adds a **second, independent social media signal** and allows you to test
whether cross-platform sentiment agreement is more predictive than either alone.

### Priority 2 — Add a Policy-Risk Sentiment Factor to the Return Equation

The current ARIMAX has `L1_sentiment` and `L2_sentiment` but no explicit policy-risk signal.
Add a `policy_risk_index` (built from Tier 1 posts) as a new regressor:

```r
xreg = as.matrix(model_df[, c(
  "log_adj_close", "log_volume", "log_p_s_ttm_x", "log_eps_ttm_x",
  "L1_return", "L2_return",
  "L1_sentiment", "L2_sentiment",
  "L1_policy_risk",   # <-- NEW
  "L2_policy_risk"    # <-- NEW
)])
fit_arimax_new = Arima(y, order=c(2,0,0), xreg=xreg)
```

Compare AIC/BIC between the old and new models. If `L1_policy_risk` is significant,
that directly answers H2.

### Priority 3 — Expand to Realized Volatility as the Dependent Variable

Your current model predicts daily **returns**. Switch (or add) **realized volatility**
as the dependent variable. This is more consistent with H2 and better-suited to
ARIMAX (volatility is roughly stationary without differencing, and the ARCH test you
already run on residuals motivates this).

```r
# Build 5-day realized vol as new dependent variable
model_df$rv5 = rollapply(model_df$adj_return, width=5, FUN=sd, fill=NA, align="right")
# Then fit ARIMAX with rv5 as y
```

### Priority 4 — Add GARCH Volatility Modeling (ARIMAX-GARCH)

Your existing residuals already show ARCH effects (you run `ArchTest()`). The natural
next step is an **ARIMA-GARCH** or **ARIMAX-GARCH** model to model both the conditional
mean and conditional variance:

```r
library(rugarch)
spec <- ugarchspec(
  variance.model = list(model="sGARCH", garchOrder=c(1,1)),
  mean.model     = list(armaOrder=c(2,0), include.mean=TRUE, external.regressors=xreg_matrix),
  distribution.model = "std"   # student-t for fat tails
)
fit_garch <- ugarchfit(spec, data=y)
```

### Priority 5 — Out-of-Sample Forecasting Evaluation

Currently the model is estimated in-sample. Add a rolling or expanding window
out-of-sample evaluation:

```r
# Rolling 1-step-ahead forecast
n_train <- round(0.8 * nrow(model_df))
forecasts <- numeric(nrow(model_df) - n_train)
actuals   <- numeric(nrow(model_df) - n_train)

for (i in seq_along(forecasts)) {
  train <- model_df[1:(n_train + i - 1), ]
  y_train    <- train$adj_return
  xreg_train <- as.matrix(train[, xreg_cols])
  xreg_next  <- as.matrix(model_df[n_train + i, xreg_cols])

  fit_tmp <- Arima(y_train, order=c(2,0,0), xreg=xreg_train)
  forecasts[i] <- predict(fit_tmp, newxreg=xreg_next)$pred
  actuals[i]   <- model_df$adj_return[n_train + i]
}

# RMSE and directional accuracy
rmse <- sqrt(mean((forecasts - actuals)^2))
dir_acc <- mean(sign(forecasts) == sign(actuals))
cat("RMSE:", rmse, "\nDirectional Accuracy:", dir_acc, "\n")
```

### Priority 6 — Google Trends as an Additional Sentiment Layer

You noted the possibility of adding Google Trends. This is a useful proxy for
retail/public attention:

```python
# pip install pytrends
from pytrends.request import TrendReq

pytrends = TrendReq(hl="en-US", tz=360)
pytrends.build_payload(["semiconductor export ban", "NVIDIA China", "chip war"], timeframe="2023-01-01 2026-02-28")
interest = pytrends.interest_over_time()
```

Google Trends provides a weekly or daily normalized Search Volume Index (SVI) that
you can merge by date and include alongside Bluesky sentiment.

---

## 7. Summary Checklist — What to Do Next

| # | Task | Status |
|---|---|---|
| 1 | Let batch scraper finish (2024-05 → 2026-02) | **Running** |
| 2 | Merge all CSVs into `bluesky_all_raw.parquet` | Not started |
| 3 | Run cleaning steps 3.1–3.8 | Not started |
| 4 | Language + bot filter | Not started |
| 5 | Relevance tiering (Tier 1 vs Tier 2) | Not started |
| 6 | Trading-date alignment (look-ahead bias fix) | **Critical** |
| 7 | Run FinBERT or NRC on `text_clean` | Not started |
| 8 | Build `daily_sentiment` and `policy_risk_sentiment` | Not started |
| 9 | Pull Semiconductor stock prices via `yfinance` | Not started |
| 10 | Assign firm-level exposure scores | Not started |
| 11 | Build policy-event calendar | Not started |
| 12 | Event study (Test A) | Not started |
| 13 | ARIMAX volatility model (Test B) | Not started |
| 14 | Granger causality (Test C) | Not started |
| 15 | Extend Submition ARIMAX with policy-risk regressor | Not started |
| 16 | Add GARCH layer to Submition model | Not started |
| 17 | Add Google Trends data | Optional |

---

*Document auto-generated from workspace analysis — February 24, 2026*
