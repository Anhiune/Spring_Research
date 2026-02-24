# README – Tesla Sentiment and Return Analysis  

This repository contains the data files used in the analysis of sentiment and Tesla’s daily stock returns, as documented in the final research report. The files are organized to support replication of the empirical results, including preprocessing, sentiment construction, and econometric modeling.  

---

## Data Files  

### 1. **ARIMAX modeling (version2).csv**  
This is the **master modeling dataset** used for regression and ARIMAX estimation. It contains aligned daily Tesla returns, log-transformed fundamentals, trading volume, and sentiment indices. This dataset is the primary input for all econometric analyses.  

### 2. **Data pull from Yahoo API.xlsx**  
Contains raw financial data directly retrieved from the Yahoo Finance API. Includes adjusted closing prices, daily trading volumes, earnings per share (EPS), price-to-sales ratios, and other firm-level fundamentals prior to transformation.  

### 3. **daily_tesla_sentiment.csv**  
Aggregated sentiment scores by day, constructed from Twitter and media text data. Each row corresponds to one trading day and includes the composite sentiment index (positive − negative) along with additional metadata.  

### 4. **now_newest_cleand_tesla_tweets.csv**  
Cleaned tweet-level dataset containing Tesla-related social media posts. Each row represents an individual tweet, with associated sentiment scores derived from the NRC lexicon. Used as the basis for constructing daily sentiment indices.  

### 5. **tesla_7days_debugged_2025-08-27.csv**  
A smaller **debugging dataset** covering a seven-day window. This file was used for troubleshooting and verifying the data collection pipeline and preprocessing scripts.  

### 6. **google_api_with_sentiment.csv**  
This dataset merges **financial data pulled via the Google Finance API** with sentiment measures derived from social media and news text. Each row corresponds to one trading day and includes both market indicators (prices, returns, volume, fundamentals) and sentiment indices. This file served as an **intermediate step** during preprocessing and was used to validate the consistency of sentiment scoring across different API sources.  

---

## Supporting File  

### 7. **README.md**  
This file, providing an overview of the dataset contents and their role in the analysis.  

---

## Data Provenance and Integrity  

- **Financial data** were collected via the Yahoo Finance and Google Finance APIs, using historical releases to avoid look-ahead bias.  
- **Sentiment data** were sourced from Twitter and media posts, filtered for Tesla-related terms, and processed using lexicon-based methods (NRC).  
- All datasets were aligned by trading day to ensure consistency between financial returns and sentiment measures.  

---

## Usage Notes  

- The master dataset `ARIMAX modeling (version2).csv` should be used for replication of regression and ARIMAX results.  
- Raw and intermediate files (`daily_tesla_sentiment.csv`, `now_newest_cleand_tesla_tweets.csv`, `Data pull from Yahoo API.xlsx`, `google_api_with_sentiment.csv`) are included for transparency and preprocessing verification.  
- Debugging files such as `tesla_7days_debugged_2025-08-27.csv` are optional and not required for reproducing the final reported results.  

---
