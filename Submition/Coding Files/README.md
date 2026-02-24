# README – Python Scripts for Tesla Sentiment and Return Analysis  

This folder contains the Python scripts used to build, preprocess, and validate the datasets used in the Tesla sentiment–return analysis. Scripts cover data acquisition, preprocessing, sentiment scoring, and fundamental ratio pulls. Together, these files form the end-to-end pipeline that produces the modeling dataset (`ARIMAX modeling (version2).csv`).  

---

## Scripts  

### 1. **#Daily return computation.py**  
Computes daily log returns of Tesla’s adjusted closing price from raw Yahoo Finance data. Produces the `adj_return` variable used as the dependent variable in econometric models.  

### 2. **Practice with Bollen method.py**  
Initial practice script implementing the sentiment-scoring framework of Bollen et al. (2011). Demonstrates tokenization and lexicon-based sentiment mapping on Tesla-related text.  

### 3. **Practice with Bollen method_preprocessing_step2.py**  
Handles preprocessing of tweet and news text data, including tokenization, lowercasing, and stopword removal. Outputs cleaned text ready for sentiment scoring.  

### 4. **Practice with Bollen method_sentiment_analysis_step3.py**  
Performs lexicon-based sentiment scoring on preprocessed tweets. Generates sentiment features (positive, negative, and emotion categories) at the tweet level, later aggregated into daily indices.  

### 5. **google api daat pull.py**  
Implements API queries to retrieve Tesla-related tweets and associated metadata from Twitter v2. Handles pagination, rate-limiting, and saves results into `.csv` files.  

### 6. **import requests.py**  
Utility script demonstrating how to pull financial data from APIs (e.g., Yahoo or Google Finance) using the `requests` library.  

### 7. **pull ratio data from yahoo.py**  
Retrieves Tesla’s financial fundamentals (e.g., EPS, P/S ratio, P/E ratio, debt-to-equity) via Yahoo Finance API calls. Used to build explanatory variables in regression and ARIMAX models.  

---

## Workflow  

1. **Data acquisition**:  
   - `google api daat pull.py` → collects tweets.  
   - `pull ratio data from yahoo.py` and `import requests.py` → collect financial fundamentals and price data.  

2. **Preprocessing and sentiment scoring**:  
   - `Practice with Bollen method_preprocessing_step2.py` → cleans tweets.  
   - `Practice with Bollen method_sentiment_analysis_step3.py` → applies NRC lexicon to score emotions and polarity.  

3. **Return computation**:  
   - `#Daily return computation.py` → computes `adj_return`.  

4. **Integration**:  
   - Outputs from these scripts are merged into `ARIMAX modeling (version2).csv` for econometric analysis.  

---

## Usage Notes  

- Scripts labeled “Practice” represent exploratory versions of the final pipeline; they document the step-by-step learning and replication of Bollen et al. (2011).  
- The recommended workflow uses the cleaned outputs (`now_newest_cleand_tesla_tweets.csv`, `daily_tesla_sentiment.csv`) rather than rerunning every script.  
- All scripts are written for Python 3.x and rely on `pandas`, `requests`, and `nltk`/`text processing` libraries.  

---
