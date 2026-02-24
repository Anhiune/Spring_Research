import pandas as pd
from collections import defaultdict

# Load NRC lexicon
nrc_path = r"C:\Users\hoang\Downloads\NRC-Emotion-Lexicon\NRC-Emotion-Lexicon\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
nrc = defaultdict(list)

with open(nrc_path, 'r') as file:
    for line in file:
        word, emotion, association = line.strip().split('\t')
        if int(association) == 1:
            nrc[word].append(emotion)

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string

def preprocess(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha()]  # remove punctuation/numbers
    return tokens

def get_emotion_scores(tokens):
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'positive', 'negative']
    score = dict.fromkeys(emotions, 0)
    for token in tokens:
        if token in nrc:
            for emotion in nrc[token]:
                score[emotion] += 1
    return score

# Load your tweet data
df = pd.read_csv(r"C:\Users\hoang\Documents\Summer Research\python (locally download file)\google_api_with_sentiment.csv")

# Process and extract scores
emotion_scores = []
for text in df['text']:
    tokens = preprocess(text)
    score = get_emotion_scores(tokens)
    emotion_scores.append(score)

# Convert to DataFrame
emotion_df = pd.DataFrame(emotion_scores)
result_df = pd.concat([df.reset_index(drop=True), emotion_df], axis=1)

# Save the output
output_path = r"C:\Users\hoang\Documents\Summer Research\python (locally download file)\tesla_tweets_with_sentiment.csv"
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("✅ Multi-dimensional sentiment scores saved.")