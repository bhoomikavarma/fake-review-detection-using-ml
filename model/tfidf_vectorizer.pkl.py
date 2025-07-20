# train_vectorizer.py

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocess function
def preprocess_text(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load and clean dataset
df = pd.read_csv("data/reviews.csv")  # ensure this file exists
df = df.dropna(subset=['review'])
df['review'] = df['review'].astype(str)
df = df[df['review'].str.strip() != '']
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
vectorizer.fit(df['cleaned_review'])

# Save the vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("✅ TF-IDF Vectorizer saved to model/tfidf_vectorizer.pkl")
