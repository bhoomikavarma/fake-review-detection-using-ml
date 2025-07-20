# train_logistic_model.py

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load and preprocess data
df = pd.read_csv("data/reviews.csv")
df = df.dropna(subset=['review', 'label'])
df['label'] = df['label'].astype(int)
df['review'] = df['review'].astype(str)
df = df[df['review'].str.strip() != '']
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Load pre-trained vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
X = vectorizer.transform(df['cleaned_review'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/logistic_model.pkl")

print("✅ Logistic Regression model saved to model/logistic_model.pkl")
