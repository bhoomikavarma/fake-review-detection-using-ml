# app/main.py

import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load model and vectorizer
model = joblib.load("model/logistic_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Explanation generator
def get_reason(text, prediction):
    text = text.lower()
    word_count = len(text.split())
    if prediction == 0:
        if any(word in text for word in ['buy', 'offer', 'free', 'click', 'visit', 'win']):
            return "Contains suspicious promotional keywords."
        if word_count < 5:
            return "Too short; possibly auto-generated or vague."
        return "Generic or unnatural language."
    else:
        if word_count > 15:
            return "Detailed and human-like content."
        return "Balanced and clear language."

# Streamlit UI
st.title("🕵️‍♂️ Fake Review Detector")
st.markdown("Enter a product or service review below:")

user_input = st.text_area("✍️ Write your review:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = preprocess_text(user_input)
        vect_input = vectorizer.transform([cleaned])
        prediction = model.predict(vect_input)[0]
        label = "✅ Genuine Review" if prediction == 1 else "❌ Fake Review"
        reason = get_reason(user_input, prediction)

        st.subheader("Prediction Result:")
        st.success(label)
        st.info(f"Reason: {reason}")
