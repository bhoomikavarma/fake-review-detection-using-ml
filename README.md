
# 🕵️‍♂️ Fake Online Review Detection using Machine Learning

This project aims to detect and classify fake online reviews using machine learning and natural language processing (NLP) techniques. It helps improve trust on platforms like Amazon, Yelp, and TripAdvisor by flagging suspicious user-generated content.

---

## 📌 Features

- Classifies reviews as **Genuine** or **Fake**
- Uses **TF-IDF** vectorization for text representation
- Supports **Logistic Regression**, **Random Forest**, and **SVM** models
- Frontend built with **Streamlit** for real-time prediction
- Clean and modular codebase with preprocessing, training, and deployment modules
- Integrated reasoning logic to explain predictions

---

## 🧰 Technologies Used

- Python 3.x
- Scikit-learn
- NLTK
- Pandas
- NumPy
- TF-IDF
- Streamlit
- Flask (optional for API)
- SMOTE (imbalanced data handling)
- Joblib (model persistence)

---

## 📁 Project Structure
fake-review-detection-ml/
├── app/
│ └── main.py # Streamlit frontend
├── model/
│ ├── logistic_model.pkl # Trained model
│ └── tfidf_vectorizer.pkl # Trained vectorizer
├── data/
│ └── reviews.csv # Sample dataset
├── utils/
│ └── preprocessing.py # Text cleaning functions
├── train_model.py # Model training script
├── requirements.txt # Project dependencies
├── screenshots/
│ ├── fake_output.png
│ └── genuine_output.png
├── LICENSE
└── README.md

## ⚙️ How to Run

1. **Clone the Repository**
```bash
git clone https://github.com/bhoomikavarma/fake-review-detection-ml.git
cd fake-review-detection-ml
Install Requirements
pip install -r requirements.txt
Run the Streamlit App

streamlit run app/main.py
🧪 Sample Output
<img width="914" height="516" alt="image" src="https://github.com/user-attachments/assets/d5881d7c-54a5-4d58-ba1b-63da345baca4" />




📊 Model Performance
Accuracy: ~90%

Evaluation Metrics: Precision, Recall, F1-Score

Trained using TF-IDF + Logistic Regression

Balanced dataset using SMOTE

📈 Future Enhancements
Use of BERT, LSTM for deeper language understanding

Real-time API integration with e-commerce platforms

Browser plugin or mobile app version

Multilingual support

Feedback-driven self-learning model

📂 Dataset Source
Yelp Open Dataset

Amazon Reviews Dataset

👩‍💻 Author
Bhoomika
https://github.com/bhoomikavarma/fake-review-detection-using-ml/


📄 License
This project is licensed under the MIT License.

🙏 Acknowledgements
Scikit-learn, NLTK

Streamlit.io

Public datasets from Amazon, Yelp




