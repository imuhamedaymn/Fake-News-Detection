import streamlit as st
import re
import joblib
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

# Load vectorizer and model
Vectorizer = joblib.load("./Models/Vectorizer.pkl")
Dmodel = joblib.load("./Models/DecisionTreeClassification.pkl")

# Preprocessing function
def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
        preprocessed_text.append(' '.join(
            token.lower() for token in sentence.split()
            if token.lower() not in stopwords.words('english')
        ))
    return preprocessed_text

# Prediction function
def predict_from_text(raw_text_list):
    processed_text = preprocess_text(raw_text_list)
    vectorized_text = Vectorizer.transform(processed_text)
    predictions = Dmodel.predict(vectorized_text)
    return ["Real" if pred == 0 else "Fake" for pred in predictions]

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter one or more news headlines or paragraphs to classify them as **Real** or **Fake**.")

# Text input
user_input = st.text_area("Enter one or more news sentences (separate with new lines):", height=200)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        text_list = user_input.strip().split("\n")
        predictions = predict_from_text(text_list)

        st.subheader("🔍 Results")
        for i, (text, label) in enumerate(zip(text_list, predictions), start=1):
            color = "green" if label == "Real" else "red"
            st.markdown(f"**{i}.** _{text}_  \n👉 **Prediction:** :{color}[{label}]", unsafe_allow_html=True)
