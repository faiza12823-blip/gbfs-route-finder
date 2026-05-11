import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page Title
st.set_page_config(page_title="Fake News Detector")

st.title("📰 AI-Based Fake News Detection")
st.write("Enter any news article text below:")

# User Input
news = st.text_area("Paste News Content Here")

# Prediction
if st.button("Detect News"):

    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed_news = vectorizer.transform([news])

        prediction = model.predict(transformed_news)

        if prediction[0] == "FAKE":
            st.error("This News is FAKE")
        else:
            st.success("This News is REAL")