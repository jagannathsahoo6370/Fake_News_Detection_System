import streamlit as st
import joblib
import numpy as np
import os
from PyPDF2 import PdfReader

# Page Config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "static", "style.css")

    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Logo
logo_path = os.path.join("assets", "logo.png")

col1, col2 = st.columns([1, 6])

with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)

with col2:
    st.title("📰 AI-Powered Fake News Detection System")
    st.caption("Machine Learning Based News Verification")


# Load model
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

# Initialize prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.title("📰 AI-Powered Fake News Detection & Verification System")
st.caption("Machine Learning + NLP + AI Verification")

st.markdown("---")

# Sidebar Info
st.sidebar.title("🤖 AI Fake News Detector")

st.sidebar.markdown("---")

st.sidebar.write("### Model")
st.sidebar.success("Linear SVM + TF-IDF")

st.sidebar.write("### Accuracy")
st.sidebar.info("99.5%")

st.sidebar.write("### Dataset")
st.sidebar.info("44,898 News Articles")

st.sidebar.write("### Developer")
st.sidebar.success("Jagannath Sahoo")

st.sidebar.markdown("---")

# User Input
st.subheader("📂 Upload News File (Optional)")

uploaded_file = st.file_uploader(
    "Upload a News File",
    type=["txt", "pdf"]
)

user_input = ""

if uploaded_file is not None:

    if uploaded_file.type == "text/plain":
        user_input = uploaded_file.read().decode("utf-8")

    elif uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)

        for page in pdf.pages:
            user_input += page.extract_text() + "\n"

st.subheader("📝 Enter News Article")

user_input = st.text_area(
    label="Paste the complete news article below:",
    value=user_input if uploaded_file else "",
    height=250,
    placeholder="Paste the complete news article here..."
)

if st.button("🔍 Analyze News", use_container_width=True):

    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        with st.spinner("🤖 AI is analyzing the news..."):

            text_vectorized = vectorizer.transform([user_input])

            prediction = model.predict(text_vectorized)[0]

            decision_score = model.decision_function(text_vectorized)[0]

            confidence = min(abs(decision_score) * 10, 100)

            st.markdown("---")
            st.subheader("📊 Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.success("✅ REAL NEWS")
                else:
                    st.error("❌ FAKE NEWS")

            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")
                result = "REAL NEWS ✅" if prediction == 1 else "FAKE NEWS ❌"
                st.session_state.history.append({
                "Prediction": result,
                "Confidence": f"{confidence:.2f}%"
                })
                st.progress(min(confidence / 100, 1.0))

            st.markdown("### 🤖 AI Analysis")

            if prediction == 1:
                st.info(
        "The article's writing style closely matches patterns found in authentic news articles within the training dataset."
    )
            else:
                st.warning(
        "The article's writing style closely matches patterns found in fake news articles within the training dataset."
    )
            st.markdown("### 📋 Try Sample News")

    sample = st.selectbox(
             "Choose a sample article",
    [
        "None",
        "India successfully launched a new weather satellite from the Satish Dhawan Space Centre using PSLV.",
        "Scientists discovered a magical fruit that makes humans live 500 years."
    ]
)

    if sample != "None":
            user_input = sample

            st.markdown("### 🤖 AI Explanation")

            if prediction == 0:
                st.info(
                    """
                The language patterns in this article are similar to thosefound in fake news articles from the training dataset.

                ⚠ This is an AI prediction and should not be treated asfactual verification.""")
            else:
                st.info(
                    """
                    The language patterns in this article are similar to thosefound in trusted news articles from the training dataset.

                    ⚠ This is an AI prediction and should not be treated asfactual verification.""")

            st.progress(min(confidence / 100, 1.0))

st.subheader("📊 Model Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Linear SVM")

with col2:
    st.metric("Dataset", "44,898")

with col3:
    st.metric("Accuracy", "99.5%")

st.markdown("---")
st.subheader("📜 Prediction History")

if st.session_state.history:
    st.table(st.session_state.history)
else:
    st.info("No predictions yet.")