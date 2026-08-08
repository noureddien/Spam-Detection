import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "models"))

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from predict import classify_message

st.set_page_config(
    page_title="Spam Detector",
    page_icon="Mail",
    layout="wide"
)

st.title("Spam Message Detector")
st.write("Type or paste any message (SMS / Email) and we'll tell you if it's Spam or not.")

with st.sidebar:
    st.header("About the Model")
    st.markdown("""
    - Model: Multinomial Naive Bayes  
    - Vectorizer: TF-IDF (3000 features)  
    - Accuracy: 97.5%  
    - ROC-AUC: 0.99
    """)

message = st.text_area("Enter your message here", height=150, placeholder="e.g. WINNER! You've won a $1000 prize...")

if st.button("Check Message", type="primary"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Analyzing"):
            result = classify_message(message)

        if result == "Spam":
            st.error(f"Result: {result}")
        else:
            st.success(f"Result: {result}")

st.divider()
st.header("Model Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "97.49%")
col2.metric("Precision", "98.17%")
col3.metric("Recall", "81.68%")
col4.metric("ROC-AUC", "0.9901")

DATA_PATH = BASE_DIR / "data" / "spam.csv"
CM_PATH = BASE_DIR / "models" / "confusion_matrix.png"

col_left, col_right = st.columns(2)

df = None
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    df = df.iloc[:, :2]
    df.columns = ['label', 'text']
    df = df.drop_duplicates(keep='first')

with col_left:
    st.subheader("Class Distribution")
    if df is not None:
        counts = df['label'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#E53935'])
        ax.set_title('Ham vs Spam')
        st.pyplot(fig)
    else:
        st.info("Dataset not found at data/spam.csv")

with col_right:
    st.subheader("Confusion Matrix")
    if CM_PATH.exists():
        st.image(str(CM_PATH))
    else:
        st.info("Confusion matrix image not found at models/confusion_matrix.png")

st.subheader("Message Length Distribution")
LENGTH_DIST_PATH = BASE_DIR / "images" / "image.png"
if LENGTH_DIST_PATH.exists():
    st.image(str(LENGTH_DIST_PATH))
else:
    st.info("Image not found at images/image.png")

st.markdown("---")
st.caption("Built with scikit-learn + Streamlit | Model: Multinomial Naive Bayes")
