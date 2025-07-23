# app.py - Final, Enhanced UI/UX Version

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from transformers import pipeline
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Aegis AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NLTK Data Download ---
# This block will try to download necessary NLTK data if not present
try:
    stopwords.words('english')
except LookupError:
    with st.spinner("Downloading language models..."):
        nltk.download('stopwords')
        nltk.download('wordnet')

# --- Load All AI Models and Components ---
# Using st.cache_resource to load these heavy models only once
@st.cache_resource
def load_models():
    """Load all saved models and pipelines once to make the app fast."""
    with st.spinner("Warming up the AI engines... Please wait."):
        final_xgb_model = joblib.load('final_fraud_detection_model.joblib')
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
    return final_xgb_model, sentiment_pipeline, object_detector

final_model, sentiment_pipeline, object_detector = load_models()

# --- Helper Functions for Analysis ---
lemmatizer = WordNetLemmatizer()
stop_words_english = set(stopwords.words('english'))

def get_negative_sentiment_score(text):
    if not text: return 0
    result = sentiment_pipeline(text)[0]
    return result['score'] if result['label'] == 'NEGATIVE' else 0

category_to_labels = {
    'electronics': ['remote', 'cell phone', 'tv', 'laptop', 'screen', 'electronic device'],
    'fashion': ['shoe', 'tie', 'glove', 'hat', 'handbag', 't-shirt', 'clothing'],
    'books': ['book'],
    'home_goods': ['mug', 'cup', 'vase', 'plate', 'fork', 'couch']
}

def check_image_match(image, category):
    if image is None: return 0
    img = Image.open(image).convert("RGB")
    predictions = object_detector(img)
    allowed_labels = set(category_to_labels.get(category, []))
    for pred in predictions:
        if pred['label'] in allowed_labels:
            return 1 # Match found!
    return 0 # No match found

# --- UI Customization ---
st.markdown("""
    <style>
    .st-emotion-cache-1r4qj8v { border-radius: 10px; }
    .stButton>button { border-radius: 10px; }
    .st-emotion-cache-1y4p8pa { max-width: 100%; }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ Aegis AI")
    st.markdown("---")
    st.markdown("This application is a demonstration of a multi-modal AI system designed to proactively detect e-commerce refund fraud.")
    
    with st.expander("ℹ️ About the Creator"):
        st.info("**Name:** Somesh Shukla")
        st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/somesh-shukla-02883823b)")
        st.markdown("[GitHub Repository](https://github.com/shuklasomesh)")

# --- Main Page ---
st.title("Proactive Refund Fraud Detection")
st.markdown("This AI system acts as a detective, analyzing multiple clues to calculate a real-time fraud risk score.")

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.subheader("📝 Refund Request Details")
    
    with st.container(border=True):
        # Complaint Details
        product_category = st.selectbox("Product Category", ['electronics', 'fashion', 'books', 'home_goods'], help="Select the category of the product being refunded.")
        complaint_text = st.text_area("Complaint Message", "The screen on the phone I received is completely shattered and doesn't turn on. I want a refund immediately.", height=150)
        uploaded_image = st.file_uploader("Upload 'Proof of Damage' Photo", type=['jpg', 'png', 'jpeg'])
    
    # User History Simulation
    with st.expander("👤 Simulate User History (for demo purposes)"):
        account_age = st.slider("Account Age (days)", 1, 2000, 30)
        past_refunds = st.slider("Past Refund Count", 0, 10, 3)
        product_price = st.number_input("Product Price ($)", min_value=1.0, value=799.99)
    
    analyze_button = st.button("Analyze Fraud Risk", type="primary", use_container_width=True)

with col2:
    st.subheader("📈 Analysis Result")
    
    if analyze_button:
        if not complaint_text.strip():
            st.warning("Please enter a complaint message.")
        else:
            with st.spinner('Analyzing with Multi-Modal AI... This may take a moment.'):
                # 1. Analyze text
                neg_sentiment = get_negative_sentiment_score(complaint_text)
                
                # 2. Analyze image
                image_match = check_image_match(uploaded_image, product_category)
                
                # 3. Assemble features for the model
                features_df = pd.DataFrame({
                    'account_age_days': [account_age], 'total_orders': [5],
                    'past_refund_count': [past_refunds], 'product_price': [product_price],
                    'delivery_confirmation': [1], 'time_to_complain_hours': [2],
                    'image_match_score': [image_match], 'negative_sentiment_score': [neg_sentiment]
                })

                # 4. Make prediction
                prediction_proba = final_model.predict_proba(features_df)[0]
                risk_score = prediction_proba[1]
                
                # 5. Display dynamic gauge and result
                if risk_score > 0.7:
                    result_color = "#D32F2F" # Red
                    result_text = "HIGH RISK 🚨"
                elif risk_score > 0.3:
                    result_color = "#FBC02D" # Yellow
                    result_text = "MEDIUM RISK ⚠️"
                else:
                    result_color = "#388E3C" # Green
                    result_text = "LOW RISK ✅"
                
                # Improved Gauge using HTML/CSS
                st.markdown(f"""
                <div style="background-color: #262730; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 20px;">
                    <h3 style="color: white; margin-bottom: 10px; font-weight: 500;">Fraud Risk Score</h3>
                    <div style="position: relative; width: 150px; height: 150px; margin: auto;">
                        <svg width="150" height="150" viewBox="0 0 120 120">
                            <defs>
                                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stop-color="{result_color}" />
                                    <stop offset="100%" stop-color="{result_color}80" />
                                </linearGradient>
                            </defs>
                            <circle cx="60" cy="60" r="50" fill="none" stroke="#444" stroke-width="15" />
                            <circle cx="60" cy="60" r="50" fill="none" stroke="url(#gradient)" stroke-width="15" stroke-dasharray="{2 * 3.14159 * 50}" stroke-dashoffset="{2 * 3.14159 * 50 * (1 - risk_score)}" transform="rotate(-90 60 60)" style="transition: stroke-dashoffset 1s ease-in-out;" />
                            <text x="50%" y="50%" text-anchor="middle" dy=".3em" font-size="24" fill="{result_color}" font-weight="bold">{risk_score*100:.1f}%</text>
                        </svg>
                    </div>
                    <h3 style="color: white; margin-top: 10px; font-weight: 600;">{result_text}</h3>
                </div>
                """, unsafe_allow_html=True)

                with st.container(border=True):
                    st.write("🕵️‍♂️ **Evidence Breakdown:**")
                    
                    if neg_sentiment > 0.9:
                         st.write(f"🔴 **Text Analysis:** High negative sentiment detected (Score: {neg_sentiment:.2f})")
                    else:
                         st.write(f"🟢 **Text Analysis:** Normal sentiment detected (Score: {neg_sentiment:.2f})")
                    
                    if uploaded_image:
                        st.image(uploaded_image, caption="Uploaded Evidence", use_container_width=True)
                        if image_match == 0:
                            st.write(f"🔴 **Image Analysis:** Mismatch detected. Object in photo may not match '{product_category}'.")
                        else:
                            st.write(f"🟢 **Image Analysis:** Match confirmed. Object is consistent with product category.")
                    else:
                        st.write("⚪ **Image Analysis:** No image provided.")
    else:
        st.info("Please enter the details and click 'Analyze Fraud Risk' to see the AI's verdict.")

