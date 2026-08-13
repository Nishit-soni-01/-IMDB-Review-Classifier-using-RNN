
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        /* Import a refined serif + sans combo */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Overall background */
        .stApp {
            background-color: #faf9f6;
        }

        /* Hide default Streamlit chrome for a cleaner feel */
        #MainMenu, footer, header {visibility: hidden;}

        /* Main container width */
        .block-container {
            max-width: 720px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        /* Title */
        .app-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.4rem;
            font-weight: 700;
            color: #1a1a1a;
            text-align: center;
            margin-bottom: 0.25rem;
            letter-spacing: -0.5px;
        }

        .app-subtitle {
            text-align: center;
            color: #6b6b6b;
            font-size: 1rem;
            font-weight: 400;
            margin-bottom: 2.2rem;
        }

        /* Divider */
        .thin-divider {
            border: none;
            border-top: 1px solid #e2e0da;
            margin: 1.8rem 0;
        }

        /* Text area label */
        .stTextArea label {
            font-weight: 600 !important;
            color: #2a2a2a !important;
            font-size: 0.95rem !important;
        }

        /* Text area box */
        .stTextArea textarea {
            border: 1px solid #d8d5cd !important;
            border-radius: 10px !important;
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            caret-color: #1a1a1a !important;
            font-size: 0.98rem !important;
            padding: 14px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
            -webkit-text-fill-color: #1a1a1a !important;
        }
        .stTextArea textarea::placeholder {
            color: #9a9a9a !important;
            opacity: 1 !important;
        }
        .stTextArea textarea:focus {
            border-color: #a89a6c !important;
            box-shadow: 0 0 0 1px #a89a6c !important;
        }

        /* Button */
        .stButton > button {
            background-color: #1a1a1a;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.3px;
            transition: all 0.2s ease-in-out;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #a89a6c;
            color: #ffffff;
            transform: translateY(-1px);
        }

        /* Result card */
        .result-card {
            margin-top: 1.8rem;
            padding: 1.6rem 1.8rem;
            border-radius: 14px;
            border: 1px solid #e2e0da;
            background-color: #ffffff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }

        .result-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #8a8a8a;
            margin-bottom: 0.3rem;
        }

        .sentiment-positive {
            font-family: 'Playfair Display', serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: #2e7d32;
        }

        .sentiment-negative {
            font-family: 'Playfair Display', serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: #b23b3b;
        }

        .score-text {
            margin-top: 0.6rem;
            font-size: 0.95rem;
            color: #4a4a4a;
        }

        /* Footer note */
        .footer-note {
            text-align: center;
            color: #a3a3a3;
            font-size: 0.8rem;
            margin-top: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Step 2: Load Model and Word Index (cached for performance)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return load_model('simple_rnn_imdb.h5')

@st.cache_resource(show_spinner=False)
def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

with st.spinner("Loading model..."):
    model = load_sentiment_model()
    word_index, reverse_word_index = load_word_index()

# ---------------------------------------------------------
# Step 3: Helper Functions
# ---------------------------------------------------------
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])



def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2 - 3) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ---------------------------------------------------------
# Step 4: UI Layout
# ---------------------------------------------------------
st.markdown('<div class="app-title">🎬 IMDB Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">A simple RNN model that classifies movie reviews as positive or negative</div>',
    unsafe_allow_html=True,
)

user_input = st.text_area(
    "Movie Review",
    placeholder="Type or paste a movie review here...",
    height=160,
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classify_clicked = st.button("Classify Review")

if classify_clicked:
    if not user_input.strip():
        st.warning("Please enter a movie review before classifying.")
    else:
        with st.spinner("Analyzing sentiment..."):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            score = float(prediction[0][0])
            sentiment = "Positive" if score > 0.5 else "Negative"
            sentiment_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Sentiment</div>
                <div class="{sentiment_class}">{sentiment}</div>
                <div class="score-text">Confidence Score: <strong>{score:.4f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="footer-note">Enter a review above and click "Classify Review" to see the result.</div>',
        unsafe_allow_html=True,
    )
