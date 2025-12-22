import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

st.title("🍔 Sentiment Analysis Dashboard for Restaurant Reviews")
st.write("McDonald's Reviews Analysis using Transformer-based Models")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("McDonald_s_Reviews.csv", encoding="latin1")
    df = df.dropna(subset=["review"])
    df["rating_num"] = df["rating"].str.extract("(\d)").astype(int)
    return df

df = load_data()

# -----------------------------
# Sentiment Mapping
# -----------------------------
def map_sentiment(r):
    if r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    else:
        return "Positive"

df["true_sentiment"] = df["rating_num"].apply(map_sentiment)

# -----------------------------
# Load Models (Cached)
# -----------------------------
@st.cache_resource
def load_models():
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    emotion = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )
    return sentiment, emotion

sentiment_model, emotion_model = load_models()

# -----------------------------
# Sample Selection
# -----------------------------
st.sidebar.header("Dashboard Controls")
sample_size = st.sidebar.slider("Select number of reviews", 100, 2000, 500)

sample_df = df.sample(sample_size, random_state=42)

# -----------------------------
# Predictions
# -----------------------------
sample_df["predicted_sentiment"] = sample_df["review"].apply(
    lambda x: sentiment_model(x[:512])[0]["label"]
)

sample_df["predicted_sentiment"] = sample_df["predicted_sentiment"].map({
    "POSITIVE": "Positive",
    "NEGATIVE": "Negative"
})

sample_df["emotion"] = sample_df["review"].apply(
    lambda x: emotion_model(x[:512])[0]["label"]
)

# -----------------------------
# Visualizations
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    fig_sentiment = px.pie(
        sample_df,
        names="predicted_sentiment",
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

with col2:
    fig_emotion = px.bar(
        sample_df["emotion"].value_counts().reset_index(),
        x="index",
        y="emotion",
        labels={"index": "Emotion", "emotion": "Count"},
        title="Emotion Distribution"
    )
    st.plotly_chart(fig_emotion, use_container_width=True)

fig_rating = px.histogram(
    sample_df,
    x="rating_num",
    color="predicted_sentiment",
    barmode="group",
    title="Rating vs Predicted Sentiment"
)
st.plotly_chart(fig_rating, use_container_width=True)

# -----------------------------
# Sample Reviews Table
# -----------------------------
st.subheader("Sample Review Predictions")
st.dataframe(
    sample_df[["review", "rating_num", "predicted_sentiment", "emotion"]].head(20)
)

