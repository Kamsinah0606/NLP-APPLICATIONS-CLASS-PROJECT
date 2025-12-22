import streamlit as st
import pandas as pd
import plotly.express as px
from sentiment_utils import analyze_reviews

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("🍔 Restaurant Review Sentiment Analysis Dashboard")
st.write("McDonald's Reviews Analysis using Transformer-based NLP Models")

@st.cache_data
def load_data():
    return pd.read_csv("McDonald_s_Reviews.csv", encoding="latin1")

df = load_data()

sample_size = st.slider("Select number of reviews to analyze", 100, 2000, 1000)

with st.spinner("Analyzing reviews..."):
    results_df = analyze_reviews(df, sample_size)

# --- Sentiment Distribution ---
st.subheader("Sentiment Distribution")
fig_sent = px.pie(
    results_df,
    names="predicted_sentiment",
    title="Predicted Sentiment Distribution"
)
st.plotly_chart(fig_sent, use_container_width=True)

# --- Emotion Distribution ---
st.subheader("Emotion Distribution")
fig_emo = px.bar(
    results_df["emotion"].value_counts().reset_index(),
    x="index",
    y="emotion",
    labels={"index": "Emotion", "emotion": "Count"},
    title="Emotion Distribution"
)
st.plotly_chart(fig_emo, use_container_width=True)

# --- Rating vs Sentiment ---
st.subheader("Rating vs Predicted Sentiment")
fig_rating = px.histogram(
    results_df,
    x="rating_num",
    color="predicted_sentiment",
    barmode="group"
)
st.plotly_chart(fig_rating, use_container_width=True)

# --- Sample Reviews ---
st.subheader("Sample Review Predictions")
st.dataframe(
    results_df[["review", "rating_num", "predicted_sentiment", "emotion"]].head(10)
)
