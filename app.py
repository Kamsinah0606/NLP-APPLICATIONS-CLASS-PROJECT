# app.py
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="McDonald's Reviews Sentiment & Emotion Dashboard",
    layout="wide"
)

st.title("🍔 McDonald's Reviews Sentiment & Emotion Analysis Dashboard")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("McDonald_s_Reviews.csv", encoding="latin1")
    df['rating_num'] = df['rating'].str.extract('(\d)').astype(int)
    
    def map_sentiment(r):
        if r <= 2:
            return "Negative"
        elif r == 3:
            return "Neutral"
        else:
            return "Positive"
    
    df['true_sentiment'] = df['rating_num'].apply(map_sentiment)
    df = df.dropna(subset=['review'])
    return df

df = load_data()

st.subheader("Sample of Reviews")
st.dataframe(df[['review', 'rating_num', 'true_sentiment']].head(10))

# --- Load NLP Models ---
@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )
    return sentiment_model, emotion_model

sentiment_model, emotion_model = load_models()

# --- Sidebar Controls ---
st.sidebar.header("Options")
sample_size = st.sidebar.slider(
    "Number of Reviews to Sample for Analysis", min_value=100, max_value=2000, value=1000, step=100
)

# --- Sample Data ---
sample_df = df.sample(sample_size, random_state=42).reset_index(drop=True)

# --- Run Predictions ---
@st.cache_data
def predict_sentiment_emotion(df):
    df['predicted_sentiment'] = df['review'].apply(
        lambda x: sentiment_model(x[:512])[0]['label']
    )
    df['predicted_sentiment'] = df['predicted_sentiment'].map({
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative"
    })
    df['emotion'] = df['review'].apply(
        lambda x: emotion_model(x[:512])[0]['label']
    )
    return df

with st.spinner("Running sentiment & emotion analysis..."):
    sample_df = predict_sentiment_emotion(sample_df)

st.success("Analysis Complete!")

# --- Metrics ---
st.subheader("Model Evaluation (Sampled Data)")

conf_matrix = confusion_matrix(sample_df['true_sentiment'], sample_df['predicted_sentiment'])
st.text("Classification Report:")
st.text(classification_report(sample_df['true_sentiment'], sample_df['predicted_sentiment']))

# --- Visualizations ---
st.subheader("Visualizations")

# 1️⃣ Sentiment Distribution
st.plotly_chart(
    px.pie(
        sample_df,
        names='predicted_sentiment',
        title="Sentiment Distribution of McDonald's Reviews"
    )
)

# 2️⃣ Emotion Distribution
emotion_counts_df = sample_df['emotion'].value_counts().reset_index()
emotion_counts_df.columns = ['Emotion', 'Count']

st.plotly_chart(
    px.bar(
        emotion_counts_df,
        x='Emotion',
        y='Count',
        labels={'Emotion':'Emotion', 'Count':'Count'},
        title="Emotion Distribution in Reviews."
    )
)

# 3️⃣ Rating vs Predicted Sentiment
st.plotly_chart(
    px.histogram(
        sample_df,
        x='rating_num',
        color='predicted_sentiment',
        title="Rating vs Predicted Sentiment",
        barmode='group'
    )
)

# 4️⃣ Confusion Matrix Heatmap
st.subheader("Confusion Matrix")

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=["True Negative", "True Neutral", "True Positive"],
    columns=["Predicted Negative", "Predicted Neutral", "Predicted Positive"]
)

fig_conf_matrix = px.imshow(
    conf_matrix_df,
    text_auto=True,
    color_continuous_scale="RdBu",
    labels=dict(
        x="Predicted Sentiment",
        y="True Sentiment",
        color="Count"
    ),
    title="Confusion Matrix of Sentiment Classification Results"
)

# 🔑 FIX SPACING HERE
fig_conf_matrix.update_layout(
    title=dict(
        y=0.95,                 # Move title DOWN (closer to plot)
        x=0.5,
        xanchor="center",
        yanchor="top"
    ),
    margin=dict(
        t=60,                   # Reduce top margin (default is ~100)
        b=40,
        l=40,
        r=40
    )
)

fig_conf_matrix.update_xaxes(
    side="top",
    title_standoff=10          # Reduce space between labels and plot
)

st.plotly_chart(fig_conf_matrix, use_container_width=True)


