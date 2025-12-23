import streamlit as st
import pandas as pd
import plotly.express as px
from sentiment_utils import analyze_reviews, evaluate_model

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

# -------------------------------
# Title and Description
# -------------------------------
st.title("🍔 MacDonald Review Sentiment Analysis Dashboard")
st.write(
    "This dashboard presents sentiment and emotion analysis results "
    "for McDonald's restaurant reviews using Transformer-based NLP models."
)

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("McDonald_s_Reviews.csv", encoding="latin1")

df = load_data()

# -------------------------------
# User Controls
# -------------------------------
st.sidebar.header("Analysis Settings")
sample_size = st.sidebar.slider(
    "Select number of reviews for analysis",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100
)

# -------------------------------
# Run Analysis
# -------------------------------
with st.spinner("Analyzing customer reviews..."):
    results_df = analyze_reviews(df, sample_size)

# -------------------------------
# Sentiment Distribution
# -------------------------------
st.subheader("Sentiment Distribution")
fig_sentiment = px.pie(
    results_df,
    names="predicted_sentiment",
    title="Predicted Sentiment Distribution"
)
st.plotly_chart(fig_sentiment, use_container_width=True)

# -------------------------------
# Emotion Distribution
# -------------------------------
st.subheader("Emotion Distribution")
emotion_counts = results_df["emotion"].value_counts().reset_index()
emotion_counts.columns = ["Emotion", "Count"]

fig_emotion = px.bar(
    emotion_counts,
    x="Emotion",
    y="Count",
    title="Emotion Distribution"
)
st.plotly_chart(fig_emotion, use_container_width=True)

# -------------------------------
# Rating vs Predicted Sentiment
# -------------------------------
st.subheader("Rating vs Predicted Sentiment")
fig_rating = px.histogram(
    results_df,
    x="rating_num",
    color="predicted_sentiment",
    barmode="group",
    title="Rating vs Predicted Sentiment"
)
st.plotly_chart(fig_rating, use_container_width=True)

# -------------------------------
# Sample Review Predictions
# -------------------------------
st.subheader("Sample Review Predictions")
st.dataframe(
    results_df[
        ["review", "rating_num", "true_sentiment", "predicted_sentiment", "emotion"]
    ].head(10),
    use_container_width=True
)

# -------------------------------
# Model Evaluation Section
# -------------------------------
st.subheader("Model Evaluation (Sample Data)")

# Get evaluation results
report_df, cm = evaluate_model(
    results_df["true_sentiment"],
    results_df["predicted_sentiment"]
)

# Create confusion matrix dataframe
cm_df = pd.DataFrame(
    cm,
    index=["Negative", "Neutral", "Positive"],
    columns=["Negative", "Neutral", "Positive"]
)

# Layout: Side-by-side
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Classification Report")
    st.dataframe(
        report_df.round(2),
        use_container_width=True
    )

with col2:
    st.markdown("### Confusion Matrix of Sentiment Classification Results")
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        labels=dict(
            x="Predicted Label",
            y="True Label",
            color="Count"
        ),
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "📌 **Note:** The evaluation results are generated from a sampled subset "
    "of the dataset to ensure computational efficiency while maintaining "
    "representative sentiment distribution."
)
