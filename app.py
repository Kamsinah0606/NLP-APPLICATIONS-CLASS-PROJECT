import streamlit as st
import pandas as pd
import plotly.express as px
from sentiment_utils import analyze_reviews, evaluate_model

# --------------------------------------------------
# STREAMLIT PAGE CONFIG (McDonald's Theme)
# --------------------------------------------------
st.set_page_config(
    page_title="McDonald's Sentiment Analysis Dashboard",
    layout="wide"
)

# Custom CSS for McDonald's color theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF9E6;
    }
    h1, h2, h3 {
        color: #C8102E;
    }
    .stButton>button {
        background-color: #FFC72C;
        color: black;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# TITLE & INTRODUCTION
# --------------------------------------------------
st.title("🍔 McDonald's Sentiment Analysis Dashboard")
st.write(
    """
    This dashboard presents sentiment and emotion analysis of McDonald's
    restaurant reviews using Transformer-based Natural Language Processing models.
    """
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("McDonald_s_Reviews.csv", encoding="latin1")
    https://raw.githubusercontent.com/Kamsinah0606/NLP-APPLICATIONS-CLASS-PROJECT/refs/heads/main/processed_mcdonalds_reviews.csv

df = load_data()

# --------------------------------------------------
# DATA CONFIGURATION (OPTION A IMPLEMENTED)
# --------------------------------------------------
st.subheader("Dataset Configuration")

MAX_LIMIT = 5000                     # safety cap
dataset_size = len(df)
max_reviews = min(dataset_size, MAX_LIMIT)

sample_size = st.slider(
    "Select number of reviews for analysis",
    min_value=100,
    max_value=max_reviews,
    value=1000,
    step=100
)

st.caption(
    f"Dataset contains {dataset_size:,} reviews. "
    f"For performance efficiency, analysis is limited to a maximum of {max_reviews:,} reviews."
)

# --------------------------------------------------
# SENTIMENT ANALYSIS
# --------------------------------------------------
with st.spinner("Analyzing customer reviews..."):
    results_df = analyze_reviews(df, sample_size)

# --------------------------------------------------
# SENTIMENT DISTRIBUTION
# --------------------------------------------------
st.subheader("Sentiment Distribution")
fig_sent = px.pie(
    results_df,
    names="predicted_sentiment",
    title="Predicted Sentiment Distribution",
    color_discrete_sequence=["#C8102E", "#FFC72C", "#DA291C"]
)
st.plotly_chart(fig_sent, use_container_width=True)

# --------------------------------------------------
# EMOTION DISTRIBUTION
# --------------------------------------------------
st.subheader("Emotion Distribution")
emotion_counts = results_df["emotion"].value_counts().reset_index()
emotion_counts.columns = ["Emotion", "Count"]

fig_emo = px.bar(
    emotion_counts,
    x="Emotion",
    y="Count",
    title="Emotion Distribution in Customer Reviews",
    color_discrete_sequence=["#FFC72C"]
)
st.plotly_chart(fig_emo, use_container_width=True)

# --------------------------------------------------
# RATING VS SENTIMENT
# --------------------------------------------------
st.subheader("Rating vs Predicted Sentiment")
fig_rating = px.histogram(
    results_df,
    x="rating_num",
    color="predicted_sentiment",
    barmode="group",
    color_discrete_sequence=["#C8102E", "#FFC72C", "#DA291C"]
)
st.plotly_chart(fig_rating, use_container_width=True)

# --------------------------------------------------
# SAMPLE REVIEW PREDICTIONS
# --------------------------------------------------
st.subheader("Sample Review Predictions")
st.dataframe(
    results_df[
        ["review", "rating_num", "true_sentiment", "predicted_sentiment", "emotion"]
    ].head(10),
    use_container_width=True
)

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
st.subheader("Model Evaluation (Sample Data)")

report_df, cm = evaluate_model(
    results_df["true_sentiment"],
    results_df["predicted_sentiment"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Negative", "Neutral", "Positive"],
    columns=["Negative", "Neutral", "Positive"]
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Classification Report")
    st.dataframe(report_df.round(2), use_container_width=True)

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
        color_continuous_scale=["#FFF1B8", "#FFC72C", "#C8102E"]
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <center>
    <small>
    NLP Applications Class Project | Sentiment Analysis Dashboard using Streamlit
    </small>
    </center>
    """,
    unsafe_allow_html=True
)
