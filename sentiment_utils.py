from transformers import pipeline
import pandas as pd

# Load models once (IMPORTANT for Streamlit performance)
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def map_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

def analyze_reviews(df, sample_size=1000):
    df = df.dropna(subset=["review"])
    df["rating_num"] = df["rating"].str.extract(r"(\d)").astype(int)
    df["true_sentiment"] = df["rating_num"].apply(map_sentiment)

    df = df.sample(sample_size, random_state=42)

    df["predicted_sentiment"] = df["review"].apply(
        lambda x: sentiment_model(x[:512])[0]["label"]
    )

    df["predicted_sentiment"] = df["predicted_sentiment"].map({
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative"
    })

    df["emotion"] = df["review"].apply(
        lambda x: emotion_model(x[:512])[0]["label"]
    )

    return df

