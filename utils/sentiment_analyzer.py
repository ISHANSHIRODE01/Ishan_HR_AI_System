from textblob import TextBlob
import pandas as pd

def analyze_sentiment(feedback_df):
    sentiments = []
    for _, row in feedback_df.iterrows():
        polarity = TextBlob(row['feedback']).sentiment.polarity
        label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        sentiments.append({
            "candidate_id": row['candidate_id'],
            "sentiment": label,
            "polarity": round(polarity, 3)
        })
    return pd.DataFrame(sentiments)
