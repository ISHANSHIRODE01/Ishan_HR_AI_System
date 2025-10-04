import pandas as pd

def make_decision(match_df, sentiment_df, rl_q_table):
    merged = match_df.merge(sentiment_df, left_on="CV_ID", right_on="candidate_id", how="left")

    final = []
    for _, row in merged.iterrows():
        score = (row['similarity_score'] * 0.7) + (row['polarity'] * 0.3)
        decision = "Hire" if score > 0.3 else "Reject"
        final.append({
            "CV_ID": row["CV_ID"],
            "JD_ID": row["JD_ID"],
            "score": round(score, 3),
            "sentiment": row["sentiment"],
            "decision": decision
        })
    return pd.DataFrame(final)
