import os
import pandas as pd
from utils.matching_engine import compute_similarity
from utils.sentiment_analyzer import analyze_sentiment
from utils.rl_agent import train_rl_agent
from utils.decision_engine import make_decision
from utils.visualization import generate_all_plots

# Define paths
CV_DIR = "data/sample_cvs"
JD_DIR = "data/sample_jds"
FEEDBACK_FILE = "data/feedbacks.csv"
OUTPUT_DIR = "data"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_text_files(directory, prefix, count):
    texts = []
    for i in range(1, count + 1):
        path = os.path.join(directory, f"{prefix}{i}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        else:
            print(f"⚠️ File not found: {path}")
    return texts

# -------------------
# New: predict function for Flask/Streamlit
# -------------------
def predict(cv_texts, jd_texts, feedback_df=None):
    """
    cv_texts: list of CV texts (strings)
    jd_texts: list of JD texts (strings)
    feedback_df: optional DataFrame of feedbacks
    Returns: final decision DataFrame
    """
    # Step 1: Match CVs ↔ JDs
    match_df = compute_similarity(cv_texts, jd_texts)

    # Step 2: Sentiment Analysis (optional)
    if feedback_df is not None and not feedback_df.empty:
        sentiment_df = analyze_sentiment(feedback_df)
    else:
        sentiment_df = pd.DataFrame()

    # Step 3: Train RL Agent
    states = ["High", "Medium", "Low"]
    rewards = {"High": 1, "Medium": 0, "Low": -1}
    q_table = train_rl_agent(states, rewards)

    # Step 4: Make Final Decision
    final_df = make_decision(match_df, sentiment_df, q_table)

    return final_df

# -------------------
# Existing main workflow
# -------------------
def main():
    # Step 0: Load CVs and JDs
    cv_texts = load_text_files(CV_DIR, "cv", 2)
    jd_texts = load_text_files(JD_DIR, "jd", 2)
    
    # Step 1: Match CVs ↔ JDs
    match_df = compute_similarity(cv_texts, jd_texts)
    match_csv = os.path.join(OUTPUT_DIR, "match_scores.csv")
    match_df.to_csv(match_csv, index=False)
    print(f"📄 CV-JD similarity scores saved to {match_csv}")

    # Step 2: Sentiment Analysis
    if os.path.exists(FEEDBACK_FILE):
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        sentiment_df = analyze_sentiment(feedback_df)
        sentiment_csv = os.path.join(OUTPUT_DIR, "sentiment_results.csv")
        sentiment_df.to_csv(sentiment_csv, index=False)
        print(f"📄 Sentiment analysis results saved to {sentiment_csv}")
    else:
        print(f"⚠️ Feedback file not found: {FEEDBACK_FILE}")
        sentiment_df = pd.DataFrame()

    # Step 3: Train RL Agent
    states = ["High", "Medium", "Low"]
    rewards = {"High": 1, "Medium": 0, "Low": -1}
    q_table = train_rl_agent(states, rewards)
    print("🤖 RL agent training completed")

    # Step 4: Make Final Decision
    final_df = make_decision(match_df, sentiment_df, q_table)
    final_csv = os.path.join(OUTPUT_DIR, "final_results.csv")
    final_df.to_csv(final_csv, index=False)
    print(f"✅ Final decisions saved to {final_csv}")

    # Step 5: Generate Visualizations
    try:
        generate_all_plots(match_df, sentiment_df)
        print("📊 Visualizations generated successfully")
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")

if __name__ == "__main__":
    main()
