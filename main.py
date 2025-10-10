import os
import pandas as pd
from utils.matching_engine import compute_similarity
from utils.sentiment_analyzer import analyze_sentiment
from utils.rl_agent import train_rl_agent, ACTIONS
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
# Updated predict function
# -------------------
def predict(cv_texts, jd_texts, feedback_df=None):
    """
    cv_texts: list of CV texts
    jd_texts: list of JD texts
    feedback_df: optional DataFrame of feedbacks
    Returns: final decision DataFrame
    """
    # Step 1: Compute CV ↔ JD similarity
    match_df = compute_similarity(cv_texts, jd_texts)

    # Step 2: Sentiment Analysis (optional)
    if feedback_df is not None and not feedback_df.empty:
        sentiment_df = analyze_sentiment(feedback_df)
    else:
        sentiment_df = pd.DataFrame()

    # Step 3: Generate automatic rewards for RL agent
    states = [f"cv_{i}" for i in range(len(cv_texts))]
    rewards = {}
    for i, cv in enumerate(cv_texts):
        rewards[states[i]] = {}
        for action in ACTIONS:
            if action == "Hire":
                # Simple reward: number of common words with JD i
                rewards[states[i]][action] = len(set(cv.split()) & set(jd_texts[i].split()))
            elif action == "Reject":
                rewards[states[i]][action] = 0
            else:  # Reassign
                rewards[states[i]][action] = 0.5

    # Step 4: Train RL Agent
    q_table = train_rl_agent(states, rewards)

    # Step 5: Make final decision
    final_df = make_decision(match_df, sentiment_df, q_table)

    return final_df

# -------------------
# Main workflow
# -------------------
def main():
    # Load CVs and JDs
    cv_texts = load_text_files(CV_DIR, "cv", 2)
    jd_texts = load_text_files(JD_DIR, "jd", 2)

    # Load feedbacks if available
    if os.path.exists(FEEDBACK_FILE):
        feedback_df = pd.read_csv(FEEDBACK_FILE)
    else:
        print(f"⚠️ Feedback file not found: {FEEDBACK_FILE}")
        feedback_df = pd.DataFrame()

    # Get final AI decisions
    final_df = predict(cv_texts, jd_texts, feedback_df)

    # Save outputs
    final_csv = os.path.join(OUTPUT_DIR, "final_results.csv")
    final_df.to_csv(final_csv, index=False)
    print(f"✅ Final decisions saved to {final_csv}")

    # Optionally generate visualizations
    try:
        match_df = compute_similarity(cv_texts, jd_texts)
        if not feedback_df.empty:
            sentiment_df = analyze_sentiment(feedback_df)
        else:
            sentiment_df = pd.DataFrame()
        generate_all_plots(match_df, sentiment_df)
        print("📊 Visualizations generated successfully")
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")

if __name__ == "__main__":
    main()
