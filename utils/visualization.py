import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ---------- 1️⃣ CV–JD Similarity Heatmap ----------
def plot_similarity_heatmap(match_df):
    pivot = match_df.pivot(index="CV_ID", columns="JD_ID", values="similarity_score")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".2f")
    plt.title("CV–JD Similarity Heatmap")
    plt.tight_layout()
    plt.savefig("data/similarity_heatmap.png")
    plt.close()


# ---------- 2️⃣ Sentiment Distribution ----------
def plot_sentiment_distribution(sentiment_df):
    counts = sentiment_df["sentiment"].value_counts()
    plt.figure(figsize=(5, 4))
    counts.plot(kind="pie", autopct="%1.1f%%", startangle=90, colors=["lightgreen", "lightcoral", "lightblue"])
    plt.title("Feedback Sentiment Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("data/sentiment_pie.png")
    plt.close()


# ---------- 3️⃣ RL Reward Trend (Mock Visualization) ----------
def plot_rl_rewards(episodes=50):
    rewards = np.random.uniform(-1, 1, episodes).cumsum()
    plt.figure(figsize=(6, 4))
    plt.plot(range(episodes), rewards, marker="o")
    plt.title("RL Agent Reward Tracking")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/rl_rewards.png")
    plt.close()


# ---------- 4️⃣ Combined Overview ----------
def generate_all_plots(match_df, sentiment_df):
    print("📊 Generating visualizations...")
    plot_similarity_heatmap(match_df)
    plot_sentiment_distribution(sentiment_df)
    plot_rl_rewards()
    print("✅ Charts saved in /data as PNG files.")
