import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from utils.rl_agent import RLAgent # Import the modified agent

# --- Configuration ---
CVS_PATH = 'data/cvs.csv'
JDS_PATH = 'data/jds.csv'
FEEDBACKS_PATH = 'data/feedbacks.csv'

st.set_page_config(layout="wide", page_title="HR RL Agent Dashboard")

# --- Function to Initialize/Run Agent ---
@st.cache_resource
def get_trained_agent():
    """Initializes and runs the agent through historical feedback for visualization."""
    try:
        agent = RLAgent(CVS_PATH, JDS_PATH)
        feedbacks_df = pd.read_csv(FEEDBACKS_PATH)
        
        # Simulate the training loop to populate history
        # NOTE: This runs the update_reward method for every entry in feedbacks.csv
        for idx, feedback_entry in feedbacks_df.iterrows():
            # Add feedback_id for logging/merging if your feedbacks.csv doesn't have it
            if 'feedback_id' not in feedback_entry:
                feedback_entry['feedback_id'] = idx + 1
                
            agent.update_reward(feedback_entry)
            
        return agent, feedbacks_df
    except Exception as e:
        st.error(f"Error loading or training agent: {e}. Check file paths, column names, or data integrity.")
        st.stop()
        return None, None

AGENT, FEEDBACK_DF = get_trained_agent()

st.title("🤖 HR RL Agent Transparency Dashboard")

if AGENT and AGENT.history:
    history_df = pd.DataFrame(AGENT.history)
    
    # --- VISUAL INDICATORS ---
    st.header("Agent Status and Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    num_feedbacks = len(FEEDBACK_DF)
    
    with col1:
        st.metric(
            label="Total Feedbacks Processed", 
            value=num_feedbacks
        )
        st.success(f"Agent has learned from **{num_feedbacks}** feedbacks.")

    with col2:
        st.metric(
            label="Cumulative Reward (Total)", 
            value=f"{AGENT.total_reward_over_time:.2f}",
            delta=f"{AGENT.history[-1]['reward']:.2f}"
        )
    
    with col3:
        # Calculate most preferred action for the most frequent state (S)
        most_frequent_state = history_df['s_tuple'].mode().iloc[0]
        q_values = AGENT.q_table[most_frequent_state]
        best_action_index = np.argmax(q_values)
        best_action = ['accept', 'reject', 'reconsider'][best_action_index]
        
        st.markdown(f"**Most Preferred Action for State {most_frequent_state}**")
        st.markdown(f"## {best_action.upper()}")

    # --- ROW 1: Charts ---
    st.header("RL Performance Trends")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("RL Cumulative Reward History")
        # Use Plotly for interactive chart
        fig_reward = px.line(
            history_df.reset_index(), # Use index as X-axis (time)
            x='index',
            y='cumulative_reward', 
            title='Agent Learning Progress (Cumulative Reward)',
            labels={'cumulative_reward': 'Cumulative Reward', 'index': 'Feedback Entry'}
        )
        st.plotly_chart(fig_reward, use_container_width=True)

    with chart_col2:
        st.subheader("Feedback Sentiment Distribution")
        # Extract sentiment level (the second element of the S_tuple)
        sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment_counts = history_df['s_tuple'].apply(lambda x: sentiment_mapping[x[1]]).value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        fig_sentiment = px.bar(
            sentiment_counts, 
            x='Sentiment', 
            y='Count', 
            title='Distribution of Feedback Sentiment',
            color='Sentiment',
            color_discrete_map={'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    # --- ROW 2: Tables & Logs ---
    st.header("Policy & Log Data")
    
    # Policy Visualization (Q-Table)
    st.subheader(f"Q-Values for Most Frequent State {most_frequent_state}")
    
    # Convert Q-table slice to DataFrame for display
    q_df = pd.DataFrame(
        q_values.reshape(1, -1),
        columns=['Q(accept)', 'Q(reject)', 'Q(reconsider)'],
        index=[f"State {most_frequent_state}"]
    ).T.rename(columns={0: "Q-Value"})
    
    st.dataframe(q_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)


    # Feedback logs (expandable list)
    st.subheader("Historical Feedback Logs (Latest 10)")
    
    # Merge history and original feedback for a rich log display
    # Ensure all columns needed are available for merging
    log_df = history_df.merge(FEEDBACK_DF[['candidate_id', 'jd_id', 'comment', 'feedback_score']], 
                                 on=['candidate_id', 'jd_id', 'feedback_score', 'comment'], how='left')
    
    for idx, row in log_df.sort_values(by=log_df.columns[0], ascending=False).head(10).iterrows():
        # Use color-coded scores for ranking (HR Feedback Score)
        color = 'green' if row['feedback_score'] >= 4 else ('orange' if row['feedback_score'] > 2 else 'red')
        
        st.markdown(f"**Candidate {row['candidate_id']} / Job {row['jd_id']}** - Score: :{color}[{row['feedback_score']}]")
        with st.expander(f"Details (Action: {['accept','reject','reconsider'][row['action_taken']]})"):
            st.json({
                "Comment": row['comment'],
                "RL Reward Received": row['reward'],
                "State Tuple (S)": row['s_tuple'],
                "Cumulative Reward": f"{row['cumulative_reward']:.2f}"
            })

else:
    st.error("Could not load agent or history data. Please check data files and RLAgent implementation.")