import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob # Simple sentiment analysis

# --- 1. CONFIGURATION ---
ALPHA = 0.1     # Learning rate
GAMMA = 0.6     # Discount factor
EPSILON = 0.1   # Exploration rate (used by choose_action to simulate training decision)
NUM_ACTIONS = 3 # 0: accept, 1: reject, 2: reconsider

# Discrete levels for state features 
MATCH_THRESHOLDS = [0.2, 0.5] # For Match Score
SENTIMENT_THRESHOLDS = [-0.1, 0.1] # For Sentiment Score

# --- 2. HELPER FUNCTIONS (for creating the State) ---

def calculate_match_score(cv_text, jd_text, vectorizer):
    """Simple Cosine Similarity for a Match Score"""
    # Note: TFIDF vectorizer must be fitted on every call due to Streamlit caching design,
    # but for simplicity, we assume consistent content structure.
    corpus = [cv_text, jd_text]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    # Cosine similarity calculation
    return np.dot(tfidf_matrix[0].toarray(), tfidf_matrix[1].toarray().T)[0][0]

def discretize_state(match_score, sentiment_score, prev_reward, reconsideration_count):
    """Converts continuous features to discrete levels for the Q-Table index."""
    
    # 1. Match Score (0: Low, 1: Medium, 2: High)
    if match_score < MATCH_THRESHOLDS[0]:
        match_level = 0
    elif match_score < MATCH_THRESHOLDS[1]:
        match_level = 1
    else:
        match_level = 2
        
    # 2. Sentiment Score (0: Negative, 1: Neutral, 2: Positive)
    if sentiment_score < SENTIMENT_THRESHOLDS[0]:
        sentiment_level = 0
    elif sentiment_score < SENTIMENT_THRESHOLDS[1]:
        sentiment_level = 1
    else:
        sentiment_level = 2
        
    # 3. Previous Reward (0: Bad (-1), 1: Neutral (0), 2: Good (+1))
    prev_reward_level = int(prev_reward + 1)

    # 4. Action History (0: Low (<2), 1: High (>=2))
    history_level = 1 if reconsideration_count >= 2 else 0
    
    return (match_level, sentiment_level, prev_reward_level, history_level)

# --- 3. RL Agent Class ---

class RLAgent:
    def __init__(self, cvs_path, jds_path):
        # Load Data
        self.cvs = pd.read_csv(cvs_path)
        self.jds = pd.read_csv(jds_path)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        
        # Q-Table dimensions: 3 (Match) x 3 (Sentiment) x 3 (Reward) x 2 (History) x 3 (Action)
        state_space_size = (3, 3, 3, 2)
        self.q_table = np.zeros(state_space_size + (NUM_ACTIONS,))
        
        # Internal state tracking for the test loop: stores reward/reconsideration count for each pair
        self.pair_tracking = {}
        
        self.history = []
        self.total_reward_over_time = 0
        # -----------------------------------------------------------

    def get_state(self, candidate_id, jd_id, comment):
        """Calculates and discretizes the current State for a pair."""
        
        # 1. Get Text & Compute Match Score (using 'skills' and 'description' as confirmed)
        cv_text = self.cvs[self.cvs['candidate_id'] == candidate_id]['skills'].iloc[0]
        jd_text = self.jds[self.jds['jd_id'] == jd_id]['description'].iloc[0]
        match_score = calculate_match_score(cv_text, jd_text, self.vectorizer)

        # 2. Get Sentiment Score from the latest feedback comment
        sentiment_score = TextBlob(comment).sentiment.polarity

        # 3. Get Previous Reward & Reconsideration Count
        pair_key = (candidate_id, jd_id)
        if pair_key not in self.pair_tracking:
            self.pair_tracking[pair_key] = {'prev_reward': 0, 'reconsider_count': 0}
            
        prev_reward = self.pair_tracking[pair_key]['prev_reward']
        reconsider_count = self.pair_tracking[pair_key]['reconsider_count']

        # 4. Discretize
        state_tuple = discretize_state(match_score, sentiment_score, prev_reward, reconsider_count)
        return state_tuple

    def choose_action(self, state_tuple):
        """Epsilon-greedy policy for action selection."""
        # When called from the Flask app, this simulates the policy's prediction.
        state_index = state_tuple
        if np.random.uniform(0, 1) < EPSILON:
             # Exploration (random action) - low chance
             return np.random.randint(NUM_ACTIONS)
        else:
             # Exploitation (best action from Q-table)
             return np.argmax(self.q_table[state_index])

    def calculate_reward(self, chosen_action, feedback_score):
        """
        Calculates the Reward based on the action taken and the final HR outcome.
        """
        # Target HR outcome based on feedback score thresholds
        if feedback_score > 4:
            hr_outcome = 'GOOD'
        elif feedback_score < 2:
            hr_outcome = 'BAD'
        else:
            hr_outcome = 'NEUTRAL'
            
        # Reward Logic
        if chosen_action == 0: # 'accept'
            return 1 if hr_outcome == 'GOOD' else -1
        elif chosen_action == 1: # 'reject'
            return 1 if hr_outcome == 'BAD' else -1
        elif chosen_action == 2: # 'reconsider'
            return 0 # Neutral reward for a delay action
        
        return 0

    def update_reward(self, feedback_entry):
        """
        The core function: Updates the Q-table based on new HR feedback.
        """
        candidate_id = feedback_entry['candidate_id']
        jd_id = feedback_entry['jd_id']
        feedback_score = feedback_entry['feedback_score']
        comment = feedback_entry['comment']
        
        pair_key = (candidate_id, jd_id)
        
        # 1. Get the current State (S) and its index (S is based on data *before* this feedback)
        s_tuple = self.get_state(candidate_id, jd_id, comment)
        s_index = s_tuple

        # 2. Determine the Action (A) the RL agent would have taken
        action_taken = self.choose_action(s_tuple)
        
        # 3. Calculate Reward (R)
        reward = self.calculate_reward(action_taken, feedback_score)

        # 4. Update tracking for S' calculation and persistence
        self.pair_tracking[pair_key]['prev_reward'] = reward
        if action_taken == 2: # 'reconsider'
            self.pair_tracking[pair_key]['reconsider_count'] += 1
            
        # S' is the state that results from the action/feedback
        s_prime_tuple = self.get_state(candidate_id, jd_id, comment)
        s_prime_index = s_prime_tuple
        
        # 5. Q-Learning Update Equation: Q(S, A) <- Q(S, A) + ALPHA * [R + GAMMA * max(Q(S', a)) - Q(S, A)]
        old_q = self.q_table[s_index][action_taken]
        next_max_q = np.max(self.q_table[s_prime_index])
        
        new_q = old_q + ALPHA * (reward + GAMMA * next_max_q - old_q)
        self.q_table[s_index][action_taken] = new_q
        
        self.total_reward_over_time += reward
        self.history.append({
            'candidate_id': candidate_id,
            'jd_id': jd_id,
            's_tuple': s_tuple,
            'action_taken': action_taken,
            'reward': reward,
            'cumulative_reward': self.total_reward_over_time,
            'feedback_score': feedback_score,
            'comment': comment
        })
        # ---------------------------------------

        print(f"\n--- RL Update for Pair ({candidate_id}, {jd_id}) ---")
        print(f"S: {s_tuple}, A: {action_taken} ('{['accept','reject','reconsider'][action_taken]}'), R: {reward}")
        print(f"Q-Table update: Q{s_index + (action_taken,)} from {old_q:.4f} to {new_q:.4f}")
        print("------------------------------------------------------")
        
        return self.q_table
