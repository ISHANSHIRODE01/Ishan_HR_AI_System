import numpy as np
import random

ACTIONS = ["Hire", "Reject", "Reassign"]

def train_rl_agent(states, rewards, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.2):
    """
    Q-Learning RL Agent for HR decisions.
    states: list of states
    rewards: dict of dicts, e.g., rewards[state][action] = reward_value
    """
    # Initialize Q-table
    q_table = {s: np.zeros(len(ACTIONS)) for s in states}

    for _ in range(episodes):
        state = random.choice(states)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randint(0, len(ACTIONS)-1)  # explore
        else:
            action_idx = np.argmax(q_table[state])          # exploit

        action = ACTIONS[action_idx]
        reward = rewards[state][action]  # reward depends on state-action pair

        # Q-learning update
        old_value = q_table[state][action_idx]
        next_max = np.max(q_table[state])  # simplified next state as same (no transition)
        q_table[state][action_idx] = old_value + alpha * (reward + gamma * next_max - old_value)

    return q_table
