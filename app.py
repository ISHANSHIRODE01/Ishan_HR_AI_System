from flask import Flask, request, jsonify
import pandas as pd
import os

# --- NEW IMPORTS FOR GEMINI ---
from google import genai
from google.genai.errors import APIError

from utils.rl_agent import RLAgent  # Assuming RLAgent is importable from utils

app = Flask(__name__)

# --- Initialization ---
CVS_PATH = 'data/cvs.csv'
JDS_PATH = 'data/jds.csv'
FEEDBACK_LOG_PATH = 'data/feedback_log.csv'  # For optional logging

# Instantiate the RL Agent
try:
    AGENT = RLAgent(CVS_PATH, JDS_PATH)
    print("RL Agent initialized successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Ensure data/cvs.csv and data/jds.csv exist.")
    AGENT = None  # Fallback

# Initialize Gemini Client
try:
    GEMINI_CLIENT = genai.Client()  # Automatically uses GEMINI_API_KEY or GOOGLE_API_KEY
    print("Gemini Client initialized successfully.")
except Exception as e:
    print(f"Gemini Client Error: {e}. Check GEMINI_API_KEY environment variable.")
    GEMINI_CLIENT = None


# --- Home Route (For Browser Check) ---
@app.route('/')
def home():
    return "✅ HR RL Agent Backend is Running! Use /update_feedback (POST) to send feedback."


# --- Helper Function: Summarize Feedback ---
def summarize_feedback_with_gemini(candidate_id, jd_id, comment, feedback_score):
    if not GEMINI_CLIENT:
        return "Gemini client not initialized."

    prompt = f"""
    HR Feedback Summary Task:
    Candidate ID: {candidate_id}
    Job ID: {jd_id}
    Raw Comment: "{comment}"
    Score (1=Bad, 5=Good): {feedback_score}

    Analyze the Raw Comment, determine the core reason for the score, and provide
    a single, concise sentence summary (max 15 words) suitable for an HR Slack channel.
    """

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=150,
            )
        )

        if response.text:
            return response.text.strip()

        if response.candidates and response.candidates[0].finish_reason:
            reason = response.candidates[0].finish_reason.name
            if reason == 'SAFETY':
                return "Gemini Summary failed: Content blocked by safety filters (SAFETY)."
            return f"Gemini Summary failed: Generation stopped prematurely (Reason: {reason})."

        return "Gemini Summary failed: Empty or unparseable response."

    except APIError as e:
        return f"Gemini API Summary failed: Google API error occurred. Error: {e}"
    except Exception as e:
        return f"Gemini Summary failed: Unexpected Python error occurred. Error: {e}"


# --- Feedback Update Endpoint (POST /update_feedback) ---
@app.route('/update_feedback', methods=['POST'])
def update_feedback():
    if not AGENT:
        return jsonify({"status": "error", "message": "RL Agent not initialized. Check data files."}), 500

    data = request.json
    required_keys = ['candidate_id', 'jd_id', 'feedback_score', 'comment']
    if not all(key in data for key in required_keys):
        return jsonify({"status": "error", "message": "Missing data fields."}), 400

    # --- Update RL Agent ---
    feedback_entry = pd.Series({
        'candidate_id': data['candidate_id'],
        'jd_id': data['jd_id'],
        'feedback_score': data['feedback_score'],
        'comment': data['comment']
    })

    AGENT.update_reward(feedback_entry)

    # --- Generate Gemini Summary ---
    summary = summarize_feedback_with_gemini(
        data['candidate_id'],
        data['jd_id'],
        data['comment'],
        data['feedback_score']
    )

    # --- Determine Updated RL Policy ---
    s_prime_tuple = AGENT.get_state(data['candidate_id'], data['jd_id'], data['comment'])
    best_action_index = AGENT.choose_action(s_prime_tuple)
    policy_action = ['accept', 'reject', 'reconsider'][best_action_index]

    # --- Optional: Log all feedbacks to CSV for dashboard tracking ---
    try:
        new_entry = pd.DataFrame([{
            "candidate_id": data['candidate_id'],
            "jd_id": data['jd_id'],
            "feedback_score": data['feedback_score'],
            "comment": data['comment'],
            "feedback_summary": summary,
            "policy_action": policy_action
        }])

        if not os.path.exists(FEEDBACK_LOG_PATH):
            new_entry.to_csv(FEEDBACK_LOG_PATH, index=False)
        else:
            new_entry.to_csv(FEEDBACK_LOG_PATH, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Warning: Failed to log feedback: {e}")

    # --- Return JSON Response ---
    return jsonify({
        "status": "updated_and_summarized",
        "candidate_id": data['candidate_id'],
        "jd_id": data['jd_id'],
        "rl_policy_change": f"New Policy suggests '{policy_action}' for this state.",
        "feedback_summary": summary
    })


# --- Main Entry ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
