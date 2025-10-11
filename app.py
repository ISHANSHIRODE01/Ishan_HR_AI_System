# -----------------------------------------------------------
# Adaptive AI HR Brain v2 — Flask Backend (with Gemini + N8N)
# -----------------------------------------------------------
from flask import Flask, request, jsonify
import pandas as pd
import os
import json
import requests

# --- NEW IMPORTS FOR GEMINI ---
from google import genai
from google.genai.errors import APIError

# --- RL Agent Import ---
from utils.rl_agent import RLAgent  # Assuming RLAgent is in utils/

app = Flask(__name__)

# --- File Paths ---
CVS_PATH = 'data/cvs.csv'
JDS_PATH = 'data/jds.csv'
FEEDBACK_LOG_PATH = 'data/feedback_log.csv'  # Optional log for dashboard tracking

# --- Initialize RL Agent ---
try:
    AGENT = RLAgent(CVS_PATH, JDS_PATH)
    print("✅ RL Agent initialized successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}. Ensure data/cvs.csv and data/jds.csv exist.")
    AGENT = None

# --- Initialize Gemini Client ---
try:
    GEMINI_CLIENT = genai.Client()  # Uses GEMINI_API_KEY or GOOGLE_API_KEY automatically
    print("✅ Gemini Client initialized successfully.")
except Exception as e:
    print(f"⚠️ Gemini Client Error: {e}. Check GEMINI_API_KEY environment variable.")
    GEMINI_CLIENT = None


# -----------------------------------------------------------
# Helper: Summarize Feedback using Gemini
# -----------------------------------------------------------
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
    a concise sentence summary (max 15 words) suitable for an HR Slack channel.
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
                return "⚠️ Gemini Summary blocked by safety filters."
            return f"⚠️ Gemini Summary incomplete (Reason: {reason})."

        return "⚠️ Gemini Summary empty or unparseable."

    except APIError as e:
        return f"⚠️ Gemini API Error: {e}"
    except Exception as e:
        return f"⚠️ Unexpected Error: {e}"


# -----------------------------------------------------------
# Helper: Send Feedback Data to N8N Workflow
# -----------------------------------------------------------
def send_feedback_to_n8n(candidate_id, jd_id, feedback_score, comment, summary):
    """
    Sends feedback data to N8N webhook for automation (e.g., summary → Slack/email).
    """
    webhook_url = "http://localhost:5678/webhook/feedback_update"  # Replace with your actual N8N webhook URL

    payload = {
        "candidate_id": candidate_id,
        "jd_id": jd_id,
        "feedback_score": feedback_score,
        "comment": comment,
        "summary": summary
    }

    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            print(f"[N8N ✅] Feedback sent successfully → Candidate {candidate_id}, Job {jd_id}")
        else:
            print(f"[N8N ⚠️] Failed to send feedback: Status {response.status_code}")
    except Exception as e:
        print(f"[N8N ❌] Error sending feedback: {e}")


# -----------------------------------------------------------
# Route: Home (Simple Health Check)
# -----------------------------------------------------------
@app.route('/')
def home():
    return "✅ HR RL Agent Backend is Running! Use /update_feedback (POST) to send feedback."


# -----------------------------------------------------------
# Route: POST /update_feedback
# -----------------------------------------------------------
@app.route('/update_feedback', methods=['POST'])
def update_feedback():
    if not AGENT:
        return jsonify({"status": "error", "message": "RL Agent not initialized. Check data files."}), 500

    data = request.json
    required_keys = ['candidate_id', 'jd_id', 'feedback_score', 'comment']
    if not all(key in data for key in required_keys):
        return jsonify({"status": "error", "message": "Missing one or more required fields."}), 400

    # --- Update RL Agent with Feedback ---
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

    # --- Get New Policy Suggestion ---
    s_prime_tuple = AGENT.get_state(data['candidate_id'], data['jd_id'], data['comment'])
    best_action_index = AGENT.choose_action(s_prime_tuple)
    policy_action = ['accept', 'reject', 'reconsider'][best_action_index]

    # --- Log Feedback to CSV ---
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
        print(f"⚠️ Failed to log feedback: {e}")

    # --- Send to N8N for Workflow Automation ---
    send_feedback_to_n8n(
        data['candidate_id'],
        data['jd_id'],
        data['feedback_score'],
        data['comment'],
        summary
    )

    # --- Return Response to Client ---
    return jsonify({
        "status": "updated_and_summarized",
        "candidate_id": data['candidate_id'],
        "jd_id": data['jd_id'],
        "rl_policy_change": f"New policy suggests '{policy_action}' for this candidate.",
        "feedback_summary": summary
    })


# -----------------------------------------------------------
# Run Flask App
# -----------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
