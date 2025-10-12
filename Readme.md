# Adaptive AI HR Brain v2 — Reinforcement Learning + N8N + Gemini Integration

### Author: **Ishan Shirode**

---

## 🧠 Overview

This project implements an **Adaptive HR Decision System** powered by **Reinforcement Learning (RL)**, **N8N automation**, and **Google Gemini summarization** (used instead of OpenAI due to credit limitations).  
It automatically learns from HR feedback, updates decision policies, generates concise summaries, and triggers workflow automations.

---

## 🚀 Features

- **RL Loop Functionality**  
  Adaptive Q-learning agent updates its policy with each new feedback.

- **Gemini Automation**  
  Generates human-like feedback summaries and fit explanations using Gemini 2.5 Flash API.

- **N8N Workflow Integration**  
  Sends each feedback and summary to N8N via a webhook for automated notifications (Slack/Email).

- **Data Handling & Scaling**  
  Works with 100+ candidate profiles and 10+ job descriptions stored as CSV files.

- **Explainability & Visualization**  
  Streamlit dashboard visualizes reward history, sentiment trends, and RL decisions for full transparency.

- **Documentation & Repo Cleanliness**  
  Well-documented flow, clear folder structure, and modular design.

- **Speed & Efficiency**  
  End-to-end processing and automation run within 2 minutes for demo.

---

## 🧩 System Architecture

```text
Feedback (HR/Candidate)
        ↓
Flask Backend (app.py)
  ├── RLAgent.update_reward()     → Learns from feedback
  ├── Gemini Summary              → Summarizes comment
  ├── send_feedback_to_n8n()      → Triggers automation workflow
        ↓
N8N Workflow (Webhook → Slack/Email)
        ↓
Dashboard (Streamlit)
  ├── Reward Trend
  ├── Sentiment Distribution
  ├── Q-Table / Policy Logs
```

---

## ⚙️ Tech Stack

| Layer | Tools / Libraries |
|-------|--------------------|
| Backend | Flask, Pandas, NumPy, Scikit-Learn |
| AI Core | Custom RL Agent (Q-Learning), TextBlob (Sentiment), Gemini API |
| Automation | N8N Workflow via Webhook |
| Visualization | Streamlit Dashboard, Matplotlib |
| Data | CSV (CVs, JDs, Feedback Log) |

---

## 📁 Project Structure

```text
Ishan_HR_AI_System/
│
├── app.py                    # Flask backend (RL + Gemini + N8N)
├── utils/
│   ├── rl_agent.py           # Reinforcement Learning Agent
│
├── data/
│   ├── cvs.csv               # Candidate profiles
│   ├── jds.csv               # Job descriptions
│   ├── feedback_log.csv      # Feedback + summaries
│
├── dashboard.py              # Streamlit dashboard visualization
└── README.md
```

---

## 🔄 Workflow Demo Summary

### 1️⃣ RL Agent Upgrade  
Agent updates its Q-table and modifies policy after every feedback.

### 2️⃣ User Input Loop  
Feedbacks (via JSON API or form) update the learning state automatically.

### 3️⃣ N8N Integration  
Webhook triggers a workflow: `Feedback → Gemini Summary → Notification`

### 4️⃣ Gemini Summarization  
Summarizes HR comments into short, readable insights.

### 5️⃣ Explainability  
Dashboard shows cumulative rewards, sentiment charts, and updated policy.

### 6️⃣ Scalability  
Handles large CSV datasets and updates results in real-time.

---

## 🧰 API Endpoint

### `POST /update_feedback`

**Description:** Updates the RL model and triggers Gemini + N8N workflow.

**Example Request:**
```bash
curl -X POST http://127.0.0.1:5000/update_feedback -H "Content-Type: application/json" -d '{"candidate_id":1, "jd_id":2, "feedback_score":4, "comment":"Strong technical match, minor communication issue."}'
```

**Example Response:**
```json
{
  "status": "updated_and_summarized",
  "candidate_id": 1,
  "jd_id": 2,
  "rl_policy_change": "New policy suggests 'accept' for this candidate.",
  "feedback_summary": "Strong technical skills; minor communication issue noted."
}
```

---

## 🖥️ Dashboard (Streamlit)

**Run Dashboard:**
```bash
streamlit run dashboard.py
```

**Visual Sections:**
- Cumulative Reward History
- Sentiment Distribution
- Q-Values Table
- Feedback Log & Summary

---

## 🧾 Author

**Ishan Shirode**  
📍 Vasai, India  
🎓 B.E. Artificial Intelligence & Machine Learning  
🔗 [GitHub: ISHANSHIRODE01](https://github.com/ISHANSHIRODE01/Ishan_HR_AI_System)

---
