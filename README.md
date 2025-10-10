# 🤖 HR RL Agent: Adaptive Candidate Screening System

This project implements a **Reinforcement Learning (RL) Agent** that automates candidate screening decisions by **learning adaptively from live HR feedback**.  
It connects a **Python backend (Flask)**, **Gemini LLM for feedback summarization**, and a **Streamlit dashboard** for transparent visualization of the agent’s decision-making.

---

## 🚀 Key Features

- **🧠 Adaptive Learning:**  
  Uses **Q-Learning** to continuously update its decision policy based on real HR feedback.

- **⚙️ Automation via API:**  
  Exposes a **REST API endpoint** to integrate with automation tools like **n8n** or **Zapier**.

- **💬 LLM Integration (Gemini):**  
  Summarizes long HR feedback comments into **concise, actionable insights** for notifications.

- **📊 Transparent Dashboard:**  
  The **Streamlit dashboard** visualizes:
  - Reward history (agent learning progress)
  - Current Q-Table (decision policy)

---

## 🛠️ Setup and Installation

### 🔹 Prerequisites
- Python **3.9+**
- Active **Gemini API Key**

---

### 1️⃣ Clone and Install Dependencies
```bash
# Clone the repository
git clone <repo-url>
cd Ishan_HR_AI_System

# Install required Python libraries
pip install flask pandas numpy scikit-learn textblob streamlit plotly google-genai
