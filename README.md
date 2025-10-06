# Ishan HR AI System 🤖💼

**AI-powered HR assistant for CV-JD matching, sentiment analysis, and intelligent hiring decisions.**

---

## 🚀 Features

* **CV/JD Matching:** AI embeddings + similarity scoring + RL-based rules
* **Sentiment Analysis:** Classifies feedback as positive, negative, or neutral
* **Reinforcement Learning Agent:** Optimizes hiring decisions based on past feedback
* **Decision Engine:** Combines all outputs → Hire / Reject / Reassign
* **Visualizations:** Confusion matrix, similarity graphs, sentiment pie, RL reward tracking

---

## ⚡ Quick Start

```bash
git clone https://github.com/ISHANSHIRODE01/Ishan_HR_AI_System.git
cd Ishan_HR_AI_System

# Optional: create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt

# Run the system
python main.py  # or app.py
```

**Notebook Demo:** `notebooks/Ishan_HR_AI_System_demo.ipynb`

**Flask API:** `python app.py` → POST JSON to `/predict`

---

## 📂 Project Structure

```
Ishan_HR_AI_System/
│
├─ main.py / app.py
├─ requirements.txt
├─ data/           # Sample CVs, JDs, feedbacks
├─ models/         # Pre-trained models
├─ utils/          # Preprocessing, scoring, RL logic
├─ notebooks/      # End-to-end demo
└─ visualizations/ # Generated charts
```

---

## 🖼 Example Output

| CV_ID | JD_ID | Similarity | Sentiment | RL Decision | Final Decision |
| ----- | ----- | ---------- | --------- | ----------- | -------------- |
| CV001 | JD001 | 0.87       | Positive  | Hire        | Hire           |
| CV002 | JD002 | 0.45       | Neutral   | Reject      | Reassign       |

---

## 🛠 Bonus

* Flask REST API for easy integration
* Dockerfile for containerization
* GitHub Actions: Linting & automated testing

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push branch (`git push origin feature`)
5. Open a Pull Request
