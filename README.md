Ishan HR AI System
A Python-based HR automation system that analyzes CVs and job descriptions, performs sentiment analysis, and uses a Reinforcement Learning (RL) agent to make data-driven hiring decisions.

🚀 Purpose of the System
The Ishan HR AI System automates HR recruitment by:
Matching CVs to job descriptions using similarity scoring
Performing sentiment analysis on candidate profiles
Making hiring decisions (Hire, Reject, Reassign) using a reinforcement learning agent
Generating visual insights and reports for HR teams

💻 How to Run Locally
Clone the repository
git clone https://github.com/ISHANSHIRODE01/Ishan_HR_AI_System.git
cd Ishan_HR_AI_System

Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows


Install dependencies
pip install -r requirements.txt


Run the main script
python main.py


Outputs will be saved in the data/ folder:
match_scores.csv → CV-JD similarity scores
sentiment_results.csv → Sentiment analysis of candidates
final_results.csv → Final decisions (Hire, Reject, Reassign)
Visualizations → PNG charts of analysis

📂 Example Inputs/Outputs
Input: Candidate CVs and Job Description files placed in the data/ folder.
Output:
File	Description
match_scores.csv	CV-JD similarity scores
sentiment_results.csv	Sentiment analysis results
final_results.csv	Final HR decisions
Visualizations (PNG)	Charts summarizing decisions and scores

🤖 How the RL Agent Behaves
The Reinforcement Learning agent:
Evaluates candidate-job matches using similarity scores and sentiment
Decides between Hire, Reject, or Reassign
Learns from past outcomes over multiple episodes to improve decision accuracy

⚙️ How to Extend the System
Support more CV formats (PDF, DOCX)
Integrate advanced NLP models for better skill extraction
Add more criteria for RL decisions (experience, certifications, etc.)
Build a web-based dashboard for HR users
Incorporate feedback loops to continuously improve agent decisions

📌 License
MIT License © Ishan Shirode
