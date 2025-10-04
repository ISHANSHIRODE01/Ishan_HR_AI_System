from flask import Flask, request, jsonify
from main import predict
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return "HR AI System API is running!"

@app.route('/predict', methods=['POST'])
def make_prediction():
    """
    Expects JSON:
    {
        "cv_texts": ["CV text 1", "CV text 2"],
        "jd_texts": ["JD text 1", "JD text 2"],
        "feedbacks": [ {"feedback": "...", "sentiment": "..."} ]  # optional
    }
    """
    data = request.json

    cv_texts = data.get("cv_texts", [])
    jd_texts = data.get("jd_texts", [])
    feedbacks = data.get("feedbacks", None)

    if not cv_texts or not jd_texts:
        return jsonify({"error": "cv_texts and jd_texts are required"}), 400

    # Convert feedbacks list to DataFrame if provided
    feedback_df = pd.DataFrame(feedbacks) if feedbacks else None

    try:
        final_df = predict(cv_texts, jd_texts, feedback_df)
        # Convert final_df to JSON
        result = final_df.to_dict(orient="records")
        return jsonify({"predictions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
