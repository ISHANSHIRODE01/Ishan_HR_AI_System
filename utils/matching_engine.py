import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocess import clean_text
import joblib

def compute_similarity(cv_texts, jd_texts):
    all_docs = cv_texts + jd_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    cv_vectors = tfidf_matrix[:len(cv_texts)]
    jd_vectors = tfidf_matrix[len(cv_texts):]

    similarity = cosine_similarity(cv_vectors, jd_vectors)
    joblib.dump(vectorizer, "models/tfidf_model.pkl")

    results = []
    for i, cv in enumerate(cv_texts):
        for j, jd in enumerate(jd_texts):
            results.append({
                "CV_ID": f"CV{i+1}",
                "JD_ID": f"JD{j+1}",
                "similarity_score": similarity[i][j]
            })
    return pd.DataFrame(results)
