import streamlit as st
import pandas as pd
from main import predict

st.title("🤖 HR AI System")
st.write("Upload CVs, JDs, and feedback to get AI-driven hiring recommendations")

cv_input = st.text_area("Enter CV Text:")
jd_input = st.text_area("Enter JD Text:")
feedback = st.text_area("Enter Feedback (optional):")

if st.button("Predict"):
    if cv_input and jd_input:
        df = predict([cv_input], [jd_input], None)
        st.success("Prediction complete!")
        st.dataframe(df)
    else:
        st.error("Please enter both CV and JD text.")
