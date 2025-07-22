import streamlit as st
import pandas as pd
import sys, os
import re
import joblib
import time
sys.path.append(os.path.abspath("src"))
from src.utils import TfidfWrapper, ravel_values
from src.evaluate_model import predict_resume
from src.feature_engineering import *
from src.data_preprocessing import *


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath('data'))


model = joblib.load('models/model.pkl')
encoder = joblib.load('models/label_encoder.pkl')

def vocab_density(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    unique_words = len(set(words))
    
    return unique_words / (total_words if total_words > 0 else 1)


def clean_input(samp: str) -> str:
    samp = remove_dates(samp)
    samp = remove_newlines(samp)
    samp = remove_non_ascii_regex(samp)
    samp = remove_urls(samp)
    samp = clean_text(samp)
    samp = strip_column_whitespace_inplace(samp)
    return samp



st.set_page_config(page_title= "RESUME AI", page_icon= "📄", layout= "centered")
st.title("Resume Screening AI")
st.markdown("Paste your Resume for screening ")


resume_text = st.text_area("Paste your resume here:", height= 300)

if st.button("Analyze Resume"):
    if not resume_text.strip():
        st.warning("Input Resume to analyze")
    else:
        with st.spinner("Analyzing..."):
            sample_resume = clean_input(resume_text)
            result = predict_resume(text= sample_resume, encoder= encoder, pipeline= model)


        st.success("Analysis Complete!")
        st.subheader("Prediction")
        st.markdown(f"Predicted Role: {str(result["prediction"]).upper()}")
        st.subheader("Salary")
        st.markdown(f"Salary range: ${result["salary range"]}k")

        st.subheader("Potential Job Titles:")
        for job in result["potential job titles"]:
            st.markdown(f"- {job}")

            
            
        st.subheader("Confidence Score")
        conf_int = int(result["confidence"])
        progress_bar = st.progress(0, text="Calculating!")
        for percent_comp in range(1, conf_int +1):
            time.sleep(0.1)
            progress_bar.progress(percent_comp, text= f"{percent_comp}% ")
        

        if conf_int >= 85:
            st.metric(label= " ", value= f"{conf_int}%", delta= "High")
        elif conf_int >=60: 
            st.metric(label= " ", value= f"{conf_int}%", delta= "Medium")
        else:
            st.metric(label= " ", value= f"{conf_int}%", delta= "Low")


            
            



















