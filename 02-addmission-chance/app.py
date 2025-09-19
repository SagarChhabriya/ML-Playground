import streamlit as st 
import numpy as np
import pickle 
import os 
import pandas as pd 

st.set_page_config("Admission Chance",page_icon="🎟️")

st.title("Admission Chance Predictor 🎟️")



st.write("""
This app predicts your probability of admission to graduate programs based on your academic profile.
Enter your information below and click the 'Predict Chance Probability' button.
""")

# Function to resolve the model path
def get_model_path():
    # Check if running on Streamlit Cloud
    if "STREAMLIT_CLOUD" in os.environ:
        # Path for Streamlit Cloud
        return "02-addmission-chance/model.pkl"
    else:
        # Path for local environment
        return os.path.join(os.path.dirname(__file__), "model.pkl")
    
with open(get_model_path(),'rb') as file: 
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


    



gre_score = st.number_input("GRE Score",min_value=260.0,max_value=340.0, value=310.0, step=1.0)
toefl_score = st.number_input("Toefl Score",min_value=0.0,max_value=120.0, value=90.0, step=1.0)
uni_rating = st.number_input("University Rating",min_value=1.0,max_value=5.0, value=2.0, step=1.0)
sop = st.number_input("SOP",min_value=1.0,max_value=5.0, value=2.0, step=1.0)
lor = st.number_input("LOR",min_value=1.0,max_value=5.0, value=2.0, step=1.0)
cgpa = st.number_input("CGPA",min_value=0.0,max_value=4.0, value=2.0, step=0.1)
research = st.number_input("Research",min_value=0,max_value=1, value=0)

cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']


if st.button("Predict Chance Probability"):
    
    input_data = pd.DataFrame([[gre_score, toefl_score, uni_rating, sop, lor, cgpa, research]], columns=cols)
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)
    admit_prob = prob[0][1]  # positive class probability
    st.success(f"Probability of Admission: {admit_prob * 100:.6f}%")
    # Add a progress bar for visual effect
    st.progress(admit_prob)

st.markdown("---")
st.caption("Admission Chance Predictor | Built with 🤍 using Streamlit by Sagar Chhabriya")
    