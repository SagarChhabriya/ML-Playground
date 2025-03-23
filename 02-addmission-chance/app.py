import streamlit as st 
import numpy as np
import pickle 
import os 

st.set_page_config("Admission Chance",page_icon="🎟️")

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
    

st.title("Admission Chance Predictor 🎟️")

# # Input Fields
# Column: GRE Score, Max: 340, Min: 290
# Column: TOEFL Score, Max: 120, Min: 92
# Column: University Rating, Max: 5, Min: 1
# Column:  SOP, Max: 5.0, Min: 1.0
# Column: LOR , Max: 5.0, Min: 1.0
# Column: CGPA, Max: 9.92, Min: 6.8
# Column: Research, Max: 1, Min: 0

gre_score = st.number_input("GRE Score",min_value=290,max_value=340, value=290)
tofel_score = st.number_input("Tofel Score",min_value=92,max_value=120, value=92)
uni_rating = st.number_input("University Rating",min_value=1,max_value=5, value=1)
sop = st.number_input("SOP",min_value=1,max_value=5, value=1)
lor = st.number_input("LOR",min_value=1,max_value=5, value=1)
cgpa = st.number_input("CGPA",min_value=0,max_value=10, value=0)
research = st.number_input("Research",min_value=0,max_value=1, value=0)


result = model.predict([[gre_score,tofel_score,uni_rating,sop,lor,cgpa,research]])

if st.button("Predict Chance"):
    st.success(np.round(result[0],2))