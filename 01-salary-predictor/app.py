import streamlit as st 
import numpy as np 
import pickle
import os



# To make the code work both locally and on Streamlit Cloud, you can use a dynamic approach to resolve the file path. 

# Function to resolve the model path
def get_model_path():
    # Check if running on Streamlit Cloud
    if "STREAMLIT_CLOUD" in os.environ:
        # Path for Streamlit Cloud
        return "01-salary-predictor/model.pkl"
    else:
        # Path for local environment
        return os.path.join(os.path.dirname(__file__), "model.pkl")


with open(get_model_path(), 'rb') as file:
    model = pickle.load(file)

# Title of the App
st.title("Salary Predictor")

# Input Filed

app_input = st.number_input(
    "Enter Years of Experience",
    min_value=0,
    max_value= 50,
    value=0
    
)

model_input = np.array([[app_input]])

if st.button("Predict Salary"):
    # Predic salary
    salary = model.predict(model_input)
    
    st.success(f"{np.round(salary[0][0],2)}")
