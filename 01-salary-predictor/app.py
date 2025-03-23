import streamlit as st 
import numpy as np 
import pickle
import os

# Load the model
try:
    with open("./model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: The file 'model.pkl' was not found. Please ensure it is in the correct directory.")
    st.stop()


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
