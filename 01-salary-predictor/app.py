import streamlit as st 
import numpy as np 
import pickle


# Load the model
with open("model.pkl", 'rb') as file:
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
