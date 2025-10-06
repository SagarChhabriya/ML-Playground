# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature columns saved during training
model = joblib.load('best_knn_model.joblib')
scaler = joblib.load('scaler.joblib')
feature_columns = joblib.load('feature_columns.joblib')

# Mappings for ordinal variables
sleep_quality_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
stress_level_map_reverse = {0: 'Low', 1: 'Medium', 2: 'High'}

st.title("Stress Level Predictor based on Lifestyle and Health Features")

# User inputs
age = st.slider("Age", 18, 100, 30,42)
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], index=0)
country = st.selectbox("Country", ['Brazil', 'USA', 'India', 'Germany', 'Other'], index=0)
occupation = st.selectbox("Occupation", ['Office', 'Manual', 'Student', 'Other'], index=0)
coffee_intake = st.slider("Coffee Intake (cups/day)", 0.0, 10.0, 5.3)
caffeine_mg = st.slider("Caffeine intake (mg/day)", 0, 800, 503)
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 5.9)
sleep_quality = st.selectbox("Sleep Quality", ['Poor', 'Fair', 'Good', 'Excellent'], index=1)
bmi = st.slider("BMI", 10.0, 50.0, 22.7)
heart_rate = st.slider("Heart Rate (bpm)", 40, 120, 59)
physical_activity_hours = st.slider("Physical Activity Hours per Week", 0.0, 20.0, 11.2)
smoking = st.radio("Smoking", [0, 1], index=0)
alcohol_consumption = st.radio("Alcohol Consumption", [0, 1], index=0) 

def preprocess_input():
    # Create zero DataFrame with all expected feature columns
    df_input = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Numerical features (to be scaled)
    num_cols = ['Age', 'BMI', 'Caffeine_mg', 'Coffee_Intake', 'Sleep_Hours', 'Heart_Rate', 'Physical_Activity_Hours']
    raw_num = np.array([[age, bmi, caffeine_mg, coffee_intake, sleep_hours, heart_rate, physical_activity_hours]])
    scaled_num = scaler.transform(raw_num)
    for i, col in enumerate(num_cols):
        df_input.at[0, col] = scaled_num[0, i]

    # Ordinal and binary features
    df_input.at[0, 'Sleep_Quality'] = sleep_quality_map[sleep_quality]
    df_input.at[0, 'Smoking'] = smoking
    df_input.at[0, 'Alcohol_Consumption'] = alcohol_consumption

    # One-hot encode Gender
    gender_col = f'Gender_{gender}'
    if gender_col in df_input.columns:
        df_input.at[0, gender_col] = 1

    # One-hot encode Country
    country_col = f'Country_{country}'
    if country_col in df_input.columns:
        df_input.at[0, country_col] = 1
    else:
        # If unseen country, assign to Country_Other if exists
        if 'Country_Other' in df_input.columns:
            df_input.at[0, 'Country_Other'] = 1

    # One-hot encode Occupation
    occupation_col = f'Occupation_{occupation}'
    if occupation_col in df_input.columns:
        df_input.at[0, occupation_col] = 1
    else:
        # Assign to Occupation_Other if exists
        if 'Occupation_Other' in df_input.columns:
            df_input.at[0, 'Occupation_Other'] = 1

    return df_input

if st.button("Predict Stress Level"):
    X_user = preprocess_input()
    prediction = model.predict(X_user)[0]
    stress_pred = stress_level_map_reverse[prediction]
    st.success(f"Predicted Stress Level: {stress_pred}")
