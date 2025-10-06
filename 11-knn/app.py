import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------
# Load the trained model, scaler, and columns
# --------------------------------------------
@st.cache_resource
def load_model():
    # File names
    model_file = 'best_knn_model.joblib'
    scaler_file = 'scaler.joblib'
    columns_file = 'feature_columns.joblib'

    # Absolute path to current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Full paths to files (root dir)
    model_path = os.path.join(current_dir, model_file)
    scaler_path = os.path.join(current_dir, scaler_file)
    columns_path = os.path.join(current_dir, columns_file)

    if all(os.path.exists(p) for p in [model_path, scaler_path, columns_path]):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(columns_path)
        st.success("Loaded model from root directory.")
        return model, scaler, feature_columns
    else:
        st.error("Required files not found in root directory.")
        st.stop()

# --------------------------------------------
# Mappings
# --------------------------------------------
sleep_quality_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
stress_level_map_reverse = {0: 'Low', 1: 'Medium', 2: 'High'}

# --------------------------------------------
# UI Header
# --------------------------------------------
st.title("Stress Level Predictor")
st.markdown("Predict your stress level based on health and lifestyle inputs.")

# --------------------------------------------
# User Inputs
# --------------------------------------------
age = st.slider("Age", 18, 100, 42)
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

# --------------------------------------------
# Input Preprocessing Function
# --------------------------------------------
def preprocess_input():
    # Empty input row with all columns
    df_input = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Scale numerical features
    numeric_features = ['Age', 'BMI', 'Caffeine_mg', 'Coffee_Intake', 'Sleep_Hours', 'Heart_Rate', 'Physical_Activity_Hours']
    raw_numeric = np.array([[age, bmi, caffeine_mg, coffee_intake, sleep_hours, heart_rate, physical_activity_hours]])
    scaled_numeric = scaler.transform(raw_numeric)

    for i, col in enumerate(numeric_features):
        df_input.at[0, col] = scaled_numeric[0, i]

    # Ordinal + binary
    df_input.at[0, 'Sleep_Quality'] = sleep_quality_map[sleep_quality]
    df_input.at[0, 'Smoking'] = smoking
    df_input.at[0, 'Alcohol_Consumption'] = alcohol_consumption

    # One-hot encoded categories
    gender_col = f'Gender_{gender}'
    if gender_col in df_input.columns:
        df_input.at[0, gender_col] = 1

    country_col = f'Country_{country}'
    if country_col in df_input.columns:
        df_input.at[0, country_col] = 1
    elif 'Country_Other' in df_input.columns:
        df_input.at[0, 'Country_Other'] = 1

    occupation_col = f'Occupation_{occupation}'
    if occupation_col in df_input.columns:
        df_input.at[0, occupation_col] = 1
    elif 'Occupation_Other' in df_input.columns:
        df_input.at[0, 'Occupation_Other'] = 1

    return df_input

# --------------------------------------------
# Prediction Button
# --------------------------------------------
if st.button("🔍 Predict Stress Level"):
    X_user = preprocess_input()
    prediction = model.predict(X_user)[0]
    stress_level = stress_level_map_reverse[prediction]
    st.success(f"🎯 Predicted Stress Level: **{stress_level}**")

