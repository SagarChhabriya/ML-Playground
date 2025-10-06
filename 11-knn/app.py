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

    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, model_file)
    scaler_path = os.path.join(current_dir, scaler_file)
    columns_path = os.path.join(current_dir, columns_file)

    if all(os.path.exists(p) for p in [model_path, scaler_path, columns_path]):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(columns_path)
        st.success("✅ Loaded model and preprocessing files.")
        return model, scaler, feature_columns
    else:
        st.error("❌ Required files not found in root directory.")
        st.stop()

model, scaler, feature_columns = load_model()

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
st.header("📊 Input Parameters")

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
    
    # Create raw numeric array
    raw_numeric = np.array([[age, bmi, caffeine_mg, coffee_intake, sleep_hours, heart_rate, physical_activity_hours]])
    
    # Debug: Show raw values
    st.write("🔍 **Debug - Raw Numeric Values:**")
    for i, col in enumerate(numeric_features):
        st.write(f"  {col}: {raw_numeric[0, i]}")
    
    try:
        scaled_numeric = scaler.transform(raw_numeric)
        
        st.write("🔍 **Debug - Scaled Numeric Values:**")
        for i, col in enumerate(numeric_features):
            st.write(f"  {col}: {scaled_numeric[0, i]:.4f}")
            
    except Exception as e:
        st.error(f"Scaling error: {e}")
        # If scaling fails, use raw values (not ideal but for debugging)
        scaled_numeric = raw_numeric

    for i, col in enumerate(numeric_features):
        if col in df_input.columns:
            df_input.at[0, col] = scaled_numeric[0, i]

    # Ordinal + binary
    df_input.at[0, 'Sleep_Quality'] = sleep_quality_map[sleep_quality]
    df_input.at[0, 'Smoking'] = smoking
    df_input.at[0, 'Alcohol_Consumption'] = alcohol_consumption

    # One-hot encoded categories
    gender_col = f'Gender_{gender}'
    if gender_col in df_input.columns:
        df_input.at[0, gender_col] = 1
    else:
        st.warning(f"⚠️ Gender column '{gender_col}' not found in features")

    country_col = f'Country_{country}'
    if country_col in df_input.columns:
        df_input.at[0, country_col] = 1
    elif 'Country_Other' in df_input.columns:
        df_input.at[0, 'Country_Other'] = 1
    else:
        st.warning(f"⚠️ Country column for '{country}' not found")

    occupation_col = f'Occupation_{occupation}'
    if occupation_col in df_input.columns:
        df_input.at[0, occupation_col] = 1
    elif 'Occupation_Other' in df_input.columns:
        df_input.at[0, 'Occupation_Other'] = 1
    else:
        st.warning(f"⚠️ Occupation column for '{occupation}' not found")

    # Debug: Show final feature vector
    st.write("🔍 **Debug - Final Feature Values (non-zero only):**")
    non_zero_features = df_input.loc[0][df_input.loc[0] != 0]
    for feature, value in non_zero_features.items():
        st.write(f"  {feature}: {value}")

    return df_input

# --------------------------------------------
# Prediction Button
# --------------------------------------------
if st.button("🔍 Predict Stress Level"):
    with st.spinner("Processing..."):
        X_user = preprocess_input()
        
        # Debug: Show what we're sending to the model
        st.write("🔍 **Debug - Input shape to model:**", X_user.shape)
        
        try:
            # Get both prediction and probabilities
            prediction = model.predict(X_user)[0]
            probabilities = model.predict_proba(X_user)[0]
            
            stress_level = stress_level_map_reverse[prediction]
            
            st.success(f"🎯 Predicted Stress Level: **{stress_level}**")
            
            # Show confidence scores
            st.subheader("📊 Prediction Confidence")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Low Stress", f"{probabilities[0]:.1%}", 
                         delta="✓" if prediction == 0 else "")
            with col2:
                st.metric("Medium Stress", f"{probabilities[1]:.1%}",
                         delta="✓" if prediction == 1 else "")
            with col3:
                st.metric("High Stress", f"{probabilities[2]:.1%}",
                         delta="✓" if prediction == 2 else "")
            
            # Show raw probabilities
            st.write("**Raw probabilities:**", [f"{p:.4f}" for p in probabilities])
            
            # Model analysis
            st.subheader("🔍 Model Analysis")
            
            # Check if model always predicts the same class
            if max(probabilities) > 0.95:
                st.warning("⚠️ Model shows very high confidence (>95%). This might indicate it's always predicting the same class.")
            
            if abs(probabilities[0] - probabilities[1]) < 0.1 and abs(probabilities[1] - probabilities[2]) < 0.1:
                st.info("ℹ️ Probabilities are very close - model is uncertain")
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

# --------------------------------------------
# Test with Extreme Values
# --------------------------------------------
st.header("🧪 Test Extreme Cases")

col1, col2 = st.columns(2)

with col1:
    if st.button("Test Low Stress Profile"):
        st.info("""
        Low stress profile:
        - Age: 30
        - Sleep: 8 hours, Excellent quality  
        - BMI: 22
        - Activity: 15 hours/week
        - No smoking/alcohol
        - Low caffeine
        """)

with col2:
    if st.button("Test High Stress Profile"):
        st.info("""
        High stress profile:
        - Age: 50  
        - Sleep: 5 hours, Poor quality
        - BMI: 30
        - Activity: 2 hours/week
        - Smoking: Yes
        - High caffeine
        """)

# --------------------------------------------
# Model Information
# --------------------------------------------
with st.expander("🔧 Model Information"):
    st.write(f"**Model type:** {type(model).__name__}")
    st.write(f"**Number of features:** {len(feature_columns)}")
    st.write(f"**Feature columns:** {feature_columns}")
    
    # Show first few features
    st.write("**First 10 features:**", feature_columns[:10])
    
    if hasattr(model, 'classes_'):
        st.write(f"**Model classes:** {model.classes_}")
    
    if hasattr(model, 'n_neighbors'):
        st.write(f"**KNN neighbors:** {model.n_neighbors}")
