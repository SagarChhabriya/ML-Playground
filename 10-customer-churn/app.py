import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import sys

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", page_icon="✨", layout="wide")

st.title("Customer Churn Prediction App")
st.markdown("""
This app predicts whether a customer will churn based on their behavior and demographics.
Adjust the parameters below and click **Predict** to see the result!
""")

# Load the trained model and scaler with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/logistic_regression_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        st.success("✅ Model loaded successfully!")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please make sure the model files are in the 'models' directory:")
        st.info("- logistic_regression_model.pkl")
        st.info("- scaler.pkl") 
        st.info("- feature_names.pkl")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

model, scaler, feature_names = load_model()

# Sidebar for input parameters
st.sidebar.header("Customer Parameters")

def user_input_features():
    call_failure = st.sidebar.slider('Call Failure', 0, 36, 8)
    complains = st.sidebar.selectbox('Complains', [0, 1])
    subscription_length = st.sidebar.slider('Subscription Length (months)', 3, 47, 32)
    charge_amount = st.sidebar.slider('Charge Amount', 0, 10, 1)
    seconds_of_use = st.sidebar.slider('Seconds of Use', 0, 17000, 4500)
    frequency_of_sms = st.sidebar.slider('Frequency of SMS', 0, 500, 70)
    distinct_called_numbers = st.sidebar.slider('Distinct Called Numbers', 0, 97, 23)
    age = st.sidebar.slider('Age', 15, 55, 30)
    
    # Use cleaned feature names (without extra spaces)
    data = {
        'Call Failure': call_failure,
        'Complains': complains,
        'Subscription Length': subscription_length,
        'Charge Amount': charge_amount,
        'Seconds of Use': seconds_of_use,
        'Frequency of SMS': frequency_of_sms,
        'Distinct Called Numbers': distinct_called_numbers,
        'Age': age
    }
    
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display user input
st.header("Customer Details")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Call Failure", input_df['Call Failure'].iloc[0])
    st.metric("Subscription Length", f"{input_df['Subscription Length'].iloc[0]} months")

with col2:
    st.metric("Charge Amount", input_df['Charge Amount'].iloc[0])
    st.metric("Age", input_df['Age'].iloc[0])

with col3:
    st.metric("Seconds of Use", f"{input_df['Seconds of Use'].iloc[0]:,}")
    st.metric("Distinct Called Numbers", input_df['Distinct Called Numbers'].iloc[0])

with col4:
    st.metric("Frequency of SMS", input_df['Frequency of SMS'].iloc[0])
    complain_status = "Yes" if input_df['Complains'].iloc[0] == 1 else "No"
    st.metric("Complains", complain_status)

# Prediction with error handling
if st.button('Predict Churn Probability', type="primary"):
    if model is None or scaler is None or feature_names is None:
        st.error("Cannot make prediction - model not loaded properly.")
    else:
        try:
            # Prepare input data - align with training feature names
            # Create a properly formatted input DataFrame
            formatted_data = {}
            
            # Map the input features to the expected feature names
            feature_mapping = {
                'Call Failure': 'Call Failure',
                'Complains': 'Complains', 
                'Subscription Length': 'Subscription Length',
                'Charge Amount': 'Charge Amount',
                'Seconds of Use': 'Seconds of Use',
                'Frequency of SMS': 'Frequency of SMS',
                'Distinct Called Numbers': 'Distinct Called Numbers',
                'Age': 'Age'
            }
            
            for expected_feature in feature_names:
                # Find the corresponding input feature
                input_feature = expected_feature
                if input_feature in input_df.columns:
                    formatted_data[expected_feature] = input_df[input_feature].iloc[0]
                else:
                    # Try to find a close match
                    matched = False
                    for imp_feature in input_df.columns:
                        if imp_feature.replace(' ', '').lower() == expected_feature.replace(' ', '').lower():
                            formatted_data[expected_feature] = input_df[imp_feature].iloc[0]
                            matched = True
                            break
                    
                    if not matched:
                        st.error(f"Feature mismatch: {expected_feature} not found in input")
                        st.stop()
            
            # Create the final input DataFrame
            prediction_input = pd.DataFrame(formatted_data, index=[0])
            
            # Ensure correct column order
            prediction_input = prediction_input[feature_names]
            
            # Scale the input
            input_scaled = scaler.transform(prediction_input)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Display results
            st.header("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 1:
                    st.error(f"🚨 High Churn Risk: {prediction_proba[0][1]:.1%}")
                    st.write("This customer has a high probability of churning.")
                else:
                    st.success(f"✅ Low Churn Risk: {prediction_proba[0][0]:.1%}")
                    st.write("This customer is likely to stay.")
            
            with col2:
                # Probability gauge
                proba = prediction_proba[0][1]
                st.metric("Churn Probability", f"{proba:.1%}")
                
                # Interpretation
                if proba < 0.3:
                    st.info("**Low Risk**: Customer is satisfied")
                elif proba < 0.7:
                    st.warning("**Medium Risk**: Monitor customer behavior")
                else:
                    st.error("**High Risk**: Immediate action recommended")

            # Show detailed probabilities
            st.subheader("Detailed Probabilities")
            prob_df = pd.DataFrame({
                'Class': ['Not Churn', 'Churn'],
                'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
            })
            
            st.bar_chart(prob_df.set_index('Class')['Probability'])
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.info("Debug information:")
            st.write(f"Input features: {list(input_df.columns)}")
            st.write(f"Expected features: {feature_names}")
            st.write(f"Formatted data keys: {list(formatted_data.keys())}")

# Add some insights
st.markdown("---")
st.header("📈 Model Insights")
st.markdown("""
**Top factors influencing churn prediction:**
- High Call Failure rates
- Customer complaints  
- Charge amounts
- Usage patterns (Seconds of Use, SMS Frequency)

**Recommendations for high-risk customers:**
- Offer personalized retention offers
- Proactive customer service outreach
- Review service quality issues
- Consider loyalty programs
""")

# Footer
st.markdown("---")
st.markdown("*Built with Logistic Regression • Customer Churn Prediction*")

# Debug section (collapsed by default)
with st.expander("Debug Information"):
    if feature_names:
        st.write("Expected feature names:", feature_names)
    st.write("Input feature names:", list(input_df.columns))
