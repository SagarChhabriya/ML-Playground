import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="California Housing", page_icon='💰')
st.write("# California Housing")

# Unpack the data and target
X, y = fetch_california_housing(return_X_y=True)

# Convert to DataFrame
X = pd.DataFrame(X, columns=fetch_california_housing().feature_names)
y = pd.DataFrame(y, columns=['MedHouseVal'])

# Caching the model training process (train the model once)
@st.cache_resource
def train(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Check if model is already cached
if 'model' not in st.session_state:
    st.session_state.model = train(X, y)

model = st.session_state.model

# Sidebar for user input
st.sidebar.header("Enter Parameters")
def parameters():
    features = {name: st.sidebar.number_input(name, value=float(X[name].mean())) for name in X.columns}
    return pd.DataFrame([features])

# Create a DataFrame with user input values
df = parameters()
st.header("Entered Values")
st.write(df)

# Make predictions with the pre-trained model
y_pred = model.predict(df)

# Show the predicted value
st.header("Predicted House Value")
st.success(f"The predicted median house value for the entered parameters is: ${y_pred[0] * 1000:.2f}")
