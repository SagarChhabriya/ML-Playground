# Step 1: Import Libraries
import streamlit as st 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Step 2: Set Page
st.set_page_config(page_title='IRIS', page_icon='🌷')

st.write("# IRIS Flower Classifier App")

st.sidebar.header("Parameters")

def input_features():
    # Adding the minimum, maximum, and step values to the sliders for each feature
    sepal_length = st.sidebar.slider('Sepal Length (cm)', min_value=4.0, max_value=8.0, value=5.0, step=0.1)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', min_value=2.0, max_value=4.5, value=3.0, step=0.1)
    petal_length = st.sidebar.slider('Petal Length (cm)', min_value=1.0, max_value=7.0, value=3.0, step=0.1)
    petal_width = st.sidebar.slider('Petal Width (cm)', min_value=0.1, max_value=3.0, value=1.0, step=0.1)

    # Storing the user input into a dictionary and then converting it into a DataFrame
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

    features = pd.DataFrame(data=data, index=[0])
    
    return features

# Cache the model to avoid retraining every time
@st.cache_resource
def train_model():
    # Load the iris dataset
    iris = load_iris()
    # Train a RandomForest classifier
    model = RandomForestClassifier()
    model.fit(iris.data, iris.target)
    return model, iris

# Load the trained model once
model, iris = train_model()

# Collecting and Displaying the user input
df = input_features()

st.subheader("Parameter Values")
st.write(df)

# Step 4: Predict the flower species based on the user's input
prediction = model.predict(df)

flower_names = iris.target_names
predicted_flower = flower_names[prediction][0]

st.subheader("Prediction")
st.success(f"The predicted species is: **{predicted_flower}**")
