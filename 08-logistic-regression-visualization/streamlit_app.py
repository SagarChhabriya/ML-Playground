import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="LogReg Visualizer", page_icon="✨")





# Function to load and plot the initial dataset
def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
    elif dataset == "Multiclass":
        X, y = make_blobs(n_features=2, centers=3, random_state=2)

    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
    return X, y

# Function to draw meshgrid from dataset
def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

# Streamlit UI setup
plt.style.use('fivethirtyeight')
st.sidebar.markdown("# Logistic Regression Classifier")

dataset = st.sidebar.selectbox('Select Dataset', ('Binary', 'Multiclass'))

penalty = st.sidebar.selectbox('Regularization (penalty)', ('l2', 'l1', 'elasticnet', 'none'))

solver = st.sidebar.selectbox('Solver', ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))

c_input = float(st.sidebar.number_input('C (Inverse Regularization Strength)', value=1.0))

max_iter = int(st.sidebar.number_input('Max Iterations', value=100))

multi_class = st.sidebar.selectbox('Multi Class Strategy', ('auto', 'ovr', 'multinomial'))

# Only show l1_ratio input if 'elasticnet' is selected
l1_ratio = None
if penalty == 'elasticnet':
    l1_ratio = st.sidebar.number_input('l1 Ratio (only for elasticnet)', min_value=0.0, max_value=1.0, value=0.5)

# Load and plot initial dataset
fig, ax = plt.subplots()
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
plot_handle = st.pyplot(fig)

# Run model on button click
if st.sidebar.button('Run Algorithm'):
    ax.clear()  # Clear previous plot

    # Retrain data and prepare for meshgrid
    X, y = load_initial_graph(dataset, ax)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    try:
        # Build classifier with appropriate parameters
        clf_params = {
            'penalty': penalty,
            'C': c_input,
            'solver': solver,
            'max_iter': max_iter,
            'multi_class': multi_class
        }

        # Only pass l1_ratio if needed
        if penalty == 'elasticnet':
            clf_params['l1_ratio'] = l1_ratio

        clf = LogisticRegression(**clf_params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        XX, YY, input_array = draw_meshgrid(X)
        labels = clf.predict(input_array)

        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        plot_handle.pyplot(fig)
        st.subheader("Accuracy for Logistic Regression: " + str(round(accuracy_score(y_test, y_pred), 2)))

    except ValueError as e:
        st.error(f"Invalid parameter combination: {e}")

