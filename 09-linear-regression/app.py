import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Linear Regression Visualizer", page_icon="✨")

# Function to generate regression dataset
def generate_regression_data(n_samples, n_features, noise, random_state, bias=0.0):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
        bias=bias
    )
    return X, y

# Function to plot regression results
def plot_regression_results(X, y, y_pred, model_type, ax, model=None):
    if X.shape[1] == 1:  # Single feature - 2D plot
        ax.scatter(X, y, alpha=0.7, label='Actual', color='blue')
        if model_type != "Polynomial":
            # Sort for nice line plotting
            sorted_idx = np.argsort(X.flatten())
            ax.plot(X[sorted_idx], y_pred[sorted_idx], color='red', linewidth=2, label='Predicted')
        else:
            # For polynomial, we need to plot smoothly
            x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
            y_plot = model.predict(x_plot)
            ax.plot(x_plot, y_plot, color='red', linewidth=2, label='Predicted')
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        ax.legend()
    else:  # Multiple features - show predicted vs actual
        ax.scatter(y, y_pred, alpha=0.7)
        # Perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted")

# Function to plot feature importance for multiple features
def plot_feature_importance(model, model_type, ax, feature_names=None):
    if model_type == 'Polynomial':
        if hasattr(model, 'named_steps'):
            linear_model = model.named_steps['linear']
            if hasattr(linear_model, 'coef_'):
                coefs = linear_model.coef_
                if len(coefs.shape) > 1:
                    coefs = coefs[0]
                # For polynomial, we might have many coefficients - show first 20
                n_to_show = min(20, len(coefs))
                ax.bar(range(n_to_show), coefs[:n_to_show])
                ax.set_xlabel("Polynomial Feature Index")
                ax.set_ylabel("Coefficient Value")
                ax.set_title(f"First {n_to_show} Polynomial Coefficients")
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) > 1:
            coef = model.coef_[0] if len(model.coef_) > 1 else model.coef_
        else:
            coef = model.coef_
        
        features = range(len(coef))
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in features]
        
        ax.bar(feature_names, coef)
        ax.set_xlabel("Features")
        ax.set_ylabel("Coefficient Value")
        ax.set_title("Feature Coefficients")
        plt.xticks(rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Streamlit UI setup
plt.style.use('fivethirtyeight')
st.sidebar.markdown("# Linear Regression Visualizer")

# Dataset parameters
st.sidebar.markdown("## Dataset Parameters")
n_samples = st.sidebar.slider('Number of Samples', 50, 1000, 100)
n_features = st.sidebar.slider('Number of Features', 1, 10, 1)
noise = st.sidebar.slider('Noise Level', 0.1, 50.0, 10.0)
bias = st.sidebar.slider('Bias/Intercept', -10.0, 10.0, 0.0)
random_state = st.sidebar.number_input('Random State', value=42)

# Model selection
st.sidebar.markdown("## Model Parameters")
model_type = st.sidebar.selectbox(
    'Regression Type',
    ('Linear', 'Ridge', 'Lasso', 'ElasticNet', 'Polynomial')
)

# Show warning for polynomial with multiple features
if model_type == 'Polynomial' and n_features > 1:
    st.sidebar.warning("⚠️ Polynomial with multiple features may create many terms!")

# Common parameters
fit_intercept = st.sidebar.checkbox('Fit Intercept', value=True)

# Regularization parameters
if model_type in ['Ridge', 'Lasso', 'ElasticNet']:
    alpha = st.sidebar.number_input('Alpha (Regularization)', 0.01, 10.0, 1.0)
    max_iter = int(st.sidebar.number_input('Max Iterations', 100, 10000, 1000))
    tol = st.sidebar.number_input('Tolerance', 1e-5, 1e-1, 1e-4, format="%.6f")

# ElasticNet specific parameters
if model_type == 'ElasticNet':
    l1_ratio = st.sidebar.slider('L1 Ratio', 0.0, 1.0, 0.5)

# Polynomial specific parameters
if model_type == 'Polynomial':
    degree = st.sidebar.slider('Polynomial Degree', 1, 5, 2)
    include_bias = st.sidebar.checkbox('Include Polynomial Bias', value=True)

# Test size
test_size = st.sidebar.slider('Test Size Ratio', 0.1, 0.5, 0.2)

# Generate dataset
X, y = generate_regression_data(n_samples, n_features, noise, random_state, bias)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Create and train model
try:
    if model_type == 'Linear':
        model = LinearRegression(fit_intercept=fit_intercept)
    elif model_type == 'Ridge':
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, random_state=random_state)
    elif model_type == 'Lasso':
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, random_state=random_state)
    elif model_type == 'ElasticNet':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, random_state=random_state)
    elif model_type == 'Polynomial':
        # Create polynomial features pipeline
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ('linear', LinearRegression(fit_intercept=fit_intercept))
        ])
    
    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Create plots
    if n_features == 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Main regression plot
    plot_regression_results(X_test, y_test, y_pred, model_type, ax1, model)
    ax1.set_title(f'{model_type} Regression - Prediction vs Actual')
    
    # Residual plot
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    
    # Feature importance/coefficients
    feature_names = [f'Feature {i+1}' for i in range(n_features)]
    plot_feature_importance(model, model_type, ax3, feature_names)
    
    # Distribution of errors
    ax4.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residual Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals')

    plt.tight_layout()
    st.pyplot(fig)

    # Display metrics and model info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.4f}")
    with col3:
        st.metric("MSE", f"{mse:.4f}")
    with col4:
        st.metric("MAE", f"{mae:.4f}")

    # Model coefficients and intercept
    st.subheader("Model Details")
    
    if model_type == 'Polynomial':
        if hasattr(model, 'named_steps'):
            linear_model = model.named_steps['linear']
            poly = model.named_steps['poly']
            st.write(f"Polynomial Features: {poly.n_output_features_}")
            if hasattr(linear_model, 'coef_'):
                coefs = linear_model.coef_
                if len(coefs.shape) > 1:
                    coefs = coefs[0]
                st.write(f"Number of Coefficients: {len(coefs)}")
                # Show first 10 coefficients
                st.write("First 10 Coefficients:", coefs[:10])
            if hasattr(linear_model, 'intercept_'):
                st.write("Intercept:", linear_model.intercept_)
    else:
        if hasattr(model, 'coef_'):
            st.write("Coefficients:", model.coef_)
        if hasattr(model, 'intercept_'):
            st.write("Intercept:", model.intercept_)

    # Data summary
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Training set size:", X_train.shape[0])
        st.write("Test set size:", X_test.shape[0])
        st.write("Number of features:", X.shape[1])
    with col2:
        st.write("Data shape:", X.shape)
        if model_type == 'Polynomial':
            if hasattr(model, 'named_steps'):
                poly = model.named_steps['poly']
                st.write(f"Polynomial features: {poly.n_output_features_}")

except Exception as e:
    st.error(f"Error in model training: {str(e)}")
    st.info("Try adjusting the parameters, especially for Polynomial regression with multiple features.")

# Additional information
with st.expander("Understanding the Parameters"):
    st.markdown("""
    **Dataset Parameters:**
    - **Noise Level**: Controls the amount of Gaussian noise added to the output
    - **Bias**: Adds a constant bias term to the target variable
    - **Random State**: Ensures reproducible results
    
    **Model Types:**
    - **Linear**: Ordinary Least Squares regression
    - **Ridge**: L2 regularization to prevent overfitting
    - **Lasso**: L1 regularization that can zero out coefficients
    - **ElasticNet**: Combination of L1 and L2 regularization
    - **Polynomial**: Fits polynomial features to capture non-linear relationships
    
    **Key Metrics:**
    - **R² Score**: Proportion of variance explained (1 is perfect)
    - **RMSE**: Root Mean Square Error (lower is better)
    - **MSE**: Mean Square Error (lower is better)
    - **MAE**: Mean Absolute Error (lower is better)
    
    **Note on Polynomial Regression:**
    - With multiple features, polynomial regression creates interaction terms
    - The number of features grows rapidly: (n_features + degree)! / (n_features! × degree!)
    - For 3 features and degree 2: (3+2)!/(3!×2!) = 10 features
    - Consider using lower degrees with multiple features
    """)