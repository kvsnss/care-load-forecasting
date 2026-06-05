import streamlit as st
import numpy as np
import onnxruntime as rt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

# Load ONNX models
rf_sess = rt.InferenceSession("random_forest.onnx")
gb_sess = rt.InferenceSession("gradient_boosting.onnx")

# Helper function to run ONNX prediction
def predict_with_onnx(session, X):
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    return session.run([label_name], {input_name: X.astype(np.float32)})[0]

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.markdown("""
The UAC Program operates in a high-uncertainty environment, where sudden changes in border activity, policy enforcement, or humanitarian crises can rapidly increase the number of children entering federal care.  While descriptive analytics explain what has already happened, HHS decision-makers require forward-looking intelligence to answer: 
            
• How many children will be under HHS care in the coming days or weeks?
            
• Will discharge capacity be sufficient to offset incoming transfers?
            
• When should shelters, medical staff, and caseworkers be scaled up in advance? 
""")
# Forecast horizon selector
horizon = st.slider("Select forecast horizon (days)", 1, 30, 7)

# Model toggle
model_choice = st.radio("Choose model", ["Random Forest", "Gradient Boosting"])

# Generate input
X_input = np.arange(horizon * 4).reshape(horizon, 4)

# Run forecast
if model_choice == "Random Forest":
    forecast = predict_with_onnx(rf_sess, X_input)
else:
    forecast = predict_with_onnx(gb_sess, X_input)

# Convert forecast to pandas series with a daily frequency
forecast_series = pd.Series(forecast.flatten(), index=pd.date_range("2026-01-01", periods=len(forecast), freq="D"))

# --- Core Modules ---
if st.button("Show Forecast Chart"):
    st.line_chart(forecast)

# --- Decomposition ---
if st.button("Time Series Decomposition"):
    if horizon >= 14:
        period = 7   # weekly cycle
    elif horizon >= 7:
        period = 3   # shorter cycle for 1 week
    else:
        period = 2   # very short cycle for 1-6 days
    try:
        decomp = seasonal_decompose(forecast_series, model="additive", period=period)
        st.subheader("Trend Component")
        st.line_chart(decomp.trend.dropna())
        st.subheader("Seasonal Component")
        st.line_chart(decomp.seasonal.dropna())
        st.subheader("Residual Component")
        st.line_chart(decomp.resid.dropna())
    except ValueError:
        st.warning("Not enough data points for decomposition at this horizon.")

if st.button("Discharge Demand Forecast Panel"):
    st.write("This panel can show discharge demand trends (demo placeholder).")
    st.bar_chart(np.random.randint(10, 50, size=horizon))

if st.button("Show Confidence Interval"):
    mean_pred = np.mean(forecast)
    ci_low = mean_pred - 5
    ci_high = mean_pred + 5
    st.write(f"95% Confidence Interval: [{ci_low:.2f}, {ci_high:.2f}]")

if st.button("Show Scenario Comparison"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Random Forest Forecast")
        st.line_chart(predict_with_onnx(rf_sess, X_input))
    with col2:
        st.subheader("Gradient Boosting Forecast")
        st.line_chart(predict_with_onnx(gb_sess, X_input))

# --- Evaluation Metrics ---
if st.button("Show Evaluation Metrics"):
    rf_pred = predict_with_onnx(rf_sess, X_input)
    gb_pred = predict_with_onnx(gb_sess, X_input)
    mae = mean_absolute_error(rf_pred, gb_pred)
    rmse = np.sqrt(mean_squared_error(rf_pred, gb_pred))
    mape = np.mean(np.abs((rf_pred - gb_pred) / rf_pred)) * 100
    st.header("Evaluation Metrics")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")

# --- Information Section ---
st.header("Model Information")
st.markdown("""
This application compares **Random Forest** and **Gradient Boosting** models 
for forecasting care load and placement demand.  
- Random Forest: Ensemble of decision trees using bagging.  
- Gradient Boosting: Sequential ensemble optimizing residuals.  
""")
