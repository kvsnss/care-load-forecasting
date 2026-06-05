import streamlit as st
import numpy as np
import onnxruntime as rt

# Load ONNX models
rf_sess = rt.InferenceSession("random_forest.onnx")
gb_sess = rt.InferenceSession("gradient_boosting.onnx")

# Helper function to run ONNX prediction
def predict_with_onnx(session, X):
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    return session.run([label_name], {input_name: X.astype(np.float32)})[0]

st.title("Predictive Forecasting of Care Load & Placement Demand")

# Forecast horizon selector
horizon = st.slider("Select forecast horizon (days)", 1, 30, 7)

# Model toggle
model_choice = st.radio("Choose model", ["Random Forest", "Gradient Boosting"])

# Generate predictions
X_input = np.arange(horizon * 4).reshape(horizon, 4)

if model_choice == "Random Forest":
    forecast = predict_with_onnx(rf_sess, X_input)
else:
    forecast = predict_with_onnx(gb_sess, X_input)

# Forecast chart
st.header("Future Care Load Forecast")
st.line_chart(forecast)

# Confidence interval visualization (simple demo)
mean_pred = np.mean(forecast)
ci_low = mean_pred - 5
ci_high = mean_pred + 5
st.write(f"95% Confidence Interval: [{ci_low}, {ci_high}]")

# Scenario comparison view
col1, col2 = st.columns(2)
with col1:
    st.subheader("Random Forest Forecast")
    st.line_chart(predict_with_onnx(rf_sess, X_input))
with col2:
    st.subheader("Gradient Boosting Forecast")
    st.line_chart(predict_with_onnx(gb_sess, X_input))
