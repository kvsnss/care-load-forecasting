import streamlit as st
import joblib
import numpy as np

# Load models
rf =joblib.load('random_forest.pkl')
gb = joblib.load('gradient_boosting.pkl')

st.title("Predictive Forecasting of Care Load & Placement Demand")

# Forecast horizon selector
horizon = st.slider("Select forecast horizon (days)", 1, 30, 7)

# Model toggle
model_choice = st.radio("Choose model", ["Random Forest", "Gradient Boosting"])

# Generate predictions
if model_choice == "Random Forest":
    forecast = rf.predict(np.arange(horizon).reshape(-1,1))
else:
    forecast = gb.predict(np.arange(horizon).reshape(-1,1))

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
    st.line_chart(rf.predict(np.arange(horizon).reshape(-1,1)))
with col2:
    st.subheader("Gradient Boosting Forecast")
    st.line_chart(gb.predict(np.arange(horizon).reshape(-1,1)))
