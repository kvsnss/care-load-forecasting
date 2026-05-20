# Predictive Forecasting of Care Load & Placement Demand

Detailed guide and project requirements for the Predictive Forecasting of Care Load & Placement Demand analysis.
## Background and Context

The UAC Program operates in a high-uncertainty environment, where sudden changes in border activity, policy enforcement, or humanitarian crises can rapidly increase the number of children entering federal care.  While descriptive analytics explain what has already happened, HHS decision-makers require forward-looking intelligence to answer:

• How many children will be under HHS care in the coming days or weeks?

• Will discharge capacity be sufficient to offset incoming transfers?

• When should shelters, medical staff, and caseworkers be scaled up in advance?

This project introduces predictive modeling to enable proactive, rather than reactive, healthcare and child-welfare planning
## Problem Statement

Despite having high-quality daily time-series data, the UAC Program currently lacks:

• Short-term forecasts of children in HHS care

• Predictive estimates of discharge (placement) demand

• Early-warning indicators of upcoming capacity stress

As a result, operational responses are often delayed, increasing:

• Overcrowding risk

• Staff burnout

• Length of stay for children

## Project Objectives

• Forecast the number of children in HHS care

• Estimate future imbalance between intake and exits

• Predict short-term discharge demand

## Secondary Objectives

• Provide early warnings for healthcare planners

• Quantify forecast uncertainty

• Compare statistical vs machine-learning forecasting approaches

## Dataset Description

<img width="502" height="211" alt="Screenshot 2026-05-20 at 7 57 45 PM" src="https://github.com/user-attachments/assets/e0a97c2a-eaa6-401b-b902-f70da1156482" />

## Analytical Methodology (Step-by-Step)

1.Time-Series Preparation

• Convert Date to datetime index

• Ensure continuity of daily observations

• HHandle missing days via interpolation or masking

• Decompose series into trend, seasonality, and residuals

2.Feature Engineering for Forecasting

Derived predictive features include:

• Lag features (t-1, t-7, t-14 values)

• Rolling averages (7-day and 14-day rolling mean and variance)

• Flow-Based Signals: Transfers − Discharges (net pressure indicator)

• Calendar Effects: Day of week, month, holiday proxies (if available)

3.Train–Test Strategy

• Strict time-based split (no random sampling)

• Walk-forward validation

• Multi-horizon evaluation

4.Forecasting Models

• Baseline Models:
 Naïve persistence model, Moving average forecast
 
• Statistical Models:
 ARIMA / SARIMA (trend & seasonality), Exponential smoothing
 
• Machine Learning Models:
 Random Forest Regressor, Gradient Boosting Regressor
 
5.Model Evaluation
<img width="1336" height="534" alt="image" src="https://github.com/user-attachments/assets/55d7f005-fe25-44fc-a081-730128955dc1" />


## Key Performance Indicators (KPIs)
<img width="1352" height="514" alt="image" src="https://github.com/user-attachments/assets/a5b7cdad-6f3c-4deb-b9b8-41075d5474ed" />



## Streamlit Web Application Requirements

Core Modules

• Future Care Load Forecast Chart

• Discharge Demand Forecast Panel

• Model Selection & Comparison

• Confidence Interval Visualization

User Capabilities

• Forecast horizon selector

• Model toggle

• Scenario comparison view

## Deliverables and Submission

• Research paper (EDA, insights, recommendations)

• Streamlit dashboard (live analytics)

• Executive summary for government stakeholders

## Conclusion

This project elevates the UAC dataset from historical reporting to predictive intelligence. By applying rigorous time-series and machine-learning techniques, it enables HHS stakeholders to anticipate future care demands, allocate resources proactively, and strengthen child-welfare outcomes through data-driven foresight.
