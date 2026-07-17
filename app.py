import streamlit as st
import pandas as pd
import numpy as np
import joblib

with st.sidebar:
    st.title("Project Information")

    st.markdown("""
### 💳 Credit Card Fraud Detection

**Developer:** Sudha

**Model:** Random Forest

**Data Balancing:** SMOTE

**Features:** 30

**Objective:** Detect fraudulent credit card transactions.
""")

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load Model
model = joblib.load("credit_card_fraud_model_smote.pkl")

# Title
st.title("Credit Card Fraud Detection System")

st.write("""
This web application predicts whether a credit card transaction is **Fraudulent** or **Legitimate**
using a **Random Forest Machine Learning Model** trained with **SMOTE**.
""")

st.success("Model Loaded Successfully")

# ==============================
# INPUT FIELDS START HERE
# ==============================

st.header("Enter Transaction Details")

features = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

input_data = []

col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    if i % 2 == 0:
        value = col1.number_input(feature, format="%.6f", placeholder=f"Enter {feature}")
    else:
        value = col2.number_input(feature, format="%.6f", placeholder=f"Enter {feature}")

    input_data.append(value)

# ==============================
# PREDICTION BUTTON
# ==============================

if st.button("Predict"):

    input_array = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_array)

    probability = model.predict_proba(input_array)

if prediction[0] == 0:
    st.success("Legitimate Transaction")
else:
    st.error("Fraudulent Transaction Detected!")

# Show probabilities
st.subheader("Prediction Details")

legitimate_prob = probability[0][0] * 100
fraud_prob = probability[0][1] * 100

st.write(f"**Legitimate Transaction Probability:** {legitimate_prob:.2f}%")
st.write(f"**Fraudulent Transaction Probability:** {fraud_prob:.2f}%")

st.progress(int(max(legitimate_prob, fraud_prob)))