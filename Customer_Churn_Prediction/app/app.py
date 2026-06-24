import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Load Model and Feature Columns
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "customer_churn_model.pkl")
)

features = joblib.load(
    os.path.join(BASE_DIR, "models", "feature_columns.pkl")
)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a telecom customer is likely to churn.")

# -----------------------------
# User Inputs
# -----------------------------
senior = st.selectbox("Senior Citizen", [0, 1])

tenure = st.slider(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=500.0
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):

    data = dict.fromkeys(features, 0)

    data["SeniorCitizen"] = senior
    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly_charges
    data["TotalCharges"] = total_charges

    if gender == "Male":
        data["gender_Male"] = 1

    if partner == "Yes":
        data["Partner_Yes"] = 1

    if dependents == "Yes":
        data["Dependents_Yes"] = 1

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ Customer likely to churn\n\nProbability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Customer likely to stay\n\nProbability of churn: {probability:.2%}"
        )

    st.write("---")
    st.write(f"**Churn Probability:** {probability:.2%}")
