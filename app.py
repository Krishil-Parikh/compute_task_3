import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

model = joblib.load('/Users/krishilparikh/CODING/Compute/compute_task_3/svm_credit_risk_model.pkl')

def standardize_input(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

st.title("Credit Risk Prediction App")
st.header("Enter Applicant's Information")

age = st.number_input("Person's Age", min_value=18, max_value=100, value=25)
income = st.number_input("Person's Income", min_value=0, value=50000)
amnt = st.number_input("Loan Amount", min_value=0, value=10000)
int_rate = st.number_input("Loan Interest Rate", min_value=0.0, value=10.0)
emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=40, value=5)
percent_income = st.number_input("Loan Percent Income", min_value=0.0, value=0.1)
cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=30, value=10)

home_ownership = st.selectbox("Home Ownership Status", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
intent = st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
cb_person_default_on_file = st.selectbox("Default On File", ['Y', 'N'])

# Map categorical values
home_ownership_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
intent_mapping = {'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 2, 'PERSONAL': 3, 'DEBTCONSOLIDATION': 4, 'HOMEIMPROVEMENT': 5}
grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
default_on_file_mapping = {'N': 0, 'Y': 1}

# Collect input into a DataFrame
input_data = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'loan_amnt': [amnt],
    'loan_int_rate': [int_rate],
    'person_emp_length': [emp_length],
    'loan_percent_income': [percent_income],
    'cb_person_cred_hist_length': [cred_hist_length],
    'person_home_ownership': [home_ownership_mapping[home_ownership]],
    'loan_intent': [intent_mapping[intent]],
    'loan_grade': [grade_mapping[grade]],
    'cb_person_default_on_file': [default_on_file_mapping[cb_person_default_on_file]]
})

# Standardize the numerical columns
numerical_cols = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'person_emp_length', 'loan_percent_income', 'cb_person_cred_hist_length']
scaler = StandardScaler()
input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])

if st.button("Predict Credit Risk"):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.success("Loan Status: Likely to default")
    else:
        st.success("Loan Status: Likely to repay")

