import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Train model in background
X, y = make_classification(
    n_samples=1000, n_features=8,
    n_informative=6, n_redundant=2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# App UI
st.title("💳 Credit Risk Predictor")
st.write("Enter your financial details to check your loan default risk.")

st.sidebar.header("Input Your Details")

income = st.sidebar.slider("Annual Income (scaled)", -3.0, 3.0, 0.0)
age = st.sidebar.slider("Age (scaled)", -3.0, 3.0, 0.0)
loan_amount = st.sidebar.slider("Loan Amount (scaled)", -3.0, 3.0, 0.0)
credit_score = st.sidebar.slider("Credit Score (scaled)", -3.0, 3.0, 0.0)
employment_years = st.sidebar.slider("Employment Years (scaled)", -3.0, 3.0, 0.0)
debt_ratio = st.sidebar.slider("Debt Ratio (scaled)", -3.0, 3.0, 0.0)
num_credit_lines = st.sidebar.slider("Number of Credit Lines (scaled)", -3.0, 3.0, 0.0)
payment_history = st.sidebar.slider("Payment History (scaled)", -3.0, 3.0, 0.0)

# Prediction
input_data = np.array([[income, age, loan_amount, credit_score,
                         employment_years, debt_ratio,
                         num_credit_lines, payment_history]])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0]

st.subheader("🔍 Prediction Result")
if prediction == 0:
    st.success(f"✅ Low Risk — No Default Expected ({probability[0]*100:.1f}% confidence)")
else:
    st.error(f"⚠️ High Risk — Default Likely ({probability[1]*100:.1f}% confidence)")

st.subheader("📊 Risk Probability")
col1, col2 = st.columns(2)
col1.metric("No Default", f"{probability[0]*100:.1f}%")
col2.metric("Default Risk", f"{probability[1]*100:.1f}%")

st.subheader("📋 Your Input Summary")
import pandas as pd
input_df = pd.DataFrame({
    'Feature': ['Income', 'Age', 'Loan Amount', 'Credit Score',
                'Employment Years', 'Debt Ratio',
                'Credit Lines', 'Payment History'],
    'Value': [income, age, loan_amount, credit_score,
              employment_years, debt_ratio,
              num_credit_lines, payment_history]
})
st.dataframe(input_df)