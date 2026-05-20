
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.predict import predict_default


ROOT_DIR = Path(__file__).resolve().parent
META_PATH = ROOT_DIR / "models" / "metadata.json"


st.set_page_config(page_title="Credit Risk Prediction", page_icon=":bank:", layout="centered")
st.title("Credit Risk Prediction App")
st.caption("Enter applicant details to estimate default risk.")

if META_PATH.exists():
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    metrics = metadata.get("metrics", {})
    st.info(
        "Model: {name} | Recall (defaulters): {recall:.2f} | Threshold: {threshold:.2f}".format(
            name=metadata.get("model_name", "unknown"),
            recall=float(metrics.get("test_recall_default_class", 0.0)),
            threshold=float(metadata.get("threshold", 0.5)),
        )
    )
else:
    st.warning("Model artifacts not found. Train the model first using: python src/train.py")

with st.form("prediction_form"):
    applicant_income = st.number_input("Applicant Income", min_value=0, value=50000, step=1000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=200000, step=1000)
    credit_history = st.selectbox("Credit History (1 = good, 0 = poor)", [1, 0])
    employment_length = st.number_input("Employment Length (years)", min_value=0, max_value=60, value=5, step=1)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["home", "car", "education", "personal", "business"],
        index=0,
    )

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    payload = {
        "applicant_income": applicant_income,
        "loan_amount": loan_amount,
        "credit_history": credit_history,
        "employment_length": employment_length,
        "interest_rate": interest_rate,
        "loan_purpose": loan_purpose,
    }
    try:
        pred, prob = predict_default(payload)
        st.subheader("Result")
        st.write(f"Default probability: **{prob:.2%}**")
        if pred == 1:
            st.error("High credit risk: applicant is likely to default.")
        else:
            st.success("Lower credit risk: applicant is less likely to default.")
    except Exception as exc:
        st.exception(exc)
