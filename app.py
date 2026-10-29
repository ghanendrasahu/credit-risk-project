"""Credit risk assessment UI.

A single-screen Streamlit app: applicant details on the left, the scored result
on the right, and the model card tucked behind an expander so the interface
speaks in plain language first.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import ui
from src.predict import predict_default

ROOT_DIR = Path(__file__).resolve().parent
META_PATH = ROOT_DIR / "models" / "metadata.json"

ui.setup("Credit Risk", icon=ui.app_icon(ROOT_DIR / "assets"))

metadata: dict = {}
if META_PATH.exists():
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
metrics = metadata.get("metrics", {})
threshold = float(metadata.get("threshold", 0.5))

model_ready = bool(metadata)
ui.page_header(
    "Credit Risk",
    "Estimate the probability that a loan applicant defaults, and see the factors that move the score.",
    eyebrow="Credit Risk Lab",
    badge="Model ready" if model_ready else "Model unavailable",
    badge_tone="ok" if model_ready else "bad",
)

form_col, result_col = st.columns([1.4, 1], gap="large")

result: tuple[int, float] | None = None
failure: str | None = None

with form_col:
    ui.section(
        "Applicant details",
        "Prefilled with a representative application. Adjust any field to test a scenario.",
    )
    with st.form("prediction_form"):
        applicant_income = st.number_input(
            "Annual income (₹)",
            min_value=0,
            value=50000,
            step=1000,
            help="Gross annual income before tax.",
        )
        loan_amount = st.number_input(
            "Loan amount (₹)",
            min_value=0,
            value=200000,
            step=1000,
            help="Total amount requested.",
        )
        credit_history_label = st.selectbox(
            "Credit history",
            options=["Good", "Poor"],
            help="Good means no missed payments in the last two years.",
        )
        employment_length = st.number_input(
            "Years employed",
            min_value=0,
            max_value=60,
            value=5,
            step=1,
        )
        interest_rate = st.number_input(
            "Interest rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=12.0,
            step=0.1,
            help="Annual rate offered on the loan.",
        )
        loan_purpose = st.selectbox(
            "Loan purpose",
            options=["home", "car", "education", "personal", "business"],
            format_func=str.title,
        )
        submitted = st.form_submit_button(
            "Assess risk",
            type="primary",
            width="stretch",
        )

if submitted:
    payload = {
        "applicant_income": applicant_income,
        "loan_amount": loan_amount,
        "credit_history": 1 if credit_history_label == "Good" else 0,
        "employment_length": employment_length,
        "interest_rate": interest_rate,
        "loan_purpose": loan_purpose,
    }
    try:
        result = predict_default(payload)
    except Exception as exc:  # surfaced in the result column
        failure = str(exc)

with result_col:
    if failure:
        ui.error_state(
            f"The scoring service could not score this application: {failure}",
            title="Scoring failed",
            hint="Check the model artifacts and try again.",
        )
    elif result is not None:
        prediction, probability = result
        high_risk = prediction == 1
        tone = ui.DANGER if high_risk else ui.SUCCESS
        band = "High risk" if high_risk else "Lower risk"
        narrative = (
            "Above the review threshold. A credit analyst should review this application "
            "before any approval decision."
            if high_risk
            else "Below the review threshold. The application looks consistent with the "
            "lower-risk population."
        )
        badge_tone = "bad" if high_risk else "ok"
        bar_width = max(3.0, min(probability * 100, 100.0))
        st.markdown(
            f'<div class="ui-result">'
            f'<div class="ui-result-head"><span class="ui-result-label">Estimated default probability</span>'
            f"{ui.badge(band, badge_tone)}</div>"
            f'<div class="ui-result-value" style="color:{tone}">{probability:.1%}</div>'
            f'<div class="ui-meter"><span style="width:{bar_width:.1f}%;background:{tone}"></span></div>'
            f'<div class="ui-result-note">{narrative}</div>'
            f'<dl style="margin:1.1rem 0 0">'
            f'<div class="ui-kv"><dt>Review threshold</dt><dd>{threshold:.0%}</dd></div>'
            f'<div class="ui-kv"><dt>Loan amount</dt><dd>₹{loan_amount:,}</dd></div>'
            f'<div class="ui-kv"><dt>Interest rate</dt><dd>{interest_rate:.2f}%</dd></div>'
            f'<div class="ui-kv"><dt>Credit history</dt><dd>{credit_history_label}</dd></div>'
            f"</dl></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "This is a statistical estimate, not a lending decision. "
            "Thresholds are tuned for recall on defaulters."
        )
    else:
        ui.empty_state(
            "No assessment yet",
            "Fill in the applicant details and select Assess risk. The probability, "
            "risk band and review threshold will appear here.",
            icon_name="shield",
        )

    with st.expander("About this model"):
        st.markdown(
            "**Logistic regression** trained on historical loan applications. "
            "The decision threshold is tuned to catch defaulters, so it trades precision for recall."
        )
        st.markdown(
            f"- Recall on defaulters: **{float(metrics.get('test_recall_default_class', 0)):.2f}**\n"
            f"- Precision on defaulters: **{float(metrics.get('test_precision_default_class', 0)):.2f}**\n"
            f"- F1 on defaulters: **{float(metrics.get('test_f1_default_class', 0)):.2f}**\n"
            f"- Decision threshold: **{threshold:.2f}**"
        )
        st.caption(
            "Inputs: annual income, loan amount, credit history, years employed, "
            "interest rate and loan purpose."
        )
