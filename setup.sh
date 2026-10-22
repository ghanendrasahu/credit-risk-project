#!/usr/bin/env bash
# Streamlit Cloud entry point.
#
# The dataset (data/raw/loan_data.csv) is committed; model artifacts under
# models/ are build outputs, so they are trained once at boot unless present.
set -euo pipefail
cd "$(dirname "$0")"

pip install -r requirements.txt

if [ ! -f models/credit_risk_pipeline.joblib ]; then
  python src/train.py
fi

exec streamlit run app.py
