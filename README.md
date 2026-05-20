
# Credit Risk / Loan Default Prediction

## Objective
Predict whether a loan applicant will default on a loan so lenders can reduce financial risk.

## ML Task
Binary classification (`default` = 1 means likely to default).

## Key Metric
Recall for defaulters (class `1`), with threshold tuning to prioritize catching risky cases.

## Project Structure
- `data/raw/loan_data.csv`: input dataset
- `src/train.py`: training pipeline + model selection + threshold tuning
- `src/predict.py`: inference helper for single prediction
- `models/`: saved model and metadata after training
- `app.py`: Streamlit UI for manual risk prediction

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`

## Train Model
Run:
`python src/train.py`

This creates:
- `models/credit_risk_pipeline.joblib`
- `models/metadata.json`

## Launch App
Run:
`streamlit run app.py`

Then open the local URL shown in your terminal and enter applicant details to get:
- default probability
- risk category based on tuned threshold
