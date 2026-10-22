# Credit Risk & Loan Default Prediction

Binary classification of loan default, benchmarked across two model families
(Logistic Regression and Random Forest) with a **threshold tuned for recall**
on high-risk defaults, plus two things that turn a notebook into a risk
engine:

- **Scenario stress testing** — "what happens to default probability if
  interest rates increase 2%?"
- **Per-applicant explainability** — why was *this* applicant flagged?

## Stress testing

```bash
python src/stress.py --scenarios rates_plus_2 income_minus_10
```

```
Baseline: 48.2% average default probability.
rates_plus_2: mean default probability moves to 50.6% (+2.37 pp),
43 additional applicant(s) cross the review threshold.
```

Presets cover rate shocks (`+2`, `+4` points), income shocks (`-10%`,
`-25%`), leverage (`loan_amount +10%`) and a combined `severe_recession`.
The report is written to `models/stress_report.json`.

## Explainability

```bash
python src/explain.py --applicant '{"applicant_income": 45250, "loan_amount": 150000,
  "credit_history": 1, "employment_length": 4, "interest_rate": 11.2, "loan_purpose": "home"}'
```

```
P(default) = 47.3% vs threshold 44.3% -> FLAGGED for review.
  - interest_rate (11.2): raising it +10% moves P(default) by +0.014
  - credit_history (1): raising it +10% moves P(default) by +0.005
```

`--global-importance` prints the ranked feature importance for the whole
portfolio (permutation-based, model-agnostic, and stable across scikit-learn
versions).

## Layout

```text
src/train.py     trains both families, tunes the threshold, writes the model
src/predict.py   loads the pipeline and scores a single applicant
src/stress.py    scenario stress testing (CLI)
src/explain.py   applicant sensitivity + permutation importance (CLI)
models/          committed pipeline + metadata (threshold, features, metrics)
data/raw/        the loan dataset
tests/           stress-testing and explainability tests
app.py           Streamlit model-serving app
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python src/train.py                 # retrain -> models/
python src/stress.py                # scenario report
streamlit run app.py                # dashboard
```

## Notes

- The model is trained on `data/raw/loan_data.csv`; `models/metadata.json`
  records the feature list, the decision threshold and the evaluation
  metrics, and is what the stress and explain modules read.
- Requirements are pinned to the scikit-learn line the committed model was
  trained with, so the pipeline unpickles without version drift.
