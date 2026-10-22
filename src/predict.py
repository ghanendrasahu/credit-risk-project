from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "credit_risk_pipeline.joblib"
META_PATH = ROOT_DIR / "models" / "metadata.json"


def load_model_and_metadata():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run `python src/train.py` first.")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, metadata


def predict_default(input_payload: dict) -> tuple[int, float]:
    model, metadata = load_model_and_metadata()
    threshold = float(metadata.get("threshold", 0.5))
    features = metadata.get("features", [])

    row = {feature: input_payload.get(feature) for feature in features}
    input_df = pd.DataFrame([row])

    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= threshold)
    return prediction, probability
