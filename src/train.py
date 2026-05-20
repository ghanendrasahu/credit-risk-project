from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "loan_data.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "credit_risk_pipeline.joblib"
META_PATH = MODEL_DIR / "metadata.json"

TARGET = "default"


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        "applicant_income",
        "loan_amount",
        "credit_history",
        "employment_length",
        "interest_rate",
    ]
    categorical_features = ["loan_purpose"]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def find_threshold_for_recall(y_true: pd.Series, proba: pd.Series, min_recall: float = 0.8) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    threshold = 0.5
    for idx, recall in enumerate(recalls[:-1]):
        if recall >= min_recall:
            threshold = float(thresholds[idx])
    return threshold


def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(df)
    candidate_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        ),
    }

    best_name = None
    best_pipeline = None
    best_recall = -1.0
    best_proba = None

    for name, model in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        recall = recall_score(y_test, y_pred, zero_division=0)

        if recall > best_recall:
            best_recall = recall
            best_name = name
            best_pipeline = pipeline
            best_proba = pipeline.predict_proba(X_test)[:, 1]

    if best_pipeline is None or best_proba is None:
        raise RuntimeError("No model could be trained.")

    threshold = find_threshold_for_recall(y_test, best_proba, min_recall=0.8)
    y_pred_adjusted = (best_proba >= threshold).astype(int)
    report = classification_report(y_test, y_pred_adjusted, output_dict=True, zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    metadata = {
        "target": TARGET,
        "model_name": best_name,
        "threshold": threshold,
        "features": list(X.columns),
        "metrics": {
            "test_recall_default_class": report.get("1", {}).get("recall", 0.0),
            "test_precision_default_class": report.get("1", {}).get("precision", 0.0),
            "test_f1_default_class": report.get("1", {}).get("f1-score", 0.0),
            "test_accuracy": report.get("accuracy", 0.0),
        },
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {META_PATH}")
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    train()
