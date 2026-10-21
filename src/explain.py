"""Explainability for the credit risk model.

Two views, both dependency-light (no SHAP needed):

1. `applicant_sensitivity`  - per-applicant "what-if" report: how the default
   probability responds to +-10% perturbations of every numeric feature.
2. `global_permutation_importance` - permutation importance over a labelled
   dataset, ranked.

The result is evidence for *"why was this applicant flagged"* rather than a
raw score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from predict import load_model_and_metadata  # noqa: E402

DEFAULT_DATA = ROOT_DIR / "data" / "raw" / "loan_data.csv"

CATEGORICAL_HINT = "loan_purpose"


def applicant_sensitivity(payload: dict[str, Any], perturb: float = 0.10) -> dict[str, Any]:
    """Default probability under +-``perturb`` shifts of every numeric input.

    Small deltas mean the feature barely moves the decision; large deltas are
    the levers driving this applicant's score.
    """
    model, metadata = load_model_and_metadata()
    features = metadata.get("features", [])
    threshold = float(metadata.get("threshold", 0.5))

    row = {feature: payload.get(feature) for feature in features}
    frame = pd.DataFrame([row])

    base_probability = float(model.predict_proba(frame)[0][1])
    sensitivities: list[dict[str, Any]] = []

    numeric = [c for c in features if c != CATEGORICAL_HINT and isinstance(payload.get(c), (int, float))]
    frame[numeric] = frame[numeric].astype(float)  # avoid int64 -> float FutureWarnings
    for feature in numeric:
        value = float(payload[feature])
        deltas = {}
        for label, sign in (("up", +1), ("down", -1)):
            nudged = frame.copy()
            nudged.loc[0, feature] = value * (1 + sign * perturb)
            nudged_probability = float(model.predict_proba(nudged)[0][1])
            deltas[label] = round(nudged_probability - base_probability, 4)
        sensitivities.append(
            {
                "feature": feature,
                "value": value,
                "prob_delta_if_up": deltas["up"],
                "prob_delta_if_down": deltas["down"],
                "impact": round(abs(deltas["up"]) + abs(deltas["down"]), 4),
            }
        )

    sensitivities.sort(key=lambda item: item["impact"], reverse=True)
    flagged = base_probability >= threshold
    report = {
        "base_probability": round(base_probability, 4),
        "threshold": threshold,
        "flagged": flagged,
        "verdict": "FLAGGED for review" if flagged else "APPROVED",
        "top_drivers": [
            {
                "feature": s["feature"],
                "value": s["value"],
                "note": f"raising it +10% moves P(default) by {s['prob_delta_if_up']:+.3f}",
            }
            for s in sensitivities[:3]
        ],
        "sensitivities": sensitivities,
    }
    return report


def applicant_verdict(report: dict[str, Any]) -> str:
    lines = [
        f"P(default) = {report['base_probability']:.1%} vs threshold "
        f"{report['threshold']:.1%} -> {report['verdict']}."
    ]
    for driver in report["top_drivers"]:
        lines.append(f"  - {driver['feature']} ({driver['value']:g}): {driver['note']}")
    return "\n".join(lines)


def global_permutation_importance(
    data: str | Path | pd.DataFrame | None = None, n_repeats: int = 8
) -> list[dict[str, float]]:
    """Ranked permutation importance computed MANUALLY (version-proof).

    Scoring: log-loss degradation when a feature's column is randomly
    shuffled - larger positive degradation means the feature matters more.
    Hand-rolled because pipeline wrappers are not always accepted by
    ``sklearn.inspection.permutation_importance`` across versions.
    """
    from sklearn.metrics import log_loss
    from sklearn.model_selection import train_test_split

    if data is None:
        df = pd.read_csv(DEFAULT_DATA)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data)
    target = "default"
    if target not in df.columns:
        raise ValueError("Permutation importance needs the ground-truth 'default' column.")

    model, metadata = load_model_and_metadata()
    features = metadata.get("features", [])

    _, X_eval = train_test_split(df, test_size=0.5, random_state=42)
    y_true = X_eval[target].to_numpy()

    def _score(frame: pd.DataFrame) -> float:
        probas = model.predict_proba(frame[features])
        return float(log_loss(y_true, probas, labels=[0, 1]))

    baseline = _score(X_eval)
    rng = np.random.RandomState(42)

    importances: list[dict[str, float]] = []
    for feature in features:
        degradations: list[float] = []
        for _ in range(n_repeats):
            shuffled = X_eval.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            degradations.append(_score(shuffled) - baseline)
        importances.append(
            {"feature": feature, "importance": round(float(np.mean(degradations)), 5)}
        )

    importances.sort(key=lambda item: item["importance"], reverse=True)
    return importances


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a (or the dataset's) credit decision.")
    parser.add_argument("--applicant", help="JSON dict of feature values, e.g. '{\"applicant_income\": 50000, ...}'")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Labelled CSV for global permutation importance")
    parser.add_argument("--global-importance", action="store_true", help="Run global permutation instead of/next to applicant view")
    args = parser.parse_args()

    if args.applicant:
        payload = json.loads(args.applicant)
        report = applicant_sensitivity(payload)
        print(applicant_verdict(report))
        print(json.dumps(report["top_drivers"], indent=2))
    if args.global_importance or not args.applicant:
        ranked = global_permutation_importance(args.data)
        print("\nGlobal permutation importance (neg-log-loss):")
        for entry in ranked[:8]:
            print(f"  {entry['feature']:<22} {entry['importance']:>10.5f}")


if __name__ == "__main__":
    main()
