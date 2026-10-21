"""Scenario stress testing for the credit risk model.

Answers questions like *"What happens to default probability if interest
rates increase 2%"* by applying defined shocks to a portfolio and re-scoring
the persisted pipeline. Shocks are feature-level projections (no retraining);
results are deltas against the unshocked baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from predict import load_model_and_metadata  # noqa: E402

DEFAULT_DATA = ROOT_DIR / "data" / "raw" / "loan_data.csv"

# Shock presets: feature -> [operation, operand].
#   "pct"  = add percentage points (interest_rate 8.7 -> 10.7 with ["pct", 2.0])
#   "add"  = add an absolute amount
#   "mul"  = multiply by a factor
PRESETS: dict[str, dict[str, list]] = {
    "rates_plus_2": {"interest_rate": ["pct", 2.0]},
    "rates_plus_4": {"interest_rate": ["pct", 4.0]},
    "income_minus_10": {"applicant_income": ["mul", 0.90]},
    "income_minus_25": {"applicant_income": ["mul", 0.75]},
    "leverage_plus_10": {"loan_amount": ["mul", 1.10]},
    "severe_recession": {
        "applicant_income": ["mul", 0.75],
        "interest_rate": ["pct", 4.0],
        "loan_amount": ["mul", 1.10],
    },
}


def apply_shocks(df: pd.DataFrame, shocks: dict[str, list]) -> tuple[pd.DataFrame, list[str]]:
    """Return a shocked copy of ``df`` plus the descriptions actually applied."""
    shocked = df.copy()
    applied: list[str] = []
    for column, spec in shocks.items():
        op, value = str(spec[0]), float(spec[1])
        if column not in df.columns:
            continue
        if op == "add":
            shocked[column] = df[column] + value
        elif op == "mul":
            shocked[column] = df[column] * value
        elif op == "pct":
            shocked[column] = df[column] + value  # percentage points
        else:
            raise ValueError(f"Unknown shock operation: {op}")
        applied.append(f"{column} {op}{value:+g}")
    return shocked, applied


def _scores(model: Any, features: list[str], frame: pd.DataFrame) -> pd.Series:
    proba = model.predict_proba(frame[features])[:, 1]
    return pd.Series(proba, index=frame.index)


def run_stress(
    data: str | Path | pd.DataFrame | None = None,
    scenarios: list[str] | None = None,
    max_rows: int = 100_000,
) -> dict[str, Any]:
    """Score the persisted model against the baseline and every scenario."""
    if data is None:
        df = pd.read_csv(DEFAULT_DATA)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data)
    if len(df) > max_rows:
        df = df.head(max_rows)

    model, metadata = load_model_and_metadata()
    features = metadata.get("features", [])
    threshold = float(metadata.get("threshold", 0.5))

    base = _scores(model, features, df)
    baseline = {
        "mean_default_probability": round(float(base.mean()), 4),
        "flagged_above_threshold": int((base >= threshold).sum()),
        "share_flagged": round(float((base >= threshold).mean()), 4),
    }

    names = scenarios or list(PRESETS)
    results: list[dict[str, Any]] = []
    for name in names:
        if name not in PRESETS:
            raise ValueError(f"Unknown scenario '{name}'. Available: {sorted(PRESETS)}")
        shocked, applied = apply_shocks(df, PRESETS[name])
        shocked_scores = _scores(model, features, shocked)
        results.append(
            {
                "scenario": name,
                "shocks_applied": applied,
                "mean_default_probability": round(float(shocked_scores.mean()), 4),
                "uplift_pp": round(float(shocked_scores.mean() - base.mean()) * 100, 2),
                "flagged_above_threshold": int((shocked_scores >= threshold).sum()),
                "additional_flagged": int(((shocked_scores >= threshold) & (base < threshold)).sum()),
            }
        )

    return {
        "n_rows": int(len(df)),
        "threshold": threshold,
        "baseline": baseline,
        "scenarios": results,
    }


def narrative(result: dict[str, Any]) -> str:
    """Human-readable answer to the flagship what-if question."""
    rows = [f"Baseline: {result['baseline']['mean_default_probability']:.1%} average default probability."]
    for scenario in result["scenarios"]:
        rows.append(
            f"{scenario['scenario']}: mean default probability moves to "
            f"{scenario['mean_default_probability']:.1%} "
            f"({scenario['uplift_pp']:+.2f} pp), {scenario['additional_flagged']} "
            f"additional applicant(s) cross the review threshold."
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test the credit risk model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV to stress (defaults to the bundled dataset)")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Subset of scenario names (defaults to all)")
    parser.add_argument("--out", default=str(ROOT_DIR / "models" / "stress_report.json"), help="Where to write the JSON report")
    args = parser.parse_args()

    report = run_stress(args.data, args.scenarios)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(narrative(report))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
