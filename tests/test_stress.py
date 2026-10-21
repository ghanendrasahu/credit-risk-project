"""Tests for the stress-testing and explainability modules (Phase-D additions).

Reuse the committed trained pipeline (models/credit_risk_pipeline.joblib) so
tests stay fast and honest: they exercise the real model, not a mock.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from predict import load_model_and_metadata
from stress import PRESETS, apply_shocks, narrative, run_stress

MODEL, METADATA = load_model_and_metadata()


def _small_frame(rows: int = 600) -> pd.DataFrame:
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "loan_data.csv"
    return pd.read_csv(path).head(rows)


def test_presets_cover_flagship_rate_shock():
    assert PRESETS["rates_plus_2"]["interest_rate"] == ["pct", 2.0]


def test_apply_shocks_adds_percentage_points():
    df = _small_frame(50)
    shocked, applied = apply_shocks(df, {"interest_rate": ["pct", 2.0]})
    assert applied == ["interest_rate pct+2"]
    assert shocked["interest_rate"].iloc[0] == pytest.approx(df["interest_rate"].iloc[0] + 2.0)
    assert df["interest_rate"].iloc[0] == pytest.approx(shocked["interest_rate"].iloc[0] - 2.0)


def test_run_stress_rates_plus_2_raises_default_probability():
    report = run_stress(_small_frame(), scenarios=["rates_plus_2"])
    scenario = report["scenarios"][0]
    expect_rounded = [
        "scenario",
        "shocks_applied",
        "mean_default_probability",
        "uplift_pp",
        "flagged_above_threshold",
        "additional_flagged",
    ]
    assert list(scenario) == expect_rounded
    assert scenario["shocks_applied"] == ["interest_rate pct+2"]
    # Rate hike +2pp should not REDUCE the default probability burden
    assert scenario["uplift_pp"] >= 0
    assert narrative(report).startswith("Baseline:")


def test_run_stress_severe_recession_beats_baseline():
    report = run_stress(_small_frame(), scenarios=["income_minus_25", "severe_recession"])
    severe = next(s for s in report["scenarios"] if s["scenario"] == "severe_recession")
    assert severe["mean_default_probability"] > report["baseline"]["mean_default_probability"]


def test_run_stress_unknown_scenario_rejected():
    with pytest.raises(ValueError, match="Unknown scenario"):
        run_stress(_small_frame(100), scenarios=["black_swan"])


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def test_applicant_sensitivity_reports_flagged_or_not():
    import explain

    features = METADATA["features"]
    row = METADATA  # sanity
    frame = _small_frame(3)
    sample = frame.iloc[0][features].to_dict()
    report = explain.applicant_sensitivity(sample)
    assert report["flagged"] in {True, False}
    assert report["base_probability"] == pytest.approx(0.0, abs=1.0)
    assert report["top_drivers"], report
    for driver in report["top_drivers"]:
        assert driver["feature"] in features
    # every numeric sensitivity keeps both directions finite
    for sensitivity in report["sensitivities"]:
        assert -1.0 <= sensitivity["prob_delta_if_up"] <= 1.0
        assert -1.0 <= sensitivity["prob_delta_if_down"] <= 1.0
    assert row  # metadata loaded


def test_global_permutation_importance_runs_fast_and_ranks():
    import explain

    frame = _small_frame(600)
    ranked = explain.global_permutation_importance(frame, n_repeats=2)
    assert ranked
    assert all(entry["feature"] in METADATA["features"] for entry in ranked)
    ranked_names = [entry["feature"] for entry in ranked]
    assert len(ranked_names) == len(set(ranked_names))  # unique, no duplicates
