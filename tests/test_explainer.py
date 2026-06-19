"""Tests for Stage 4 explainability."""

import json

import pytest

from src.config import DATA_PROCESSED_DIR, DECISION_THRESHOLD_PATH, MODEL_PATH
from src.data.split import FEATURE_COLUMNS, split_data
from src.explainability import explainer as explain
from src.models.train import load_model

import pandas as pd


@pytest.fixture(scope="module")
def model():
    return load_model(MODEL_PATH)


@pytest.fixture(scope="module")
def split():
    return split_data(pd.read_csv(DATA_PROCESSED_DIR / "telco_clean.csv"))


@pytest.fixture(scope="module")
def built_explainer(model, split):
    threshold = json.loads(DECISION_THRESHOLD_PATH.read_text())["threshold"]
    return explain.build_explainer(model, split.X_train, threshold)


def test_original_feature_mapping():
    assert explain._original_feature("numeric__tenure") == "tenure"
    assert explain._original_feature("categorical__Contract_Two year") == "Contract"
    assert explain._original_feature("categorical__PaymentMethod_Mailed check") == "PaymentMethod"


def test_risk_level_bands():
    # medium_cut 0.243, high_cut 0.5
    assert explain.risk_level(0.10, 0.243) == "Low"
    assert explain.risk_level(0.30, 0.243) == "Medium"
    assert explain.risk_level(0.80, 0.243) == "High"


def test_global_importance_covers_all_encoded_features(model):
    importance = explain.global_importance(model)
    n_encoded = len(model.named_steps["preprocess"].get_feature_names_out())
    assert len(importance) == n_encoded
    # Sorted by absolute coefficient, strongest first.
    assert importance["abs_coefficient"].is_monotonic_decreasing


def test_permutation_importance_lists_original_columns(model, split):
    scores = explain.permutation_importance_scores(model, split.X_val, split.y_val, n_repeats=3)
    assert set(scores["feature"]) == set(FEATURE_COLUMNS)


def test_explain_customer_structure_and_sorting(built_explainer, split):
    customer = split.X_val.iloc[[0]]
    result = explain.explain_customer(built_explainer, customer)

    assert set(result) == {"probability", "risk_level", "increases_risk", "decreases_risk"}
    assert 0.0 <= result["probability"] <= 1.0

    # Increasing factors are positive and sorted strongest first.
    increases = [f["contribution"] for f in result["increases_risk"]]
    assert all(c > 0 for c in increases)
    assert increases == sorted(increases, reverse=True)

    # Decreasing factors are negative and sorted strongest first.
    decreases = [f["contribution"] for f in result["decreases_risk"]]
    assert all(c < 0 for c in decreases)
    assert decreases == sorted(decreases)


def test_explain_customer_hides_near_zero_factors(built_explainer, split):
    customer = split.X_val.iloc[[0]]
    big_threshold = explain.explain_customer(built_explainer, customer, min_contribution=10.0)
    # With an unreachably large threshold, no factor should survive.
    assert big_threshold["increases_risk"] == []
    assert big_threshold["decreases_risk"] == []


def test_explain_customer_factors_use_original_columns(built_explainer, split):
    customer = split.X_val.iloc[[0]]
    result = explain.explain_customer(built_explainer, customer)
    reported = {f["feature"] for f in result["increases_risk"] + result["decreases_risk"]}
    assert reported <= set(FEATURE_COLUMNS)
