"""Tests for the Stage 6 lift-based retention planner."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DATA_PROCESSED_DIR,
    DECISION_THRESHOLD_PATH,
    MODEL_PATH,
    MODELS_DIR,
    SIMILARITY_MAX_NEIGHBORS,
    SIMILARITY_MIN_THRESHOLD,
)
from src.data.split import split_data
from src.explainability.explainer import build_explainer, explain_customer
from src.features.preprocessor import load_preprocessor
from src.models.train import load_model
from src.retention.planner import (
    DISCLAIMER,
    build_recommendations_lift,
    build_retention_plan,
    confidence_label,
    _weighted_median,
    _weighted_share,
)
from src.similarity.search import find_similar_churned, find_similar_retained
from src.similarity.vectorizer import build_customer_vectors, vectorize_one

REQUIRED_FIELDS = {
    "feature", "action", "current_value", "suggested_alternative", "model_contribution",
    "mode", "retained_rate", "churned_rate", "lift", "support",
    "retained_count", "churned_count", "avg_similarity", "similarity_range",
    "combined_score", "confidence", "insufficient_neighbor_evidence",
}


# --- helper unit tests -----------------------------------------------------

def test_confidence_label_bands():
    assert confidence_label(0.9) == "High"
    assert confidence_label(0.5) == "Medium"
    assert confidence_label(0.1) == "Low"


def test_weighted_share_uses_weights():
    assert _weighted_share(np.array([True, False]), np.array([0.9, 0.1])) == pytest.approx(0.9)


def test_weighted_median_leans_to_heavier_weights():
    assert _weighted_median(np.array([10.0, 50.0]), np.array([0.95, 0.05])) == 10.0


# --- lift logic with controlled retained vs churned groups -----------------

def _customer(**overrides):
    base = {"Contract": "Month-to-month", "TechSupport": "No", "MonthlyCharges": 100.0}
    base.update(overrides)
    return pd.DataFrame([base])


def _explanation(*features):
    return {"increases_risk": [{"feature": f, "contribution": 1.0} for f in features]}


def test_lift_recommendation_fires_when_retained_beats_churned():
    # Retained neighbors mostly on longer contracts; churned mostly month-to-month.
    retained = pd.DataFrame({"Contract": ["Two year", "Two year", "One year", "Two year", "One year"]})
    churned = pd.DataFrame({"Contract": ["Month-to-month"] * 5})
    weights = np.array([0.95, 0.94, 0.93, 0.92, 0.91])

    recs = build_recommendations_lift(
        _explanation("Contract"), _customer(), retained, weights, churned, weights
    )
    assert len(recs) == 1
    rec = recs[0]
    assert rec["feature"] == "Contract"
    assert rec["mode"] == "lift"
    assert rec["retained_rate"] > rec["churned_rate"]
    assert rec["lift"] >= 0.10
    assert set(rec) == REQUIRED_FIELDS


def test_no_recommendation_when_churned_neighbors_share_the_better_value():
    # Both retained AND churned neighbors have longer contracts -> no lift.
    retained = pd.DataFrame({"Contract": ["Two year"] * 5})
    churned = pd.DataFrame({"Contract": ["Two year"] * 5})
    weights = np.array([0.95, 0.94, 0.93, 0.92, 0.91])

    recs = build_recommendations_lift(
        _explanation("Contract"), _customer(), retained, weights, churned, weights
    )
    assert recs == []


def test_lift_requires_model_evidence():
    retained = pd.DataFrame({"Contract": ["Two year"] * 5, "TechSupport": ["Yes"] * 5})
    churned = pd.DataFrame({"Contract": ["Month-to-month"] * 5, "TechSupport": ["No"] * 5})
    weights = np.array([0.95, 0.94, 0.93, 0.92, 0.91])

    # Model only flags TechSupport, so Contract must not be recommended.
    recs = build_recommendations_lift(
        _explanation("TechSupport"), _customer(), retained, weights, churned, weights
    )
    assert {r["feature"] for r in recs} == {"TechSupport"}


def test_recommendations_sorted_by_combined_score():
    retained = pd.DataFrame({"Contract": ["Two year"] * 5, "TechSupport": ["Yes"] * 5})
    churned = pd.DataFrame({"Contract": ["Month-to-month"] * 5, "TechSupport": ["No"] * 5})
    weights = np.array([0.95, 0.94, 0.93, 0.92, 0.91])

    recs = build_recommendations_lift(
        _explanation("Contract", "TechSupport"), _customer(), retained, weights, churned, weights
    )
    scores = [r["combined_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


# --- full plan against the real pipeline -----------------------------------

@pytest.fixture(scope="module")
def pieces():
    customers = pd.read_csv(DATA_PROCESSED_DIR / "telco_clean.csv")
    preprocessor = load_preprocessor(MODELS_DIR / "preprocessor.joblib")
    model = load_model(MODEL_PATH)
    threshold = json.loads(DECISION_THRESHOLD_PATH.read_text())["threshold"]
    explainer = build_explainer(model, split_data(customers).X_train, threshold)
    customer_vectors = build_customer_vectors(preprocessor, customers)
    return preprocessor, explainer, customer_vectors


HIGH_RISK = pd.DataFrame(
    [
        {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
        }
    ]
)


def _plan(pieces, customer):
    preprocessor, explainer, customer_vectors = pieces
    vector = vectorize_one(preprocessor, customer)
    retained = find_similar_retained(
        customer_vectors, vector, top_k=SIMILARITY_MAX_NEIGHBORS, min_similarity=SIMILARITY_MIN_THRESHOLD
    )
    churned = find_similar_churned(
        customer_vectors, vector, top_k=SIMILARITY_MAX_NEIGHBORS, min_similarity=SIMILARITY_MIN_THRESHOLD
    )
    explanation = explain_customer(explainer, customer)
    return build_retention_plan(explanation, customer, retained, churned, customer_vectors), explanation


def test_plan_structure_and_mode(pieces):
    plan, _ = _plan(pieces, HIGH_RISK)
    assert set(plan) == {
        "probability", "risk_level", "outreach_priority", "retained_count",
        "churned_count", "evidence_mode", "recommendations", "similar_customers", "disclaimer",
    }
    assert plan["disclaimer"] == DISCLAIMER
    assert plan["evidence_mode"] == "lift"
    assert plan["retained_count"] == len(plan["similar_customers"])


def test_every_recommendation_is_model_flagged_and_actionable(pieces):
    plan, explanation = _plan(pieces, HIGH_RISK)
    increasing = {f["feature"] for f in explanation["increases_risk"]}
    protected = {"gender", "SeniorCitizen", "Partner", "Dependents"}
    for rec in plan["recommendations"]:
        assert rec["feature"] in increasing
        assert rec["feature"] not in protected
        if rec["mode"] == "lift":
            assert rec["lift"] >= 0.10
            assert rec["support"] >= 0.30


def test_fallback_when_no_qualified_neighbors(pieces):
    _, explainer, customer_vectors = pieces
    explanation = explain_customer(explainer, HIGH_RISK)
    empty = pd.DataFrame({"row_index": [], "customer_id": [], "similarity": []})

    plan = build_retention_plan(explanation, HIGH_RISK, empty, empty, customer_vectors)
    assert plan["evidence_mode"] == "model_only"
    assert plan["retained_count"] == 0
    for rec in plan["recommendations"]:
        assert rec["insufficient_neighbor_evidence"] is True
