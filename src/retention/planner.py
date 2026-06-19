"""Build a similarity-driven retention plan for one customer (Stage 6).

Workflow:
    model predicts churn -> SHAP gives the risk factors -> cosine similarity finds
    comparable customers who STAYED and comparable customers who CHURNED -> for each
    actionable feature, compare how common the better value is among retained vs
    churned neighbors (a lift) -> generate and prioritise retention actions.

A feature becomes a recommendation when the model flags it as raising the
customer's risk AND the better value is meaningfully more common among similar
retained customers than among similar churned customers (a minimum lift and
support, not just a 60% majority).

Fallbacks, in order: if there are too few churned neighbors to compute a lift, use
the simpler "common among retained neighbors" rule; if there are no retained
neighbors at all, use model-supported actions and label them as such. Nothing here
claims an action caused anyone to stay.
"""

import numpy as np
import pandas as pd

from src.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    LIFT_PRIORITY_WEIGHTS,
    MIN_LIFT,
    MIN_NEIGHBORS_PER_GROUP,
    MIN_SUPPORT,
    MIN_WEIGHTED_AGREEMENT,
    MODEL_CONTRIBUTION_SCALE,
    PRIORITY_WEIGHTS,
)
from src.retention.rules import ACTIONABLE_FEATURES
from src.similarity.vectorizer import CustomerVectors

DISCLAIMER = (
    "Personalized recommendations based on model risk factors and patterns among "
    "similar retained customers. The dataset does not record which offers were "
    "applied, so these are not proof that an action caused any customer to stay."
)

# Recommendations weaker than this combined score are treated as noise and hidden.
_MIN_SCORE = 0.05


def confidence_label(score: float) -> str:
    """Map a combined priority score to High / Medium / Low confidence."""
    if score >= CONFIDENCE_HIGH:
        return "High"
    if score >= CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"


def _model_strength(contribution: float) -> float:
    return min(contribution / MODEL_CONTRIBUTION_SCALE, 1.0)


def _weighted_share(mask: np.ndarray, weights: np.ndarray) -> float:
    """Similarity-weighted fraction of neighbors where mask is True."""
    total = weights.sum()
    return float(weights[mask].sum() / total) if total > 0 else 0.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Similarity-weighted median of numeric neighbor values."""
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def _supportive_mask(feature, values: np.ndarray, customer_value) -> np.ndarray:
    """Which neighbor values are the "better" option for this feature.

    Categorical: the value is one of the favorable values.
    Numeric "lower" (charges): the neighbor pays less than the customer.
    Numeric "newer" (tenure): the neighbor has been around longer than the customer.
    """
    if feature.kind == "categorical":
        return np.array([value in feature.better_values for value in values])
    numbers = values.astype(float)
    if feature.direction == "lower":
        return numbers < float(customer_value)
    return numbers > float(customer_value)


def _suggested_alternative(feature, retained_values: np.ndarray, weights: np.ndarray, customer_value) -> str:
    if feature.kind == "categorical":
        weight_by_value = {}
        for value, weight in zip(retained_values, weights):
            if value in feature.better_values:
                weight_by_value[value] = weight_by_value.get(value, 0.0) + weight
        return max(weight_by_value, key=weight_by_value.get) if weight_by_value else feature.alternative_label
    median = _weighted_median(retained_values.astype(float), weights)
    if feature.direction == "lower":
        return f"around ${median:.0f}/month"
    return f"onboarding and loyalty support (similar stayers' median tenure: {median:.0f} months)"


def _make_recommendation(
    feature, current_value, suggested, contribution, score, *, mode,
    retained_rate=None, churned_rate=None, lift=None, support=None,
    retained_count=0, churned_count=0, avg_similarity=0.0,
    similarity_range=(0.0, 0.0), insufficient=False,
) -> dict:
    """Build one recommendation with a consistent set of fields for every mode."""
    return {
        "feature": feature.feature,
        "action": feature.action,
        "current_value": current_value,
        "suggested_alternative": suggested,
        "model_contribution": float(contribution),
        "mode": mode,
        "retained_rate": retained_rate,
        "churned_rate": churned_rate,
        "lift": lift,
        "support": support,
        "retained_count": int(retained_count),
        "churned_count": int(churned_count),
        "avg_similarity": float(avg_similarity),
        "similarity_range": similarity_range,
        "combined_score": float(score),
        "confidence": confidence_label(score),
        "insufficient_neighbor_evidence": insufficient,
    }


def _lift_recommendation(feature, customer_value, retained_values, retained_weights, churned_values, churned_weights, contribution):
    """Primary path: recommend when the better value is more common among retained
    than churned neighbors, by at least MIN_LIFT, with at least MIN_SUPPORT."""
    if feature.kind == "categorical" and customer_value in feature.better_values:
        return None

    retained_mask = _supportive_mask(feature, retained_values, customer_value)
    retained_rate = _weighted_share(retained_mask, retained_weights)
    churned_rate = _weighted_share(_supportive_mask(feature, churned_values, customer_value), churned_weights)
    lift = retained_rate - churned_rate
    if lift < MIN_LIFT or retained_rate < MIN_SUPPORT:
        return None

    supporting_weights = retained_weights[retained_mask]
    similarity_quality = float(supporting_weights.mean()) if supporting_weights.size else 0.0
    similarity_range = (
        (float(supporting_weights.min()), float(supporting_weights.max()))
        if supporting_weights.size else (0.0, 0.0)
    )
    suggested = _suggested_alternative(feature, retained_values, retained_weights, customer_value)

    weights = LIFT_PRIORITY_WEIGHTS
    score = (
        weights["model"] * _model_strength(contribution)
        + weights["lift"] * lift
        + weights["similarity"] * similarity_quality
        + weights["support"] * retained_rate
    )
    return _make_recommendation(
        feature, customer_value, suggested, contribution, score, mode="lift",
        retained_rate=retained_rate, churned_rate=churned_rate, lift=lift, support=retained_rate,
        retained_count=len(retained_weights), churned_count=len(churned_weights),
        avg_similarity=similarity_quality, similarity_range=similarity_range,
    )


def build_recommendations_lift(explanation, customer_row, retained_features, retained_weights, churned_features, churned_weights):
    """Lift-based recommendations, strongest first."""
    increasing = {factor["feature"]: factor["contribution"] for factor in explanation["increases_risk"]}
    customer = customer_row.iloc[0]

    recommendations = []
    for feature in ACTIONABLE_FEATURES:
        if feature.feature not in increasing:
            continue
        recommendation = _lift_recommendation(
            feature,
            customer[feature.feature],
            retained_features[feature.feature].to_numpy(),
            retained_weights,
            churned_features[feature.feature].to_numpy(),
            churned_weights,
            increasing[feature.feature],
        )
        if recommendation is not None and recommendation["combined_score"] >= _MIN_SCORE:
            recommendations.append(recommendation)

    recommendations.sort(key=lambda item: item["combined_score"], reverse=True)
    return recommendations


def build_recommendations_majority(explanation, customer_row, retained_features, weights):
    """Fallback when too few churned neighbors exist to compute a lift: recommend a
    better value that is common (>= 60% weighted) among retained neighbors."""
    increasing = {factor["feature"]: factor["contribution"] for factor in explanation["increases_risk"]}
    customer = customer_row.iloc[0]

    recommendations = []
    for feature in ACTIONABLE_FEATURES:
        if feature.feature not in increasing:
            continue
        current_value = customer[feature.feature]
        if feature.kind == "categorical" and current_value in feature.better_values:
            continue

        values = retained_features[feature.feature].to_numpy()
        support_mask = _supportive_mask(feature, values, current_value)
        agreement = _weighted_share(support_mask, weights)
        if agreement < MIN_WEIGHTED_AGREEMENT:
            continue

        supporting_weights = weights[support_mask]
        similarity_quality = float(supporting_weights.mean()) if supporting_weights.size else 0.0
        similarity_range = (
            (float(supporting_weights.min()), float(supporting_weights.max()))
            if supporting_weights.size else (0.0, 0.0)
        )
        suggested = _suggested_alternative(feature, values, weights, current_value)

        weight = PRIORITY_WEIGHTS
        score = (
            weight["model"] * _model_strength(increasing[feature.feature])
            + weight["agreement"] * agreement
            + weight["similarity"] * similarity_quality
            + weight["actionability"] * feature.actionability
        )
        if score < _MIN_SCORE:
            continue
        recommendations.append(
            _make_recommendation(
                feature, current_value, suggested, increasing[feature.feature], score, mode="majority",
                retained_rate=agreement, support=agreement, retained_count=len(weights),
                avg_similarity=similarity_quality, similarity_range=similarity_range,
            )
        )

    recommendations.sort(key=lambda item: item["combined_score"], reverse=True)
    return recommendations


def build_recommendations_model_only(explanation, customer_row):
    """Last-resort fallback when no retained neighbors qualify."""
    increasing = {factor["feature"]: factor["contribution"] for factor in explanation["increases_risk"]}
    customer = customer_row.iloc[0]
    actionable = {feature.feature: feature for feature in ACTIONABLE_FEATURES}

    recommendations = []
    for feature_name, contribution in increasing.items():
        feature = actionable.get(feature_name)
        if feature is None:
            continue
        current_value = customer[feature_name]
        if feature.kind == "categorical" and current_value in feature.better_values:
            continue
        score = PRIORITY_WEIGHTS["model"] * _model_strength(contribution)
        if score < _MIN_SCORE:
            continue
        recommendations.append(
            _make_recommendation(
                feature, current_value, feature.alternative_label, contribution, score,
                mode="model_only", insufficient=True,
            )
        )

    recommendations.sort(key=lambda item: item["combined_score"], reverse=True)
    return recommendations


def build_retention_plan(explanation, customer_row, retained_neighbors, churned_neighbors, customer_vectors: CustomerVectors):
    """Assemble the full retention plan, choosing the best available evidence mode.

    retained_neighbors and churned_neighbors are the similar stayers and churners
    (Stage 5), each already filtered to the similarity threshold and weighted by
    cosine score.
    """
    retained_count = len(retained_neighbors)
    churned_count = len(churned_neighbors)

    if retained_count >= MIN_NEIGHBORS_PER_GROUP and churned_count >= MIN_NEIGHBORS_PER_GROUP:
        retained_features = customer_vectors.raw_features.iloc[retained_neighbors["row_index"].to_numpy()].reset_index(drop=True)
        churned_features = customer_vectors.raw_features.iloc[churned_neighbors["row_index"].to_numpy()].reset_index(drop=True)
        recommendations = build_recommendations_lift(
            explanation, customer_row,
            retained_features, retained_neighbors["similarity"].to_numpy(dtype=float),
            churned_features, churned_neighbors["similarity"].to_numpy(dtype=float),
        )
        evidence_mode = "lift"
    elif retained_count >= 1:
        retained_features = customer_vectors.raw_features.iloc[retained_neighbors["row_index"].to_numpy()].reset_index(drop=True)
        recommendations = build_recommendations_majority(
            explanation, customer_row, retained_features, retained_neighbors["similarity"].to_numpy(dtype=float)
        )
        evidence_mode = "majority"
    else:
        recommendations = build_recommendations_model_only(explanation, customer_row)
        evidence_mode = "model_only"

    similar_customers = [
        {"customer_id": customer_id, "similarity": float(similarity)}
        for customer_id, similarity in zip(retained_neighbors["customer_id"], retained_neighbors["similarity"])
    ]

    return {
        "probability": explanation["probability"],
        "risk_level": explanation["risk_level"],
        "outreach_priority": explanation["risk_level"],
        "retained_count": int(retained_count),
        "churned_count": int(churned_count),
        "evidence_mode": evidence_mode,
        "recommendations": recommendations,
        "similar_customers": similar_customers,
        "disclaimer": DISCLAIMER,
    }
