"""Tests for the Stage 6 actionable-feature catalog."""

from src.retention.rules import ACTIONABLE_FEATURES, NON_ACTIONABLE_FEATURES


def _feature(name):
    return next(f for f in ACTIONABLE_FEATURES if f.feature == name)


def test_core_actionable_features_present():
    names = {f.feature for f in ACTIONABLE_FEATURES}
    expected = {
        "Contract", "MonthlyCharges", "TechSupport", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "PaymentMethod", "InternetService",
        "PaperlessBilling", "tenure",
    }
    assert expected <= names


def test_protected_features_are_never_actionable():
    actionable_names = {f.feature for f in ACTIONABLE_FEATURES}
    assert {"gender", "SeniorCitizen", "Partner", "Dependents"} <= NON_ACTIONABLE_FEATURES
    assert NON_ACTIONABLE_FEATURES.isdisjoint(actionable_names)


def test_contract_better_values_are_longer_terms():
    assert _feature("Contract").better_values == frozenset({"One year", "Two year"})


def test_numeric_features_have_a_direction():
    assert _feature("MonthlyCharges").direction == "lower"
    assert _feature("tenure").direction == "newer"


def test_every_feature_has_action_and_alternative_label():
    for feature in ACTIONABLE_FEATURES:
        assert feature.action.strip()
        assert feature.alternative_label.strip()
        assert feature.kind in {"categorical", "numeric"}
        assert 0.0 < feature.actionability <= 1.0
