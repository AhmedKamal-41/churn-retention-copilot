"""Tests for Stage 3 evaluation and threshold selection."""

import numpy as np
import pytest

from src.models.evaluate import (
    evaluate_predictions,
    precision_at_target_recall,
    threshold_for_target_recall,
)


@pytest.fixture
def simple_scores():
    # Ten examples where higher scores mostly mean churn (label 1).
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9])
    return y_true, y_score


def test_evaluate_predictions_returns_expected_keys(simple_scores):
    y_true, y_score = simple_scores
    metrics = evaluate_predictions(y_true, y_score, threshold=0.5)
    assert set(metrics) == {"threshold", "precision", "recall", "f1", "roc_auc", "pr_auc"}


def test_perfect_scores_give_perfect_auc():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_predictions(y_true, y_score, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_threshold_for_target_recall_meets_recall(simple_scores):
    y_true, y_score = simple_scores
    threshold = threshold_for_target_recall(y_true, y_score, target_recall=0.8)
    achieved_recall = evaluate_predictions(y_true, y_score, threshold)["recall"]
    assert achieved_recall >= 0.8


def test_lower_target_recall_allows_higher_threshold(simple_scores):
    y_true, y_score = simple_scores
    low = threshold_for_target_recall(y_true, y_score, target_recall=0.4)
    high = threshold_for_target_recall(y_true, y_score, target_recall=1.0)
    # Demanding more recall should never require a higher threshold.
    assert low >= high


def test_precision_at_target_recall_is_a_valid_proportion(simple_scores):
    y_true, y_score = simple_scores
    precision = precision_at_target_recall(y_true, y_score, target_recall=0.8)
    assert 0.0 <= precision <= 1.0
