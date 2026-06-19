"""Evaluate churn models and choose a decision threshold (Stage 3).

The default 0.5 cut-off is not what we want: this is a recall-leaning problem,
so we pick the threshold that catches a target share of churners and then report
the precision and outreach volume that come with it.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

sns.set_theme(style="whitegrid")


def evaluate_predictions(y_true, y_score, threshold: float = 0.5) -> dict:
    """Return the main metrics for one set of predicted churn probabilities.

    ROC-AUC and PR-AUC use the probabilities directly (they do not depend on the
    threshold); precision, recall, and F1 use the labels at the given threshold.
    """
    y_predicted = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_predicted, zero_division=0),
        "recall": recall_score(y_true, y_predicted),
        "f1": f1_score(y_true, y_predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def threshold_for_target_recall(y_true, y_score, target_recall: float) -> float:
    """Find the threshold that reaches at least target_recall with the best precision.

    Among all thresholds whose recall meets the target, we keep the one with the
    highest precision (which is also the highest threshold), so we hit the recall
    goal while bothering as few staying customers as possible. If the target
    cannot be reached, fall back to the lowest threshold, which gives the most
    recall available.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # thresholds has one fewer entry than precision/recall; align by dropping last.
    precision = precision[:-1]
    recall = recall[:-1]

    meets_target = recall >= target_recall
    if not meets_target.any():
        return float(thresholds[0])

    best_index = np.argmax(precision[meets_target])
    return float(thresholds[meets_target][best_index])


def precision_at_target_recall(y_true, y_score, target_recall: float) -> float:
    """Best precision achievable while keeping recall at or above the target.

    This is the tie-breaker metric when two models have similar PR-AUC.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    meeting = precision[:-1][recall[:-1] >= target_recall]
    if meeting.size == 0:
        return 0.0
    return float(meeting.max())


def plot_model_comparison(results: pd.DataFrame, metric: str = "pr_auc") -> plt.Figure:
    """Bar chart comparing one cross-validation metric across model/strategy rows.

    results is expected to have columns 'model', 'strategy', and the metric.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, x="model", y=metric, hue="strategy", ax=ax)
    ax.set_title(f"Cross-validation {metric} by model and imbalance strategy")
    ax.set_ylabel(metric)
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, y_score, label: str = "model") -> plt.Figure:
    """Precision-recall curve with the no-skill baseline (the churn base rate)."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    base_rate = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"{label} (AP={average_precision_score(y_true, y_score):.3f})")
    ax.axhline(base_rate, color="gray", linestyle="--", label=f"No-skill ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_threshold_analysis(y_true, y_score, chosen_threshold: float) -> plt.Figure:
    """Show precision, recall, and outreach volume as the threshold moves."""
    grid = np.linspace(0.05, 0.95, 91)
    precision_values = []
    recall_values = []
    flagged_share = []
    for threshold in grid:
        predicted = (np.asarray(y_score) >= threshold).astype(int)
        precision_values.append(precision_score(y_true, predicted, zero_division=0))
        recall_values.append(recall_score(y_true, predicted))
        flagged_share.append(predicted.mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, precision_values, label="Precision")
    ax.plot(grid, recall_values, label="Recall")
    ax.plot(grid, flagged_share, label="Share of customers flagged", linestyle=":")
    ax.axvline(chosen_threshold, color="red", linestyle="--", label=f"Chosen ({chosen_threshold:.2f})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Value")
    ax.set_title("Precision, recall, and outreach volume vs threshold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_score, threshold: float) -> plt.Figure:
    """Confusion matrix at a given threshold, labelled Stay/Churn."""
    predicted = (np.asarray(y_score) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold {threshold:.2f})")
    fig.tight_layout()
    return fig


def plot_calibration(y_true, y_score, n_bins: int = 10) -> plt.Figure:
    """Reliability curve: do predicted probabilities match observed churn rates?"""
    observed, predicted = calibration_curve(y_true, y_score, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(predicted, observed, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    return fig
