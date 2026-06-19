"""Explain the churn model (Stage 4).

Two kinds of explanation:

- Global: which features matter to the model overall (logistic coefficients and
  permutation importance).
- Per customer: for one customer, which factors push their churn risk up or down,
  using SHAP contributions aggregated back to the original business columns.

Everything here is model evidence and association, never proof of cause. Turning
factors into recommendations is Stage 6's job.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

from src.config import (
    CATEGORICAL_COLUMNS,
    MIN_FACTOR_CONTRIBUTION,
    RANDOM_SEED,
    RISK_HIGH_THRESHOLD,
)

sns.set_theme(style="whitegrid")


def _original_feature(encoded_name: str) -> str:
    """Map an encoded feature name back to its original column.

    "numeric__tenure" -> "tenure"
    "categorical__Contract_Two year" -> "Contract"
    """
    if encoded_name.startswith("numeric__"):
        return encoded_name[len("numeric__"):]

    stripped = encoded_name[len("categorical__"):]
    for column in CATEGORICAL_COLUMNS:
        if stripped.startswith(column + "_"):
            return column
    return stripped


def _readable_name(encoded_name: str) -> str:
    """Drop the transformer prefix so names read naturally in charts."""
    return encoded_name.replace("numeric__", "").replace("categorical__", "")


def global_importance(model) -> pd.DataFrame:
    """Logistic-regression coefficients per encoded feature, sorted by strength.

    A positive coefficient pushes toward churn, a negative one toward staying.
    Coefficients are the model's global view; they are associations, not causes.
    """
    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocess"].get_feature_names_out()

    importance = pd.DataFrame(
        {
            "feature": [_readable_name(name) for name in feature_names],
            "coefficient": classifier.coef_[0],
        }
    )
    importance["abs_coefficient"] = importance["coefficient"].abs()
    return importance.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def permutation_importance_scores(model, X, y, n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance on the raw columns, scored by PR-AUC.

    Shuffling a column and measuring the drop in PR-AUC tells us how much the
    model relies on that column, at the business-column level (not encoded).
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    scores = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return scores.sort_values("importance", ascending=False).reset_index(drop=True)


@dataclass
class ChurnExplainer:
    """Everything needed to explain predictions, built once and reused.

    The dashboard builds this at start-up and calls explain_customer per customer.
    """

    model: object
    shap_explainer: object
    feature_names: list
    decision_threshold: float


def build_explainer(model, background_X, decision_threshold: float) -> ChurnExplainer:
    """Build a reusable explainer from the model and a background sample.

    SHAP needs a background set to know what an "average" customer looks like; we
    pass the training features for that.
    """
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    background = preprocessor.transform(background_X)

    shap_explainer = shap.LinearExplainer(classifier, background)
    feature_names = list(preprocessor.get_feature_names_out())
    return ChurnExplainer(model, shap_explainer, feature_names, decision_threshold)


def shap_values_for(explainer: ChurnExplainer, X) -> shap.Explanation:
    """Return SHAP contributions for X as an Explanation carrying feature names."""
    preprocessor = explainer.model.named_steps["preprocess"]
    transformed = preprocessor.transform(X)
    explanation = explainer.shap_explainer(transformed)
    explanation.feature_names = explainer.feature_names
    return explanation


def risk_level(probability: float, medium_cut: float, high_cut: float = RISK_HIGH_THRESHOLD) -> str:
    """Bucket a churn probability into Low / Medium / High.

    medium_cut is the operating decision threshold (Low below it). high_cut is the
    point above which we call the risk High.
    """
    if probability >= high_cut:
        return "High"
    if probability >= medium_cut:
        return "Medium"
    return "Low"


def explain_customer(
    explainer: ChurnExplainer,
    customer_row: pd.DataFrame,
    min_contribution: float = MIN_FACTOR_CONTRIBUTION,
) -> dict:
    """Explain one customer's churn prediction in plain, business-level terms.

    customer_row is a one-row DataFrame with the model's feature columns. SHAP
    contributions for the encoded features are summed back to the original column,
    then split into factors that increase risk and factors that decrease it. Both
    lists are sorted strongest first; factors with a near-zero effect are dropped.
    """
    probability = float(explainer.model.predict_proba(customer_row)[0, 1])

    explanation = shap_values_for(explainer, customer_row)
    contributions = explanation.values[0]

    totals = {}
    for encoded_name, contribution in zip(explainer.feature_names, contributions):
        original = _original_feature(encoded_name)
        totals[original] = totals.get(original, 0.0) + float(contribution)

    factors = []
    for feature, contribution in totals.items():
        if abs(contribution) < min_contribution:
            continue
        factors.append(
            {
                "feature": feature,
                "value": customer_row.iloc[0][feature],
                "contribution": contribution,
            }
        )

    increases = sorted(
        [f for f in factors if f["contribution"] > 0],
        key=lambda f: f["contribution"],
        reverse=True,
    )
    decreases = sorted(
        [f for f in factors if f["contribution"] < 0],
        key=lambda f: f["contribution"],
    )

    return {
        "probability": probability,
        "risk_level": risk_level(probability, explainer.decision_threshold),
        "increases_risk": increases,
        "decreases_risk": decreases,
    }


def plot_global_importance(importance: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Horizontal bar chart of the strongest coefficients, coloured by direction."""
    top = importance.head(top_n).iloc[::-1]
    colors = ["#d62728" if c > 0 else "#2ca02c" for c in top["coefficient"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Coefficient (positive -> higher churn risk)")
    ax.set_title(f"Top {top_n} features by logistic-regression coefficient")
    fig.tight_layout()
    return fig


def plot_permutation_importance(scores: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Horizontal bar chart of permutation importance with error bars."""
    top = scores.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#1f77b4")
    ax.set_xlabel("Drop in PR-AUC when the column is shuffled")
    ax.set_title(f"Top {top_n} features by permutation importance")
    fig.tight_layout()
    return fig


def plot_shap_summary(explanation: shap.Explanation) -> plt.Figure:
    """SHAP beeswarm summary over the encoded features."""
    shap.plots.beeswarm(explanation, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_customer_waterfall(explainer: ChurnExplainer, customer_row: pd.DataFrame) -> plt.Figure:
    """SHAP waterfall plot for a single customer."""
    explanation = shap_values_for(explainer, customer_row)
    shap.plots.waterfall(explanation[0], show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig
