"""Build and train the churn model candidates (Stage 3).

Every candidate is a single leakage-safe pipeline:

    preprocessor  ->  (optional SMOTE)  ->  classifier

Because the preprocessor lives inside the pipeline, it is refit on the training
part of each cross-validation fold, and SMOTE only ever runs on training folds -
never on the data a fold is scored against. This is what keeps the comparison
honest. The pipeline also takes a raw customer record as input, so the saved
model needs no separate preprocessing step at prediction time.
"""

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.config import RANDOM_SEED
from src.features.preprocessor import build_preprocessor

# The imbalance strategies we compare for each model.
IMBALANCE_STRATEGIES = ["original", "class_weight", "smote"]

# Metrics collected during cross-validation. PR-AUC (average_precision) is the
# primary one because the classes are imbalanced and we care about the churn class.
CROSS_VALIDATION_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}


def make_classifier(name: str, strategy: str = "original", pos_weight: float = 1.0):
    """Create one base classifier configured for an imbalance strategy.

    name: "dummy", "logistic", "random_forest", or "xgboost".
    strategy: "original" or "class_weight". (SMOTE is handled by the pipeline,
        not by the classifier, so callers pass "original" together with SMOTE.)
    pos_weight: churn-to-stay ratio used by XGBoost when class weighting is on;
        compute it from the training labels as (# stayed) / (# churned).
    """
    use_class_weight = strategy == "class_weight"
    balanced = "balanced" if use_class_weight else None

    if name == "dummy":
        # Predicts the base churn rate for every customer - the baseline to beat.
        return DummyClassifier(strategy="prior")
    if name == "logistic":
        return LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, class_weight=balanced
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1, class_weight=balanced
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=pos_weight if use_class_weight else 1.0,
        )
    raise ValueError(f"unknown model name: {name!r}")


def build_pipeline(classifier, use_smote: bool = False) -> ImbalancedPipeline:
    """Wrap a classifier in the full preprocessor -> (SMOTE) -> classifier pipeline."""
    steps = [("preprocess", build_preprocessor())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))
    steps.append(("classifier", classifier))
    return ImbalancedPipeline(steps)


def make_candidate(name: str, strategy: str, pos_weight: float = 1.0) -> ImbalancedPipeline:
    """Build a ready-to-fit pipeline for one (model, imbalance strategy) pair."""
    classifier_strategy = "original" if strategy == "smote" else strategy
    classifier = make_classifier(name, classifier_strategy, pos_weight)
    return build_pipeline(classifier, use_smote=strategy == "smote")


def make_cv(n_splits: int = 5) -> StratifiedKFold:
    """Stratified k-fold splitter with a fixed seed, so results are reproducible."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


def cross_validate_model(model, X, y, n_splits: int = 5) -> dict:
    """Cross-validate a pipeline and return mean/std for each scoring metric."""
    results = cross_validate(
        model, X, y, cv=make_cv(n_splits), scoring=CROSS_VALIDATION_SCORING, n_jobs=-1
    )
    summary = {}
    for metric in CROSS_VALIDATION_SCORING:
        scores = results[f"test_{metric}"]
        summary[metric] = {"mean": float(scores.mean()), "std": float(scores.std())}
    return summary


def tune_model(model, param_grid: dict, X, y, n_splits: int = 5) -> GridSearchCV:
    """Small grid search over param_grid, scored by PR-AUC, refit on the best params."""
    search = GridSearchCV(
        model,
        param_grid,
        scoring="average_precision",
        cv=make_cv(n_splits),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search


def save_model(model, path) -> None:
    """Save a fitted model pipeline for the dashboard and tests to reuse."""
    joblib.dump(model, path)


def load_model(path):
    """Load a model saved by save_model."""
    return joblib.load(path)
