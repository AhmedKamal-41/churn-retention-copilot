"""Tests for Stage 3 model building and training."""

import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImbalancedPipeline

from src.data.clean import clean_data
from src.data.load import load_raw_data
from src.data.split import split_data
from src.models.train import (
    build_pipeline,
    load_model,
    make_candidate,
    make_classifier,
    save_model,
)


@pytest.fixture(scope="module")
def split():
    return split_data(clean_data(load_raw_data()))


def test_make_classifier_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_classifier("nonexistent")


def test_smote_candidate_includes_smote_step():
    candidate = make_candidate("logistic", "smote")
    step_names = [name for name, _ in candidate.steps]
    assert "smote" in step_names


def test_non_smote_candidate_has_no_smote_step():
    candidate = make_candidate("logistic", "original")
    step_names = [name for name, _ in candidate.steps]
    assert "smote" not in step_names


def test_pipeline_starts_with_preprocessor():
    candidate = make_candidate("random_forest", "class_weight")
    assert isinstance(candidate, ImbalancedPipeline)
    assert candidate.steps[0][0] == "preprocess"


def test_pipeline_accepts_raw_features_and_outputs_valid_probabilities(split):
    model = make_candidate("logistic", "original")
    model.fit(split.X_train, split.y_train)
    proba = model.predict_proba(split.X_val)[:, 1]

    assert proba.shape == (len(split.X_val),)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_saved_model_predicts_identically(split, tmp_path):
    model = make_candidate("logistic", "original")
    model.fit(split.X_train, split.y_train)
    expected = model.predict_proba(split.X_val)[:, 1]

    path = tmp_path / "model.joblib"
    save_model(model, path)
    reloaded = load_model(path)

    assert np.allclose(reloaded.predict_proba(split.X_val)[:, 1], expected)
