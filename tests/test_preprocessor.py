"""Tests for the Stage 2 preprocessing pipeline."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.data.clean import clean_data
from src.data.load import load_raw_data
from src.data.split import split_data
from src.features.preprocessor import (
    build_preprocessor,
    load_preprocessor,
    save_preprocessor,
    transform_to_frame,
)


@pytest.fixture(scope="module")
def split():
    return split_data(clean_data(load_raw_data()))


@pytest.fixture(scope="module")
def fitted_preprocessor(split):
    preprocessor = build_preprocessor("standard")
    preprocessor.fit(split.X_train)
    return preprocessor


def test_build_rejects_unknown_scaler():
    with pytest.raises(ValueError):
        build_preprocessor("banana")


def test_transform_before_fit_raises(split):
    preprocessor = build_preprocessor("standard")
    with pytest.raises(NotFittedError):
        preprocessor.transform(split.X_train)


def test_processed_output_shape_and_names(fitted_preprocessor, split):
    processed = transform_to_frame(fitted_preprocessor, split.X_train)
    assert processed.shape == (len(split.X_train), 46)
    assert list(processed.columns) == list(fitted_preprocessor.get_feature_names_out())


def test_fit_uses_training_data_only(fitted_preprocessor, split):
    # StandardScaler fit on train should make the train numeric features average
    # to roughly 0. If the scaler had seen other data, this would not hold.
    processed_train = transform_to_frame(fitted_preprocessor, split.X_train)
    numeric_columns = [c for c in processed_train.columns if c.startswith("numeric__")]
    assert np.allclose(processed_train[numeric_columns].mean(), 0, atol=1e-9)


def test_unknown_category_is_ignored(fitted_preprocessor, split):
    # A value never seen in training (here a made-up Contract) must not crash.
    new_customer = split.X_train.iloc[[0]].copy()
    new_customer["Contract"] = "Lifetime"
    processed = transform_to_frame(fitted_preprocessor, new_customer)
    assert processed.shape == (1, 46)


def test_saved_preprocessor_round_trips(fitted_preprocessor, split, tmp_path):
    path = tmp_path / "preprocessor.joblib"
    save_preprocessor(fitted_preprocessor, path)
    reloaded = load_preprocessor(path)

    original = transform_to_frame(fitted_preprocessor, split.X_val)
    after_reload = transform_to_frame(reloaded, split.X_val)
    assert original.equals(after_reload)
