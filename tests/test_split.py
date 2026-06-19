"""Tests for Stage 2 train/validation/test splitting."""

import pytest

from src.config import ID_COLUMN, TARGET_COLUMN
from src.data.clean import clean_data
from src.data.load import load_raw_data
from src.data.split import FEATURE_COLUMNS, split_data


@pytest.fixture(scope="module")
def split():
    return split_data(clean_data(load_raw_data()))


def test_split_sizes_are_60_20_20(split):
    total = len(split.X_train) + len(split.X_val) + len(split.X_test)
    assert total == 7043
    assert len(split.X_train) == 4225
    assert len(split.X_val) == 1409
    assert len(split.X_test) == 1409


def test_stratification_preserves_churn_rate(split):
    rates = [split.y_train.mean(), split.y_val.mean(), split.y_test.mean()]
    for rate in rates:
        assert rate == pytest.approx(0.265, abs=0.01)


def test_splits_do_not_overlap(split):
    train_idx = set(split.X_train.index)
    val_idx = set(split.X_val.index)
    test_idx = set(split.X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_features_exclude_id_and_target(split):
    assert ID_COLUMN not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert list(split.X_train.columns) == FEATURE_COLUMNS


def test_target_is_encoded_as_zero_one(split):
    assert set(split.y_train.unique()) <= {0, 1}


def test_split_is_reproducible(split):
    again = split_data(clean_data(load_raw_data()))
    assert list(again.X_train.index) == list(split.X_train.index)
