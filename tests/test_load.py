"""Tests for Stage 1 data loading.

These confirm the raw dataset is present and has the shape and columns the rest
of the project expects, so later stages can rely on a known starting point.
"""

import pandas as pd
import pytest

from src.config import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)
from src.data.load import load_raw_data


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_raw_data()


def test_dataset_shape(raw_df):
    assert raw_df.shape == (7043, 21)


def test_expected_columns_present(raw_df):
    expected = {ID_COLUMN, TARGET_COLUMN, *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS}
    assert expected == set(raw_df.columns)


def test_target_values_are_yes_no(raw_df):
    assert set(raw_df[TARGET_COLUMN].unique()) == {"Yes", "No"}


def test_customer_ids_are_unique(raw_df):
    assert raw_df[ID_COLUMN].is_unique


def test_loader_returns_raw_total_charges_as_text(raw_df):
    # Stage 1 must not clean the data: TotalCharges should still be text,
    # because the cleaning that converts it to numeric belongs to Stage 2.
    assert raw_df["TotalCharges"].dtype == object
