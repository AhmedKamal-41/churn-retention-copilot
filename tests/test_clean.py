"""Tests for Stage 2 cleaning rules."""

import pandas as pd
import pytest

from src.data.clean import clean_data
from src.data.load import load_raw_data


@pytest.fixture(scope="module")
def cleaned_df() -> pd.DataFrame:
    return clean_data(load_raw_data())


def test_total_charges_is_numeric_after_cleaning(cleaned_df):
    assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])


def test_no_missing_total_charges_after_cleaning(cleaned_df):
    assert cleaned_df["TotalCharges"].isnull().sum() == 0


def test_blank_total_charges_become_zero_for_new_customers(cleaned_df):
    new_customers = cleaned_df[cleaned_df["tenure"] == 0]
    assert (new_customers["TotalCharges"] == 0).all()
    # Stage 1 found exactly 11 such customers.
    assert len(new_customers) == 11


def test_cleaning_keeps_all_rows_and_columns(cleaned_df):
    raw = load_raw_data()
    assert cleaned_df.shape == raw.shape


def test_cleaning_does_not_change_other_columns(cleaned_df):
    raw = load_raw_data()
    untouched = [c for c in raw.columns if c != "TotalCharges"]
    pd.testing.assert_frame_equal(cleaned_df[untouched], raw[untouched])
