"""Tests for Stage 5 customer vectorisation."""

import pandas as pd
import pytest

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.features.preprocessor import load_preprocessor
from src.similarity.vectorizer import build_customer_vectors, vectorize_one


@pytest.fixture(scope="module")
def preprocessor():
    return load_preprocessor(MODELS_DIR / "preprocessor.joblib")


@pytest.fixture(scope="module")
def customers():
    return pd.read_csv(DATA_PROCESSED_DIR / "telco_clean.csv")


@pytest.fixture(scope="module")
def customer_vectors(preprocessor, customers):
    return build_customer_vectors(preprocessor, customers)


def test_vectors_have_one_row_per_customer(customer_vectors, customers):
    assert customer_vectors.vectors.shape[0] == len(customers)
    assert customer_vectors.vectors.shape[1] == 46


def test_metadata_aligns_with_vectors(customer_vectors, customers):
    assert len(customer_vectors.ids) == len(customers)
    assert len(customer_vectors.stayed) == len(customers)
    # "stayed" should match the share of non-churners.
    assert customer_vectors.stayed.sum() == (customers["Churn"] == "No").sum()


def test_a_new_customer_maps_to_same_width(preprocessor, customers):
    one = customers.iloc[[0]]
    vector = vectorize_one(preprocessor, one)
    assert vector.shape == (1, 46)
