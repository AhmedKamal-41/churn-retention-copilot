"""Tests for Stage 5 2D projection and map data."""

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.features.preprocessor import load_preprocessor
from src.similarity import projector
from src.similarity.search import find_similar_retained
from src.similarity.vectorizer import build_customer_vectors, vectorize_one


@pytest.fixture(scope="module")
def customer_vectors():
    preprocessor = load_preprocessor(MODELS_DIR / "preprocessor.joblib")
    customers = pd.read_csv(DATA_PROCESSED_DIR / "telco_clean.csv")
    return build_customer_vectors(preprocessor, customers)


@pytest.fixture(scope="module")
def fitted_projector(customer_vectors):
    return projector.fit_projector(customer_vectors.vectors)


def test_projection_is_two_dimensional(fitted_projector, customer_vectors):
    coordinates = projector.project(fitted_projector, customer_vectors.vectors)
    assert coordinates.shape == (customer_vectors.vectors.shape[0], 2)


def test_projection_is_stable_for_same_input(fitted_projector, customer_vectors):
    first = projector.project(fitted_projector, customer_vectors.vectors[:5])
    second = projector.project(fitted_projector, customer_vectors.vectors[:5])
    assert np.allclose(first, second)


def test_map_data_has_all_groups_and_one_new_point(customer_vectors, fitted_projector):
    preprocessor = load_preprocessor(MODELS_DIR / "preprocessor.joblib")
    customers = pd.read_csv(DATA_PROCESSED_DIR / "telco_clean.csv")
    new_vector = vectorize_one(preprocessor, customers.iloc[[0]])
    neighbors = find_similar_retained(customer_vectors, new_vector, top_k=5)

    map_data = projector.build_map_data(fitted_projector, customer_vectors, new_vector, neighbors)

    assert len(map_data) == len(customer_vectors.ids) + 1
    assert (map_data["group"] == "new").sum() == 1
    assert (map_data["group"] == "neighbor").sum() == 5
    # Neighbors carry a similarity score; the new point is marked 1.0.
    assert map_data.loc[map_data["group"] == "neighbor", "similarity"].notna().all()
