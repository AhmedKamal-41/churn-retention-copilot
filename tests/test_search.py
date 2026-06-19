"""Tests for Stage 5 similarity search."""

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.features.preprocessor import load_preprocessor
from src.similarity.search import compare_characteristics, find_similar_retained
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


def test_returns_requested_number_of_neighbors(customer_vectors, preprocessor, customers):
    new_vector = vectorize_one(preprocessor, customers.iloc[[0]])
    neighbors = find_similar_retained(customer_vectors, new_vector, top_k=5)
    assert len(neighbors) == 5


def test_all_neighbors_are_retained_customers(customer_vectors, preprocessor, customers):
    new_vector = vectorize_one(preprocessor, customers.iloc[[0]])
    neighbors = find_similar_retained(customer_vectors, new_vector, top_k=10)
    for row_index in neighbors["row_index"]:
        assert customer_vectors.stayed[row_index]


def test_similarity_scores_are_in_valid_range_and_sorted(customer_vectors, preprocessor, customers):
    new_vector = vectorize_one(preprocessor, customers.iloc[[0]])
    neighbors = find_similar_retained(customer_vectors, new_vector, top_k=5)
    similarities = neighbors["similarity"].to_numpy()
    assert np.all(similarities >= -1.0) and np.all(similarities <= 1.0)
    assert list(similarities) == sorted(similarities, reverse=True)


def test_exclude_index_drops_the_query_customer(customer_vectors):
    # Use a retained customer as the query; with exclusion it must not return itself.
    retained_index = int(np.where(customer_vectors.stayed)[0][0])
    query_vector = customer_vectors.vectors[retained_index : retained_index + 1]

    neighbors = find_similar_retained(
        customer_vectors, query_vector, top_k=5, exclude_index=retained_index
    )
    assert retained_index not in set(neighbors["row_index"])


def test_compare_characteristics_splits_shared_and_different(customer_vectors, preprocessor, customers):
    new_customer = customers.iloc[[0]]
    new_vector = vectorize_one(preprocessor, new_customer)
    neighbors = find_similar_retained(customer_vectors, new_vector, top_k=1)
    neighbor_features = customer_vectors.raw_features.iloc[int(neighbors.iloc[0]["row_index"])]

    comparison = compare_characteristics(new_customer, neighbor_features)
    total = len(comparison["shared"]) + len(comparison["different"])
    # Every one of the 19 feature columns lands in exactly one bucket.
    assert total == 19
