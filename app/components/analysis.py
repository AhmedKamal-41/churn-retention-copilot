"""Run the full analysis pipeline for one customer (dashboard orchestration).

This ties together the model, explainer, similarity search, and retention planner
from `src/`. The result is a single dictionary the pages render. Keeping it here
(not in a page) means the same analysis can be tested without Streamlit.
"""

import pandas as pd

from src.config import SIMILARITY_MAX_NEIGHBORS, SIMILARITY_MIN_THRESHOLD
from src.explainability.explainer import explain_customer
from src.retention.planner import build_retention_plan
from src.similarity.projector import build_map_data
from src.similarity.search import (
    compare_characteristics,
    find_similar_churned,
    find_similar_retained,
)
from src.similarity.vectorizer import vectorize_one

# How many neighbors to show the detailed shared/different breakdown for.
_PEER_DETAIL_COUNT = 5


def analyze_customer(reference, new_customer: pd.DataFrame) -> dict:
    """Predict, explain, find similar retained customers, and build a plan.

    reference is the bundle from loaders.build_reference / load_reference.
    new_customer is a one-row DataFrame of the feature columns. The retention plan
    and the map share the same qualified neighbor set (the closest retained
    customers above the similarity threshold, up to the max neighbor count).
    """
    new_vector = vectorize_one(reference.preprocessor, new_customer)
    neighbors = find_similar_retained(
        reference.customer_vectors,
        new_vector,
        top_k=SIMILARITY_MAX_NEIGHBORS,
        min_similarity=SIMILARITY_MIN_THRESHOLD,
    )
    churned_neighbors = find_similar_churned(
        reference.customer_vectors,
        new_vector,
        top_k=SIMILARITY_MAX_NEIGHBORS,
        min_similarity=SIMILARITY_MIN_THRESHOLD,
    )
    explanation = explain_customer(reference.explainer, new_customer)
    map_data = build_map_data(
        reference.projector, reference.customer_vectors, new_vector, neighbors
    )
    plan = build_retention_plan(
        explanation, new_customer, neighbors, churned_neighbors, reference.customer_vectors
    )

    comparisons = []
    for _, neighbor in neighbors.head(_PEER_DETAIL_COUNT).iterrows():
        neighbor_features = reference.customer_vectors.raw_features.iloc[
            int(neighbor["row_index"])
        ]
        comparison = compare_characteristics(new_customer, neighbor_features)
        comparisons.append(
            {
                "customer_id": neighbor["customer_id"],
                "similarity": float(neighbor["similarity"]),
                "shared": comparison["shared"],
                "different": comparison["different"],
            }
        )

    return {
        "customer": new_customer,
        "explanation": explanation,
        "neighbors": neighbors,
        "map_data": map_data,
        "plan": plan,
        "comparisons": comparisons,
    }
