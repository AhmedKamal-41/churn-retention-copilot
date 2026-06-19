"""Find similar retained customers by cosine similarity (Stage 5).

Similarity is always measured in the complete processed feature space - the same
vectors the model uses - never on the 2D map coordinates. We then keep only
customers who stayed, because the point is to learn from customers like this one
who did not churn.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TOP_K_NEIGHBORS
from src.similarity.vectorizer import CustomerVectors


def _find_similar(
    customer_vectors: CustomerVectors,
    new_vector: np.ndarray,
    eligible: np.ndarray,
    top_k: int,
    min_similarity: float | None,
    exclude_index: int | None,
) -> pd.DataFrame:
    """Closest customers to new_vector among the `eligible` rows, sorted high to low."""
    similarities = cosine_similarity(new_vector, customer_vectors.vectors)[0]

    eligible = eligible.copy()
    if exclude_index is not None:
        eligible[exclude_index] = False

    eligible_indices = np.where(eligible)[0]
    if min_similarity is not None:
        eligible_indices = eligible_indices[similarities[eligible_indices] >= min_similarity]
    ordered = eligible_indices[np.argsort(similarities[eligible_indices])[::-1]]
    top = ordered[:top_k]

    return pd.DataFrame(
        {
            "row_index": top,
            "customer_id": customer_vectors.ids.iloc[top].to_numpy(),
            "similarity": similarities[top],
        }
    )


def find_similar_retained(
    customer_vectors: CustomerVectors,
    new_vector: np.ndarray,
    top_k: int = TOP_K_NEIGHBORS,
    min_similarity: float | None = None,
    exclude_index: int | None = None,
) -> pd.DataFrame:
    """Return the closest retained customers to new_vector.

    new_vector is a (1, n_features) array. min_similarity, when given, drops any
    neighbor below that cosine score (so the retention planner can use all qualified
    close neighbors up to top_k, rather than a fixed count). exclude_index drops one
    row (use it when the query customer is itself part of the historical pool). The
    result is sorted by similarity, highest first.
    """
    return _find_similar(
        customer_vectors, new_vector, customer_vectors.stayed,
        top_k, min_similarity, exclude_index,
    )


def find_similar_churned(
    customer_vectors: CustomerVectors,
    new_vector: np.ndarray,
    top_k: int = TOP_K_NEIGHBORS,
    min_similarity: float | None = None,
    exclude_index: int | None = None,
) -> pd.DataFrame:
    """Return the closest customers who churned. Used to compare what similar
    retained customers do differently from similar churned ones."""
    return _find_similar(
        customer_vectors, new_vector, ~customer_vectors.stayed,
        top_k, min_similarity, exclude_index,
    )


def compare_characteristics(
    new_customer: pd.DataFrame,
    neighbor_features: pd.Series,
    numeric_tolerance: float = 0.15,
) -> dict:
    """Compare a new customer with one neighbor on the original columns.

    Categorical columns are shared when the values match. Numeric columns are
    shared when they are within numeric_tolerance (a relative difference), and
    counted as a difference otherwise. Returns two lists, "shared" and "different",
    each holding {feature, new_value, neighbor_value} entries.
    """
    new_values = new_customer.iloc[0]
    shared = []
    different = []

    for column in CATEGORICAL_COLUMNS:
        entry = {
            "feature": column,
            "new_value": new_values[column],
            "neighbor_value": neighbor_features[column],
        }
        if new_values[column] == neighbor_features[column]:
            shared.append(entry)
        else:
            different.append(entry)

    for column in NUMERIC_COLUMNS:
        new_number = float(new_values[column])
        neighbor_number = float(neighbor_features[column])
        entry = {
            "feature": column,
            "new_value": new_number,
            "neighbor_value": neighbor_number,
        }
        scale = max(abs(new_number), abs(neighbor_number), 1.0)
        if abs(new_number - neighbor_number) <= numeric_tolerance * scale:
            shared.append(entry)
        else:
            different.append(entry)

    return {"shared": shared, "different": different}
