"""Turn customers into numeric vectors for similarity search (Stage 5).

We reuse the exact preprocessor fitted in Stage 2, so a customer is vectorised the
same way for the model, the similarity search, and the dashboard. The churn label
and customer ID are kept beside the vectors as lookup metadata - they never go
inside a vector, which keeps the similarity space free of target leakage.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ID_COLUMN, TARGET_COLUMN
from src.data.split import FEATURE_COLUMNS


@dataclass
class CustomerVectors:
    """Historical customers as vectors, with metadata kept separate.

    vectors: processed feature vectors, shape (n_customers, n_features).
    ids: customer IDs, aligned by position with the vectors.
    stayed: True where the customer stayed (Churn == "No"); used only to filter
        neighbors, never as a model feature.
    raw_features: the original feature values, for showing shared traits later.
    """

    vectors: np.ndarray
    ids: pd.Series
    stayed: np.ndarray
    raw_features: pd.DataFrame


def build_customer_vectors(preprocessor, customers: pd.DataFrame) -> CustomerVectors:
    """Vectorise a table of historical customers (cleaned, with ID and Churn)."""
    features = customers[FEATURE_COLUMNS].reset_index(drop=True)
    vectors = preprocessor.transform(features)
    ids = customers[ID_COLUMN].reset_index(drop=True)
    stayed = (customers[TARGET_COLUMN] == "No").to_numpy()
    return CustomerVectors(vectors, ids, stayed, features)


def vectorize_one(preprocessor, customer_row: pd.DataFrame) -> np.ndarray:
    """Vectorise a single new customer; returns a (1, n_features) array.

    customer_row is a one-row DataFrame holding the feature columns.
    """
    return preprocessor.transform(customer_row[FEATURE_COLUMNS])
