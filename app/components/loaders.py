"""Load saved artifacts and build the reference set for the dashboard.

Everything here is built once and reused: the dashboard loads the saved model,
preprocessor, and threshold, and builds the customer vectors, 2D projector, and
explainer from the historical data. It never retrains the model.
"""

import json
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.config import DATA_PROCESSED_DIR, DECISION_THRESHOLD_PATH, MODEL_PATH, MODELS_DIR
from src.data.split import split_data
from src.explainability.explainer import ChurnExplainer, build_explainer
from src.features.preprocessor import load_preprocessor
from src.models.train import load_model
from src.similarity.projector import fit_projector
from src.similarity.vectorizer import CustomerVectors, build_customer_vectors


@dataclass
class Reference:
    """The fixed pieces every customer analysis needs."""

    model: object
    preprocessor: object
    decision_threshold: float
    customer_vectors: CustomerVectors
    projector: object
    explainer: ChurnExplainer


def build_reference() -> Reference:
    """Load artifacts and build the reference set (no Streamlit caching).

    Kept separate from the cached version so tests can call it directly.
    """
    customers = pd.read_csv(DATA_PROCESSED_DIR / "telco_clean.csv")
    preprocessor = load_preprocessor(MODELS_DIR / "preprocessor.joblib")
    model = load_model(MODEL_PATH)
    decision_threshold = json.loads(DECISION_THRESHOLD_PATH.read_text())["threshold"]

    training_features = split_data(customers).X_train
    explainer = build_explainer(model, training_features, decision_threshold)
    customer_vectors = build_customer_vectors(preprocessor, customers)
    projector = fit_projector(customer_vectors.vectors)

    return Reference(
        model=model,
        preprocessor=preprocessor,
        decision_threshold=decision_threshold,
        customer_vectors=customer_vectors,
        projector=projector,
        explainer=explainer,
    )


@st.cache_resource
def load_reference() -> Reference:
    """Cached reference set, built once and shared across pages and reruns."""
    return build_reference()
