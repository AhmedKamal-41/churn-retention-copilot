"""Model Reference page: methodology, metrics, and how to read outputs."""

import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json

import streamlit as st

from app.components.theme import (
    apply_theme,
    card,
    configure_page,
    footnote,
    page_header,
    render_sidebar,
    section_label,
    stat_row,
)
from src.config import DECISION_THRESHOLD_PATH

configure_page("Model Reference")
apply_theme()
render_sidebar()

page_header(
    "Model reference",
    "How the churn model was trained, validated, and deployed into this workspace.",
)

threshold_info = json.loads(DECISION_THRESHOLD_PATH.read_text())
threshold = threshold_info["threshold"]

stat_row(
    [
        ("Classifier", "Logistic regression", "Selected on PR-AUC + precision at 80% recall"),
        ("Operating threshold", f"{threshold:.3f}", "Tuned for ~80% validation recall"),
        ("Preprocessing", "Single ColumnTransformer", "Fit on train only; reused everywhere"),
    ]
)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    section_label("Training design")
    card(
        "Data & split",
        "<p>IBM Telco Customer Churn — 7,043 accounts, ~26.5% churn rate. "
        "Stratified 60/20/20 train/validation/test split before any transformation.</p>",
    )
    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
    card(
        "Imbalance handling",
        "<p>Compared original class weights, SMOTE inside imblearn pipelines, and "
        "unweighted baselines. On this dataset, SMOTE and class weights did not improve "
        "PR-AUC; logistic regression with the original distribution was retained.</p>",
    )

with col_b:
    section_label("Reading outputs")
    card(
        "Risk tiers",
        "<p><strong>Low</strong> — below the operating threshold.<br>"
        "<strong>Medium</strong> — at or above threshold but below 0.50 probability.<br>"
        "<strong>High</strong> — probability ≥ 0.50.</p>",
    )
    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
    card(
        "Similarity & map",
        "<p>Each customer is embedded in a 46-feature space after preprocessing. "
        "Cosine similarity ranks retained neighbors. The vector map is a PCA projection "
        "for orientation only — scores are computed in the full space.</p>",
    )

section_label("Limitations")
footnote(
    "Explanations report model associations, not causal mechanisms. Retention "
    "recommendations address observed risk drivers; the dataset contains no logged "
    "offer outcomes, so similar retained customers are descriptive peers rather than "
    "proof that an intervention worked."
)
