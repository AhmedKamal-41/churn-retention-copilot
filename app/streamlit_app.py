"""Home page for the Customer Churn Retention dashboard.

Run with:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.components.theme import (
    APP_TAGLINE,
    APP_TITLE,
    apply_theme,
    card,
    configure_page,
    footnote,
    hero,
    render_sidebar,
    section_label,
    stat_row,
)

configure_page("Overview")
apply_theme()
render_sidebar()

hero(
    APP_TITLE,
    APP_TAGLINE,
    kicker="Telco retention workspace",
)

stat_row(
    [
        ("Dataset", "7,043", "IBM Telco churn records"),
        ("Churn rate", "26.5%", "Imbalanced target — PR metrics drive decisions"),
        ("Pipeline", "6 stages", "EDA through retention planning"),
    ]
)

st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    card(
        "What this tool does",
        "<p>Score a single account for churn risk, surface the model drivers for that "
        "profile, locate retained customers with similar feature vectors, and translate "
        "the evidence into prioritized outreach actions.</p>",
    )

with right:
    card(
        "Typical workflow",
        "<ul>"
        "<li>Open <strong>Customer Assessment</strong> and load a profile or enter account data.</li>"
        "<li>Review probability, risk tier, and factor contributions.</li>"
        "<li>Inspect the vector map and peer comparisons on the following pages.</li>"
        "<li>Hand off the retention brief to the account team.</li>"
        "</ul>",
    )

section_label("Interpretation guardrails")
footnote(
    "Scores and factor contributions reflect patterns learned from historical data — "
    "they are not causal proof. Recommended actions target observed risk drivers; "
    "similar retained customers provide descriptive context, not evidence that a "
    "specific offer prevented churn."
)
