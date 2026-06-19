"""Vector Map page: 2D projection of the assessed customer against the base."""

import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.components.similarity_map import render_similarity_map
from app.components.theme import (
    apply_theme,
    configure_page,
    empty_state,
    footnote,
    page_header,
    render_sidebar,
)

configure_page("Vector Map")
apply_theme()
render_sidebar()

page_header(
    "Vector map",
    "Two-dimensional PCA view of the customer base. Neighbor selection uses cosine "
    "similarity on the full preprocessed feature vector — not these display coordinates.",
)

if "analysis" not in st.session_state:
    empty_state(
        "No profile loaded",
        "Run an assessment on the Customer Assessment page to plot this account against "
        "historical customers.",
    )
else:
    render_similarity_map(st.session_state["analysis"])
    footnote(
        "Background points are historical accounts (stayed vs churned). Highlighted "
        "neighbors are the closest retained profiles by cosine similarity in 46-D "
        "feature space."
    )
