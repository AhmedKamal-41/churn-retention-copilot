"""Retention Actions page: prioritized outreach recommendations."""

import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.components.retention_plan import render_retention_plan
from app.components.theme import (
    apply_theme,
    configure_page,
    empty_state,
    page_header,
    render_sidebar,
)

configure_page("Retention Actions")
apply_theme()
render_sidebar()

page_header(
    "Retention actions",
    "Evidence-linked outreach recommendations derived from the risk score, SHAP "
    "drivers, and the closest retained customer profiles.",
)

if "analysis" not in st.session_state:
    empty_state(
        "No profile loaded",
        "Run an assessment on the Customer Assessment page to generate the retention brief.",
    )
else:
    render_retention_plan(st.session_state["analysis"])
