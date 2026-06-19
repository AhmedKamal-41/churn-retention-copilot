"""Customer Assessment page: enter a profile, score churn risk, review drivers."""

import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.components.analysis import analyze_customer
from app.components.input_form import render_input_form
from app.components.loaders import load_reference
from app.components.results import render_prediction_results
from app.components.theme import (
    apply_theme,
    configure_page,
    footnote,
    page_header,
    render_sidebar,
    section_label,
)

configure_page("Customer Assessment")
apply_theme()
render_sidebar()

page_header(
    "Customer assessment",
    "Enter account attributes, run the fitted pipeline, and review the risk score with "
    "per-customer factor contributions.",
)

reference = load_reference()

with st.container(border=True):
    section_label("Account input")
    new_customer, analyze_clicked = render_input_form()

if analyze_clicked:
    with st.spinner("Scoring profile and building explanation…"):
        st.session_state["analysis"] = analyze_customer(reference, new_customer)

if "analysis" in st.session_state:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    section_label("Risk output")
    render_prediction_results(st.session_state["analysis"])
    footnote(
        "Continue to Vector Map and Retention Actions in the sidebar for peer "
        "positioning and the outreach brief."
    )
