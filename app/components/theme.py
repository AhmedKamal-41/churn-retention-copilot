"""Shared layout, styling, and UI helpers for the retention dashboard."""

import json

import streamlit as st

from src.config import DECISION_THRESHOLD_PATH, MODEL_PATH

# Palette tuned for an internal analytics console — restrained, not traffic-light loud.
COLORS = {
    "ink": "#0f172a",
    "slate": "#334155",
    "muted": "#64748b",
    "line": "#e2e8f0",
    "surface": "#ffffff",
    "accent": "#0c4a6e",
    "accent_soft": "#e0f2fe",
    "high": "#991b1b",
    "medium": "#9a3412",
    "low": "#166534",
}

RISK_COLORS = {"High": COLORS["high"], "Medium": COLORS["medium"], "Low": COLORS["low"]}

APP_TITLE = "Customer Churn Intelligence"
APP_TAGLINE = "Retention decision support for telecom account teams"

_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #0f172a;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
        color: #cbd5e1 !important;
        font-size: 0.88rem;
    }

    [data-testid="stSidebarNav"] a {
        font-weight: 500;
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 6px;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }

    .stButton > button[kind="primary"] {
        background: #0c4a6e;
        border: none;
        font-weight: 600;
        letter-spacing: 0.01em;
        border-radius: 8px;
    }

    .stButton > button[kind="primary"]:hover {
        background: #075985;
        border: none;
    }

    .stButton > button[kind="secondary"] {
        border-radius: 8px;
        border-color: #cbd5e1;
        color: #334155;
        font-weight: 500;
    }

    .dash-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0c4a6e 100%);
        color: #f8fafc;
        border-radius: 14px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .dash-hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
    }

    .dash-hero p {
        margin: 0;
        color: #cbd5e1;
        font-size: 0.98rem;
        max-width: 52rem;
        line-height: 1.55;
    }

    .dash-kicker {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7dd3fc;
        margin-bottom: 0.55rem;
    }

    .dash-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .dash-card h3 {
        margin: 0 0 0.35rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
    }

    .dash-card p, .dash-card li {
        margin: 0;
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .dash-card ul {
        margin: 0.5rem 0 0 1.1rem;
        padding: 0;
    }

    .dash-section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #64748b;
        margin: 0 0 0.45rem 0;
    }

    .dash-page-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }

    .dash-page-sub {
        color: #64748b;
        font-size: 0.95rem;
        margin: 0 0 1.25rem 0;
        line-height: 1.5;
    }

    .dash-stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-bottom: 0.5rem;
    }

    .dash-stat {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.95rem 1rem;
    }

    .dash-stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.25rem;
    }

    .dash-stat-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }

    .dash-stat-note {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    .dash-badge {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .dash-rec {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0c4a6e;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
    }

    .dash-rec-title {
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0 0 0.55rem 0;
    }

    .dash-rec-meta {
        font-size: 0.88rem;
        color: #475569;
        margin: 0.2rem 0;
        line-height: 1.45;
    }

    .dash-footnote {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.45;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid #e2e8f0;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        background: #f8fafc;
    }

    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background: #ffffff;
    }

    [data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    footer, [data-testid="stToolbar"] {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .dash-sidebar-brand {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #f8fafc !important;
        margin-bottom: 0.15rem;
    }

    .dash-sidebar-meta {
        font-size: 0.75rem !important;
        color: #94a3b8 !important;
        line-height: 1.45 !important;
        margin-bottom: 1rem !important;
    }
</style>
"""


def configure_page(page_title: str):
    """Apply shared page config. Call once at the top of each page module."""
    st.set_page_config(
        page_title=f"{page_title} · {APP_TITLE}",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_theme():
    """Inject global CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)


def _model_metadata() -> dict:
    threshold_info = json.loads(DECISION_THRESHOLD_PATH.read_text())
    return {
        "model": threshold_info.get("model", "logistic_regression").replace("_", " ").title(),
        "threshold": threshold_info["threshold"],
        "target_recall": threshold_info.get("target_recall", 0.8),
        "artifact": MODEL_PATH.name,
    }


def render_sidebar():
    """Brand block and model metadata shown on every page."""
    meta = _model_metadata()
    with st.sidebar:
        st.markdown(
            f'<p class="dash-sidebar-brand">{APP_TITLE}</p>'
            f'<p class="dash-sidebar-meta">'
            f"Model: {meta['model']}<br>"
            f"Operating threshold: {meta['threshold']:.3f}<br>"
            f"Validation recall target: {meta['target_recall']:.0%}"
            f"</p>",
            unsafe_allow_html=True,
        )


def page_header(title: str, subtitle: str):
    """Consistent page heading."""
    st.markdown(f'<p class="dash-page-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="dash-page-sub">{subtitle}</p>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, kicker: str = "Retention analytics"):
    """Landing-page hero block."""
    st.markdown(
        f'<div class="dash-hero">'
        f'<div class="dash-kicker">{kicker}</div>'
        f"<h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def section_label(text: str):
    st.markdown(f'<p class="dash-section-label">{text}</p>', unsafe_allow_html=True)


def risk_badge(level: str) -> str:
    color = RISK_COLORS.get(level, COLORS["slate"])
    bg = f"{color}18"
    return (
        f'<span class="dash-badge" style="color:{color};background:{bg};'
        f'border:1px solid {color}33;">{level}</span>'
    )


def stat_card(label: str, value: str, note: str = ""):
    note_html = f'<div class="dash-stat-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="dash-stat">'
        f'<div class="dash-stat-label">{label}</div>'
        f'<div class="dash-stat-value">{value}</div>'
        f"{note_html}</div>",
        unsafe_allow_html=True,
    )


def stat_row(items: list[tuple[str, str, str]]):
    """Render a row of stat cards. Each item is (label, value, note)."""
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        with col:
            stat_card(label, value, note)


def card(title: str, body_html: str):
    st.markdown(
        f'<div class="dash-card"><h3>{title}</h3>{body_html}</div>',
        unsafe_allow_html=True,
    )


def footnote(text: str):
    st.markdown(f'<p class="dash-footnote">{text}</p>', unsafe_allow_html=True)


def empty_state(title: str, message: str):
    """Placeholder when a page needs a prior assessment."""
    st.info(f"**{title}** — {message}")


def recommendation_card(
    index: int,
    action: str,
    priority: str,
    reason: str,
    risk_feature: str,
    risk_value: str,
    contribution: float,
    evidence: str,
):
    border_color = RISK_COLORS.get(priority, COLORS["accent"])
    st.markdown(
        f'<div class="dash-rec" style="border-left-color:{border_color};">'
        f'<p class="dash-rec-title">{index}. {action} {risk_badge(priority)}</p>'
        f'<p class="dash-rec-meta"><strong>Why:</strong> {reason}</p>'
        f'<p class="dash-rec-meta"><strong>Driver:</strong> {risk_feature} = {risk_value} '
        f"(contribution {contribution:+.2f})</p>"
        f'<p class="dash-rec-meta"><strong>Peer evidence:</strong> {evidence}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
