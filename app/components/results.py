"""Render the prediction results section of the dashboard."""

import pandas as pd
import streamlit as st

from app.components.theme import footnote, risk_badge, section_label, stat_card

# How many factors to show before the "View all" expander.
_TOP_FACTORS = 5


def _factor_table(factors: list) -> pd.DataFrame:
    rows = [
        {
            "Feature": factor["feature"],
            "Value": str(factor["value"]),
            "Contribution": round(factor["contribution"], 3),
        }
        for factor in factors
    ]
    return pd.DataFrame(rows)


def _render_factor_table(factors: list, direction: str):
    if not factors:
        st.caption("No material drivers in this direction for this profile.")
        return

    frame = _factor_table(factors[:_TOP_FACTORS])
    max_abs = max(abs(frame["Contribution"].max()), abs(frame["Contribution"].min()), 0.01)
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Contribution": st.column_config.ProgressColumn(
                "Contribution (log-odds)",
                format="%.3f",
                min_value=-max_abs,
                max_value=max_abs,
            ),
        },
    )
    if len(factors) > _TOP_FACTORS:
        with st.expander(f"All {direction} factors ({len(factors)})"):
            full = _factor_table(factors)
            full_max = max(abs(full["Contribution"].max()), abs(full["Contribution"].min()), 0.01)
            st.dataframe(
                full,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Contribution": st.column_config.ProgressColumn(
                        "Contribution (log-odds)",
                        format="%.3f",
                        min_value=-full_max,
                        max_value=full_max,
                    ),
                },
            )


def render_prediction_results(analysis: dict):
    """Show probability, risk level, outreach priority, and the risk factors."""
    explanation = analysis["explanation"]
    plan = analysis["plan"]
    probability = explanation["probability"]
    risk = explanation["risk_level"]
    priority = plan["outreach_priority"]

    columns = st.columns(3)
    with columns[0]:
        stat_card("Churn probability", f"{probability:.1%}", "Calibrated logistic output")
    with columns[1]:
        stat_card("Risk tier", risk_badge(risk), "Low / medium / high tiering")
    with columns[2]:
        stat_card("Outreach priority", risk_badge(priority), "Queue ordering for retention team")

    st.markdown(
        f'<div style="margin:0.65rem 0 0.15rem 0;font-size:0.78rem;font-weight:600;'
        f'letter-spacing:0.05em;text-transform:uppercase;color:#64748b;">'
        f"Probability relative to scale</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(probability, 0.0), 1.0))
    st.markdown(
        f'<p style="margin:0.15rem 0 0.85rem 0;font-size:0.82rem;color:#64748b;">'
        f"Operating context: tier cutoffs follow validation threshold rules.</p>",
        unsafe_allow_html=True,
    )

    section_label("Factor decomposition")
    columns = st.columns(2, gap="medium")
    with columns[0]:
        st.markdown("**Drivers increasing churn risk**")
        _render_factor_table(explanation["increases_risk"], "risk-increasing")
    with columns[1]:
        st.markdown("**Drivers supporting retention**")
        _render_factor_table(explanation["decreases_risk"], "retention-supporting")

    footnote(
        "Contributions are SHAP values in log-odds space for this account. They "
        "summarize model evidence, not confirmed causal effects."
    )
