"""Render the retention plan section of the dashboard."""

import pandas as pd
import streamlit as st

from app.components.theme import COLORS, RISK_COLORS, footnote, risk_badge, section_label

# Show this many recommendations up front; the rest go in an expander.
_VISIBLE_RECOMMENDATIONS = 3


def _evidence_line(recommendation: dict) -> str:
    mode = recommendation["mode"]
    if mode == "model_only":
        return (
            "<em>Insufficient neighbor evidence — no retained customers cleared the "
            "similarity threshold, so this rests on the model risk factor alone.</em>"
        )
    low, high = recommendation["similarity_range"]
    quality = f"avg similarity {recommendation['avg_similarity']:.3f}, range {low:.3f}–{high:.3f}"
    if mode == "lift":
        return (
            f"{recommendation['retained_rate']:.0%} of similar retained customers vs "
            f"{recommendation['churned_rate']:.0%} of similar churned customers have "
            f"{recommendation['suggested_alternative']} "
            f"(lift +{recommendation['lift']:.0%}, support {recommendation['support']:.0%}; {quality})."
        )
    return (
        f"{recommendation['retained_rate']:.0%} weighted agreement among similar retained "
        f"customers ({quality}). Too few similar churned customers to compute a lift."
    )


def _render_recommendation(index: int, recommendation: dict):
    color = RISK_COLORS.get(recommendation["confidence"], COLORS["accent"])
    st.markdown(
        f'<div class="dash-rec" style="border-left-color:{color};">'
        f'<p class="dash-rec-title">{index}. {recommendation["action"]} '
        f'{risk_badge(recommendation["confidence"])}</p>'
        f'<p class="dash-rec-meta"><strong>Current:</strong> {recommendation["current_value"]} '
        f'&nbsp;→&nbsp; <strong>Suggested:</strong> {recommendation["suggested_alternative"]}</p>'
        f'<p class="dash-rec-meta"><strong>Model evidence:</strong> raises this customer\'s '
        f'churn risk (contribution {recommendation["model_contribution"]:+.2f})</p>'
        f'<p class="dash-rec-meta"><strong>Similarity evidence:</strong> {_evidence_line(recommendation)}</p>'
        f'<p class="dash-rec-meta"><strong>Priority score:</strong> '
        f'{recommendation["combined_score"]:.2f} &nbsp;·&nbsp; {recommendation["confidence"]} confidence</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_neighbor(comparison: dict):
    with st.expander(f"{comparison['customer_id']} · cosine similarity {comparison['similarity']:.3f}"):
        columns = st.columns(2, gap="medium")
        with columns[0]:
            st.markdown("**Shared attributes**")
            if not comparison["shared"]:
                st.caption("No exact matches on compared fields.")
            for item in comparison["shared"]:
                st.markdown(f"- {item['feature']}: `{item['new_value']}`")
        with columns[1]:
            st.markdown("**Divergences**")
            if not comparison["different"]:
                st.caption("Profiles align on compared fields.")
            for item in comparison["different"]:
                st.markdown(f"- {item['feature']}: `{item['new_value']}` vs `{item['neighbor_value']}`")


def render_retention_plan(analysis: dict):
    """Show similarity-driven recommendations and the neighbors they draw on."""
    plan = analysis["plan"]
    recommendations = plan["recommendations"]

    if plan["evidence_mode"] == "model_only":
        st.warning(
            "No retained customers cleared the similarity threshold. Recommendations "
            "below use model risk factors only."
        )
    elif plan["evidence_mode"] == "majority":
        st.info(
            "Too few similar churned customers to compute a retained-vs-churned lift; "
            "recommendations use agreement among similar retained customers."
        )

    section_label("Prioritized actions")
    if not recommendations:
        st.success(
            "No actions triggered. The model's risk factors for this customer are not "
            "ones the similar retained customers handle differently, so there is no "
            "clear, evidence-backed lever to pull."
        )
    for index, recommendation in enumerate(recommendations[:_VISIBLE_RECOMMENDATIONS], start=1):
        _render_recommendation(index, recommendation)

    remaining = recommendations[_VISIBLE_RECOMMENDATIONS:]
    if remaining:
        with st.expander(f"View {len(remaining)} more recommendation(s)"):
            for offset, recommendation in enumerate(remaining, start=_VISIBLE_RECOMMENDATIONS + 1):
                _render_recommendation(offset, recommendation)

    section_label(
        f"Retained peer set ({plan['retained_count']} retained · {plan['churned_count']} churned compared)"
    )
    if plan["similar_customers"]:
        summary = pd.DataFrame(
            [
                {"Customer ID": item["customer_id"], "Cosine similarity": round(item["similarity"], 3)}
                for item in plan["similar_customers"]
            ]
        )
        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True,
            column_config={"Cosine similarity": st.column_config.NumberColumn(format="%.3f")},
        )
    else:
        st.caption("No qualified retained peers.")

    if analysis["comparisons"]:
        section_label("Peer comparison detail (closest neighbors)")
        for comparison in analysis["comparisons"]:
            _render_neighbor(comparison)

    footnote(plan["disclaimer"])
