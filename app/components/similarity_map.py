"""Interactive 2D customer map for the dashboard (Plotly).

Renders the map data built by src/similarity/projector.build_map_data. The chart
is a 2D projection for viewing only; similarity is computed on the full vectors.
"""

import plotly.graph_objects as go
import streamlit as st

from app.components.theme import COLORS

# Background points stay muted; the query account and retained neighbors read clearly.
_STYLE = {
    "stayed": {
        "color": "#94a3b8",
        "size": 5,
        "symbol": "circle",
        "opacity": 0.35,
        "label": "Retained (historical)",
    },
    "churned": {
        "color": "#cbd5e1",
        "size": 5,
        "symbol": "circle",
        "opacity": 0.35,
        "label": "Churned (historical)",
    },
    "neighbor": {
        "color": COLORS["accent"],
        "size": 13,
        "symbol": "circle",
        "opacity": 0.95,
        "label": "Nearest retained peers",
    },
    "new": {
        "color": COLORS["high"],
        "size": 16,
        "symbol": "diamond",
        "opacity": 1.0,
        "label": "Assessed account",
    },
}

_LAYOUT = dict(
    template="plotly_white",
    height=620,
    margin=dict(l=24, r=24, t=48, b=48),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="IBM Plex Sans, sans-serif", size=12, color=COLORS["slate"]),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
    xaxis=dict(
        title="PC1",
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=False,
        linecolor="#e2e8f0",
    ),
    yaxis=dict(
        title="PC2",
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=False,
        linecolor="#e2e8f0",
    ),
)


def build_map_figure(map_data) -> go.Figure:
    """Build the Plotly figure from the map-data table."""
    fig = go.Figure()

    new_point = map_data[map_data["group"] == "new"].iloc[0]
    for _, neighbor in map_data[map_data["group"] == "neighbor"].iterrows():
        fig.add_trace(
            go.Scatter(
                x=[new_point["x"], neighbor["x"]],
                y=[new_point["y"], neighbor["y"]],
                mode="lines",
                line=dict(color="#94a3b8", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    draw_order = ["stayed", "churned", "neighbor", "new"]
    for group in draw_order:
        style = _STYLE[group]
        subset = map_data[map_data["group"] == group]
        if subset.empty:
            continue

        if group == "neighbor":
            customdata = subset[["customer_id", "similarity"]].to_numpy()
            hovertemplate = (
                "<b>%{customdata[0]}</b><br>Cosine similarity: %{customdata[1]:.3f}<extra></extra>"
            )
        elif group == "new":
            customdata = None
            hovertemplate = "<b>Assessed account</b><extra></extra>"
        else:
            customdata = subset[["customer_id"]].to_numpy()
            hovertemplate = "Customer %{customdata[0]}<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                name=style["label"],
                marker=dict(
                    color=style["color"],
                    size=style["size"],
                    symbol=style["symbol"],
                    opacity=style["opacity"],
                    line=dict(color="#ffffff", width=1.5) if group in ("neighbor", "new") else dict(width=0),
                ),
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

    fig.update_layout(**_LAYOUT)
    fig.update_layout(title=dict(text="Customer embedding — PCA projection", font=dict(size=14)))
    return fig


def render_similarity_map(analysis: dict):
    """Render the interactive map for an analysed customer."""
    fig = build_map_figure(analysis["map_data"])
    st.plotly_chart(fig, use_container_width=True)
