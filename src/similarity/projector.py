"""Project customer vectors to 2D for the customer map (Stage 5).

The map is for visualization only. Cosine similarity and neighbor selection use
the full processed vectors (see search.py); the 2D coordinates here are just a
readable picture of where a customer sits among the others.

Our vectors are dense (scaled numbers plus dense one-hot columns), so PCA is the
right projection. The projector is fit once on the historical vectors, then the
same fitted projector places any new customer on the same map.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.config import RANDOM_SEED
from src.similarity.vectorizer import CustomerVectors

# How each kind of point is drawn. Background customers are light; the new
# customer and its retained neighbors are prominent.
_GROUP_STYLES = {
    "stayed": {"color": "#9ecae1", "alpha": 0.30, "size": 12, "marker": "o", "label": "Stayed"},
    "churned": {"color": "#fdae6b", "alpha": 0.30, "size": 12, "marker": "o", "label": "Churned"},
    "neighbor": {"color": "#08519c", "alpha": 0.95, "size": 110, "marker": "o", "label": "Closest retained"},
    "new": {"color": "#d62728", "alpha": 1.0, "size": 260, "marker": "*", "label": "New customer"},
}


def fit_projector(vectors: np.ndarray, n_components: int = 2) -> PCA:
    """Fit a PCA projection on the historical processed vectors."""
    projector = PCA(n_components=n_components, random_state=RANDOM_SEED)
    projector.fit(vectors)
    return projector


def project(projector: PCA, vectors: np.ndarray) -> np.ndarray:
    """Project vectors into the fitted 2D space."""
    return projector.transform(vectors)


def build_map_data(
    projector: PCA,
    customer_vectors: CustomerVectors,
    new_vector: np.ndarray,
    neighbors: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble a tidy table of 2D points for the map.

    Each row has x, y, customer_id, group ("stayed"/"churned"/"neighbor"/"new"),
    and similarity (filled for neighbors and the new customer). The same table can
    feed a static matplotlib chart now and an interactive Plotly chart later.
    """
    coordinates = project(projector, customer_vectors.vectors)
    points = pd.DataFrame(
        {
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "customer_id": customer_vectors.ids.to_numpy(),
            "group": np.where(customer_vectors.stayed, "stayed", "churned"),
            "similarity": np.nan,
        }
    )

    points.loc[neighbors["row_index"].to_numpy(), "group"] = "neighbor"
    points.loc[neighbors["row_index"].to_numpy(), "similarity"] = neighbors["similarity"].to_numpy()

    new_coordinates = project(projector, new_vector)
    new_point = pd.DataFrame(
        {
            "x": [new_coordinates[0, 0]],
            "y": [new_coordinates[0, 1]],
            "customer_id": ["NEW"],
            "group": ["new"],
            "similarity": [1.0],
        }
    )
    return pd.concat([points, new_point], ignore_index=True)


def plot_customer_map(map_data: pd.DataFrame, draw_neighbor_lines: bool = True) -> plt.Figure:
    """Draw the static 2D customer map from the table built by build_map_data."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for group, style in _GROUP_STYLES.items():
        subset = map_data[map_data["group"] == group]
        ax.scatter(
            subset["x"],
            subset["y"],
            c=style["color"],
            alpha=style["alpha"],
            s=style["size"],
            marker=style["marker"],
            label=style["label"],
            edgecolors="white" if group in ("neighbor", "new") else "none",
            linewidths=0.6,
            zorder=3 if group in ("neighbor", "new") else 1,
        )

    if draw_neighbor_lines:
        new_point = map_data[map_data["group"] == "new"].iloc[0]
        for _, neighbor in map_data[map_data["group"] == "neighbor"].iterrows():
            ax.plot(
                [new_point["x"], neighbor["x"]],
                [new_point["y"], neighbor["y"]],
                color="gray",
                linewidth=0.8,
                linestyle="--",
                alpha=0.6,
                zorder=2,
            )

    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("Customer map (2D projection - similarity uses the full vectors)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
