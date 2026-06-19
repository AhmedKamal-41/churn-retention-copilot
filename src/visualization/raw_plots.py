"""Plots for the raw-data analysis notebook (Stage 1).

Each function builds one focused chart and returns the matplotlib Figure, so the
notebook stays a thin layer that just calls a function and shows the result.

These functions never change the saved data. Where a numeric chart needs the
text-typed TotalCharges column as numbers, the conversion happens on a local
copy only, so the raw quirk is still visible elsewhere in the notebook.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN

sns.set_theme(style="whitegrid")


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the numeric columns as real numbers, for charting only.

    TotalCharges is text in the raw file because of a few blank rows, so we
    coerce it here; blanks become NaN and are simply dropped per chart.
    """
    return df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")


def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of how many customers churned versus stayed, with percentages."""
    counts = df[TARGET_COLUMN].value_counts()
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value} ({percentages.iloc[i]:.1f}%)", ha="center", va="bottom")
    ax.set_title("Churn distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame) -> plt.Figure:
    """Histograms (with density curves) for each numeric column."""
    numeric = _numeric_frame(df)

    fig, axes = plt.subplots(1, len(NUMERIC_COLUMNS), figsize=(5 * len(NUMERIC_COLUMNS), 4))
    for ax, column in zip(axes, NUMERIC_COLUMNS):
        sns.histplot(numeric[column].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
    fig.tight_layout()
    return fig


def plot_numeric_boxplots(df: pd.DataFrame) -> plt.Figure:
    """Boxplot per numeric column to reveal spread and possible outliers."""
    numeric = _numeric_frame(df)

    fig, axes = plt.subplots(1, len(NUMERIC_COLUMNS), figsize=(5 * len(NUMERIC_COLUMNS), 4))
    for ax, column in zip(axes, NUMERIC_COLUMNS):
        sns.boxplot(y=numeric[column].dropna(), ax=ax)
        ax.set_title(f"Boxplot of {column}")
    fig.tight_layout()
    return fig


def plot_categorical_distributions(df: pd.DataFrame) -> plt.Figure:
    """Grid of count plots, one per categorical column."""
    columns_per_row = 4
    row_count = math.ceil(len(CATEGORICAL_COLUMNS) / columns_per_row)

    fig, axes = plt.subplots(
        row_count, columns_per_row, figsize=(5 * columns_per_row, 4 * row_count)
    )
    axes = axes.flatten()

    for ax, column in zip(axes, CATEGORICAL_COLUMNS):
        order = df[column].value_counts().index
        sns.countplot(data=df, x=column, order=order, ax=ax)
        ax.set_title(column)
        ax.tick_params(axis="x", rotation=30)

    # Hide any leftover empty subplots in the grid.
    for ax in axes[len(CATEGORICAL_COLUMNS):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of correlations among numeric features and churn.

    Churn is encoded as 1 (Yes) / 0 (No) only to include it in the correlation
    view; this encoding stays local to the chart.
    """
    numeric = _numeric_frame(df)
    numeric = numeric.copy()
    numeric[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation: numeric features and churn")
    fig.tight_layout()
    return fig
