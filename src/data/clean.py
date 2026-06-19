"""Cleaning for the raw Telco churn dataset (Stage 2).

The only real data-quality fix this dataset needs is the TotalCharges column,
which arrives as text with a few blank values. Everything else in the raw file
is already consistent (no duplicates, no impossible values), as Stage 1 showed.
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataset.

    TotalCharges arrives as text because 11 brand-new customers (tenure 0) have a
    blank total - they have not been billed a full cycle yet. We convert the
    column to numbers and set those blanks to 0, which is the correct total for a
    customer who has not been charged yet.

    The customer ID and the target are left untouched here; selecting feature
    columns and encoding the target happen in the split step.
    """
    cleaned = df.copy()
    total_charges = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["TotalCharges"] = total_charges.fillna(0.0)
    return cleaned
