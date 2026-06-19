"""Load the raw Telco churn dataset.

Stage 1 looks at the data exactly as it arrives, so nothing here cleans or
changes the values. Cleaning starts in Stage 2.
"""

import pandas as pd

from src.config import RAW_DATASET_PATH


def load_raw_data() -> pd.DataFrame:
    """Return the raw Telco dataset as a DataFrame, with no changes applied.

    Pandas is told to keep every column as-is. In particular TotalCharges stays
    as text, because some rows hold a blank string, and Stage 1 wants to see
    that quirk rather than hide it.
    """
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATASET_PATH}. "
            "Place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/."
        )
    return pd.read_csv(RAW_DATASET_PATH)
