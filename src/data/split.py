"""Split the cleaned data into train, validation, and test sets (Stage 2).

The split happens before any preprocessing is fitted, so that scalers and
encoders only ever learn from the training data. This is the single rule that
keeps the whole project free of data leakage.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
)

# The model features are the numeric and categorical columns only.
# The customer ID is an identifier, not a feature, so it is excluded here.
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


@dataclass
class DataSplit:
    """The six pieces of a train/validation/test split, kept together by name."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_data(df: pd.DataFrame, seed: int = RANDOM_SEED) -> DataSplit:
    """Stratified 60/20/20 split into training, validation, and test sets.

    The target Churn is encoded as 1 (Yes) / 0 (No). Stratifying keeps the same
    churn ratio in every split, which matters because the classes are imbalanced.
    """
    X = df[FEATURE_COLUMNS]
    y = (df[TARGET_COLUMN] == "Yes").astype(int)

    # First take out the 20% test set.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )

    # Then split the remaining 80% into 60% train and 20% validation.
    # 0.25 of the remaining 80% is 20% of the whole dataset.
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=seed
    )

    return DataSplit(X_train, X_val, X_test, y_train, y_val, y_test)
