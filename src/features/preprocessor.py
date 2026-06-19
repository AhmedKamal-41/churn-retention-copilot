"""The single preprocessing pipeline reused across the whole project (Stage 2).

One fitted ColumnTransformer turns a customer record into a numeric feature
vector. The same fitted object is used for training, validation, test, the
similarity search, and new customers entered in the dashboard, so preprocessing
is never re-implemented anywhere else.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# The two scaling options compared in the Stage 2 notebook.
SCALERS = {"standard": StandardScaler, "robust": RobustScaler}


def build_preprocessor(scaler: str = "standard") -> ColumnTransformer:
    """Build an unfitted preprocessor for the churn features.

    Numeric columns are median-imputed and scaled. Categorical columns are
    imputed with the most frequent value and one-hot encoded. Unknown categories
    seen later (for example a new value typed in the dashboard) are ignored
    rather than causing an error.

    scaler: "standard" for StandardScaler or "robust" for RobustScaler.
    """
    if scaler not in SCALERS:
        raise ValueError(f"scaler must be one of {list(SCALERS)}, got {scaler!r}")

    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", SCALERS[scaler]()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def transform_to_frame(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Transform X with a fitted preprocessor, keeping readable feature names.

    Returning a DataFrame (instead of a bare array) means later stages never lose
    track of which column means what.
    """
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def save_preprocessor(preprocessor: ColumnTransformer, path) -> None:
    """Save a fitted preprocessor so other stages load the exact same object."""
    joblib.dump(preprocessor, path)


def load_preprocessor(path) -> ColumnTransformer:
    """Load a preprocessor saved by save_preprocessor."""
    return joblib.load(path)
