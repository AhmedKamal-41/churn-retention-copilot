"""Central configuration: file paths, the random seed, and column groupings.

Keeping these in one place means every notebook, test, and future stage refers
to the same names instead of hard-coding strings that drift apart over time.
"""

from pathlib import Path

# Project root is one level above the src/ package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_DATASET_PATH = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Saved Stage 3 artifacts.
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
DECISION_THRESHOLD_PATH = MODELS_DIR / "decision_threshold.json"

# One seed used everywhere so results can be reproduced.
RANDOM_SEED = 42

# Explainability settings (Stage 4).
# Per-customer factors below this absolute SHAP contribution (in log-odds) are
# treated as having no real effect and hidden.
MIN_FACTOR_CONTRIBUTION = 0.01
# A predicted churn probability at or above this is labelled High risk; between
# the operating threshold and this is Medium; below the operating threshold is Low.
RISK_HIGH_THRESHOLD = 0.50

# Similarity settings (Stage 5).
# How many similar retained customers the search returns by default.
TOP_K_NEIGHBORS = 5

# Retention planner settings (Stage 6, similarity-driven).
# A retained customer must be at least this cosine-similar to qualify as a neighbor,
# and we use at most this many of the closest qualified neighbors.
SIMILARITY_MIN_THRESHOLD = 0.80
SIMILARITY_MAX_NEIGHBORS = 20
# A feature's better alternative must reach this similarity-weighted agreement among
# the neighbors before it becomes a recommendation.
MIN_WEIGHTED_AGREEMENT = 0.60
# Monthly charges count as "notably higher" when above the neighbors' weighted
# median by at least this fraction.
MONTHLY_CHARGES_MARGIN = 0.10
# Combined priority score = weighted blend of model strength, neighbor agreement,
# similarity quality, and how actionable the feature is.
PRIORITY_WEIGHTS = {"model": 0.4, "agreement": 0.3, "similarity": 0.2, "actionability": 0.1}
# A SHAP contribution (log-odds) is mapped to roughly 0-1 by dividing by this, then
# clamping, so it can be blended with the other 0-1 parts of the score.
MODEL_CONTRIBUTION_SCALE = 2.0
# Combined-score cut-offs for High / Medium / Low confidence.
CONFIDENCE_HIGH = 0.66
CONFIDENCE_MEDIUM = 0.45

# Lift-based evidence (primary path): compare the similarity-weighted rate of a
# feature's better value among similar RETAINED customers vs similar CHURNED ones.
# Both neighbor groups need at least this many members to compute a lift.
MIN_NEIGHBORS_PER_GROUP = 5
# A better value is recommended when it is at least this much more common among
# retained neighbors than churned ones (a difference of weighted proportions)...
MIN_LIFT = 0.10
# ...and is held by at least this similarity-weighted share of retained neighbors.
MIN_SUPPORT = 0.30
# Lift-mode priority weights: model strength, retained-vs-churned lift, similarity
# quality, and retained support.
LIFT_PRIORITY_WEIGHTS = {"model": 0.35, "lift": 0.30, "similarity": 0.15, "support": 0.20}

# Column roles in the raw Telco dataset.
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

# Columns that hold real numeric measurements.
# Note: in the raw file TotalCharges is stored as text because a few rows hold a
# blank string (customers whose tenure is 0). It is listed here as numeric
# because that is what it represents; the type fix belongs to Stage 2 cleaning.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Everything else describing the customer is categorical.
# SeniorCitizen is stored as 0/1 but means No/Yes, so we treat it as categorical.
CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
