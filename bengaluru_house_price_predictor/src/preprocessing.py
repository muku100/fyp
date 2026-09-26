"""Preprocessing: turn raw X into numbers for the model.

Numeric: median imputation (bath has 73 missing, balcony 609 missing)
Categorical: most_frequent imputation + OneHotEncoder
    - society: 2688 unique, 41% missing -> first missing becomes 'Unknown',
      then rare societies grouped by min_frequency=10
    - location: 1305 unique -> same rare-grouping
    - handle_unknown='ignore' so a new location at predict-time won't crash

Why no scaling for XGBoost/RF? Trees split on thresholds, not distances.
Ridge (linear) DOES get scaling - see train.py.

Run: from src.preprocessing import make_preprocessor, make_ridge_preprocessor
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def make_ridge_preprocessor() -> ColumnTransformer:
    """Same as above but with scaling for the linear baseline."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
