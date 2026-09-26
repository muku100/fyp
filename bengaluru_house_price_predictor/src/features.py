"""Feature engineering.

Legit feature (uses only inputs):
    sqft_per_bhk = total_sqft / bhk
    -> catches impossible houses (e.g. 5 BHK in 350 sqft)

BANNED feature (target leakage - do NOT use as model input):
    price_per_sqft = price * 100000 / total_sqft
    -> computed FROM price, the thing we predict.
    -> use only for EDA / outlier plots, never in X.

Run: from src.features import add_features, FEATURES, TARGET
"""
import pandas as pd

TARGET = "price"

# V1 feature set - note society IS included (your request)
FEATURES = [
    "area_type",
    "availability",
    "location",
    "society",
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "sqft_per_bhk",
]

NUMERIC_FEATURES = ["total_sqft", "bath", "balcony", "bhk", "sqft_per_bhk"]
CATEGORICAL_FEATURES = ["area_type", "availability", "location", "society"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # avoid divide-by-zero; bhk is >=1 after cleaning
    df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"].replace(0, pd.NA)
    return df


def price_per_sqft_for_eda(df: pd.DataFrame) -> pd.Series:
    """EDA ONLY. Never put this column into X."""
    return df["price"] * 100000 / df["total_sqft"]
