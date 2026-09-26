"""Data cleaning for Bengaluru house prices.

What this solves:
- total_sqft has '2100 - 2850' ranges -> convert to average (2475)
- size has '2 BHK', '4 Bedroom', '1 RK' -> extract number (2, 4, 1)
- column names may have extra spaces -> strip them
- duplicate rows -> drop them

Run: from src.data_cleaning import clean_dataframe
"""
import re
import numpy as np
import pandas as pd


def convert_sqft(value) -> float:
    """'1056' -> 1056.0, '2100 - 2850' -> 2475.0, junk -> nan."""
    s = str(value).strip()
    if "-" in s:
        try:
            low, high = s.split("-")
            return (float(low.strip()) + float(high.strip())) / 2
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def extract_bhk(value) -> float:
    """'2 BHK' -> 2, '4 Bedroom' -> 4, '1 RK' -> 1, missing -> nan."""
    m = re.search(r"\d+", str(value))
    if m:
        return int(m.group())
    return np.nan


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. clean column names: "area_type " vs "area_type"
    df.columns = df.columns.str.strip()

    # 2. strip whitespace in text cols (society='Coomee ' vs 'Coomee')
    for col in ["area_type", "availability", "location", "size", "society"]:
        if col in df.columns:
            df[col] = df[col].astype("object").str.strip()
            # empty string -> NaN so imputer treats it as missing
            df[col] = df[col].replace("", np.nan)

    # 3. drop exact duplicate rows (your data has ~529)
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. parse total_sqft and bhk
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
    df["bhk"] = df["size"].apply(extract_bhk)

    # 5. original 'size' is now redundant -> drop it
    #    keep 'society' (you asked to keep it) - we handle it in preprocessing
    if "size" in df.columns:
        df = df.drop(columns=["size"])

    # 6. drop rows where we have no target or no core features
    #    (can't train if price / sqft / bhk is unknown)
    df = df.dropna(subset=["price", "total_sqft", "bhk"]).reset_index(drop=True)

    return df
