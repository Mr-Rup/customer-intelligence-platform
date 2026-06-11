"""
Data loading utilities for the Customer Intelligence Platform.

Provides consistent data loading across notebooks and modules,
with dtype enforcement and geographic data separation.
"""

import pandas as pd
from src.config import (
    RAW_FILE,
    CLEAN_FILE,
    FEATURES_FILE,
    GEOGRAPHIC_COLUMNS,
)


def load_raw() -> pd.DataFrame:
    """
    Load the raw Telco customer churn dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all 33 original columns.
    """
    df = pd.read_csv(RAW_FILE)
    print(f"✅ Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_clean() -> pd.DataFrame:
    """
    Load the cleaned dataset from the interim directory.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with leakage columns removed and types fixed.
    """
    df = pd.read_csv(CLEAN_FILE)
    print(f"✅ Loaded clean data: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_features() -> pd.DataFrame:
    """
    Load the feature-engineered dataset from the processed directory.

    Returns
    -------
    pd.DataFrame
        Dataset with all engineered features ready for modeling.
    """
    df = pd.read_csv(FEATURES_FILE)
    print(f"✅ Loaded features: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_geographic() -> pd.DataFrame:
    """
    Load geographic columns from the raw dataset.

    Geographic features are excluded from the ML pipeline to prevent
    leakage and reduce dimensionality, but they are needed for the
    geographic analysis and Streamlit map.

    Returns
    -------
    pd.DataFrame
        DataFrame with State, City, Zip Code, Latitude, Longitude
        plus CustomerID for joining.
    """
    df = pd.read_csv(RAW_FILE, usecols=["CustomerID"] + GEOGRAPHIC_COLUMNS)
    print(f"✅ Loaded geographic data: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_raw_with_geo_and_target() -> pd.DataFrame:
    """
    Load the raw dataset with geographic columns and target for EDA.

    This is used in notebooks where we need both geographic info
    and the churn target for visualization, but don't need ML features.

    Returns
    -------
    pd.DataFrame
        Full raw dataset.
    """
    return load_raw()
