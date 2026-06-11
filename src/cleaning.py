"""
Data cleaning pipeline for the Customer Intelligence Platform.

This module handles:
- Removing leakage and irrelevant columns
- Type conversions (Total Charges from string to numeric)
- Missing value imputation
- Geographic column separation
- Saving the cleaned dataset to data/interim/

Can be run as a standalone script for the DVC pipeline:
    python -m src.cleaning
"""

import pandas as pd

from src.config import (
    RAW_FILE,
    CLEAN_FILE,
    LEAKAGE_COLUMNS,
    GEOGRAPHIC_COLUMNS,
    TARGET,
)
from src.utils import save_dataframe, timer, df_summary


@timer
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco customer churn dataset.

    Steps:
        1. Drop leakage columns (CustomerID, Count, Lat Long,
           Churn Label, Churn Score, CLTV, Churn Reason, Country).
        2. Drop geographic columns (loaded separately when needed).
        3. Convert Total Charges from string to numeric.
        4. Impute missing Total Charges with 0.0 (new customers
           with tenure = 0 have blank Total Charges).
        5. Convert Senior Citizen from Yes/No to consistent format.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with all 33 columns.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for analysis and feature engineering.
    """
    df = df.copy()

    # Step 1: Drop leakage and irrelevant columns
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"   Dropped {len(cols_to_drop)} leakage/irrelevant columns")

    # Step 2: Drop geographic columns (loaded separately via load_geographic)
    geo_to_drop = [c for c in GEOGRAPHIC_COLUMNS if c in df.columns]
    df = df.drop(columns=geo_to_drop)
    print(f"   Dropped {len(geo_to_drop)} geographic columns (loaded separately)")

    # Step 3: Convert Total Charges to numeric
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    n_coerced = df["Total Charges"].isna().sum()
    print(f"   Converted Total Charges to numeric ({n_coerced} coerced to NaN)")

    # Step 4: Impute missing Total Charges
    # These are new customers with Tenure Months = 0, so Total Charges = 0
    df["Total Charges"] = df["Total Charges"].fillna(0.0)
    print(f"   Imputed {n_coerced} missing Total Charges with 0.0")

    # Step 5: Verify target variable
    assert TARGET in df.columns, f"Target column '{TARGET}' not found!"
    assert df[TARGET].isin([0, 1]).all(), "Target must be binary (0/1)"
    print(f"   Target '{TARGET}': {df[TARGET].value_counts().to_dict()}")

    return df


def run_cleaning_pipeline() -> pd.DataFrame:
    """
    Execute the full cleaning pipeline: load → clean → save.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame (also saved to disk).
    """
    print("=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)

    # Load
    df_raw = pd.read_csv(RAW_FILE)
    df_summary(df_raw, "Raw Data")

    # Clean
    df_clean = clean_data(df_raw)
    df_summary(df_clean, "Cleaned Data")

    # Save
    save_dataframe(df_clean, CLEAN_FILE)

    # Final report
    print(f"\n{'-' * 60}")
    print(f"Columns retained: {list(df_clean.columns)}")
    print(f"{'-' * 60}")

    return df_clean


if __name__ == "__main__":
    run_cleaning_pipeline()
