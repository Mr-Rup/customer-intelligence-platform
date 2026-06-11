"""
Feature engineering module for the Customer Intelligence Platform.

Creates derived features from the cleaned dataset to improve
model performance and provide business-meaningful inputs.

Can be run as a standalone script for the DVC pipeline:
    python -m src.feature_engineering
"""

import pandas as pd
import numpy as np

from src.config import (
    CLEAN_FILE,
    FEATURES_FILE,
    TARGET,
)
from src.utils import save_dataframe, timer, df_summary


def create_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure-based features.

    - tenure_group: Categorical bucket (0-12, 12-24, 24-48, 48+)
    - is_new_customer: Binary flag for customers with < 12 months tenure
    """
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["Tenure Months"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12", "12-24", "24-48", "48+"],
    )

    df["is_new_customer"] = (df["Tenure Months"] < 12).astype(int)

    return df


def create_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of subscribed services per customer.

    Services counted: Phone Service, Multiple Lines, Online Security,
    Online Backup, Device Protection, Tech Support, Streaming TV,
    Streaming Movies.
    """
    df = df.copy()

    service_cols = [
        "Phone Service", "Multiple Lines", "Online Security",
        "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies",
    ]

    # Count "Yes" values across service columns
    existing_cols = [c for c in service_cols if c in df.columns]
    df["service_count"] = df[existing_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    return df


def create_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create revenue-based features.

    - revenue_per_month: Total Charges / Tenure Months
      (inf for tenure=0 customers, replaced with Monthly Charges)
    - revenue_intensity: Ratio indicating spending pattern consistency
    """
    df = df.copy()

    # Revenue per month (handle division by zero)
    df["revenue_per_month"] = np.where(
        df["Tenure Months"] > 0,
        df["Total Charges"] / df["Tenure Months"],
        df["Monthly Charges"],
    )

    # Revenue intensity: how close actual spend is to expected
    expected = df["Monthly Charges"] * df["Tenure Months"]
    df["revenue_intensity"] = np.where(
        expected > 0,
        df["Total Charges"] / expected,
        1.0,
    )

    return df


def create_customer_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer profile features.

    - has_family: 1 if customer has Partner OR Dependents
    - tech_adoption_score: Count of tech-related services
    - service_bundle_score: Categorical bundle tier (Basic/Standard/Premium)
    """
    df = df.copy()

    # Family status
    df["has_family"] = (
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
    ).astype(int)

    # Tech adoption: count of tech-oriented services
    tech_services = ["Online Security", "Online Backup", "Device Protection",
                     "Tech Support"]
    existing_tech = [c for c in tech_services if c in df.columns]
    df["tech_adoption_score"] = df[existing_tech].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    # Service bundle score
    if "service_count" in df.columns:
        df["service_bundle"] = pd.cut(
            df["service_count"],
            bins=[-1, 1, 3, 5, np.inf],
            labels=["Basic", "Standard", "Premium", "All-In"],
        )
    else:
        df["service_bundle"] = "Unknown"

    return df


def create_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary risk indicator flags.

    - is_month_to_month: 1 if contract is Month-to-month
    - is_fiber: 1 if Internet Service is Fiber optic
    - is_electronic_check: 1 if Payment Method is Electronic check
    - risk_score: Composite risk score (sum of risk indicators)
    """
    df = df.copy()

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_fiber"] = (df["Internet Service"] == "Fiber optic").astype(int)
    df["is_electronic_check"] = (df["Payment Method"] == "Electronic check").astype(int)

    # Composite risk score
    df["risk_score"] = (
        df["is_month_to_month"]
        + df["is_fiber"]
        + df["is_electronic_check"]
        + df["is_new_customer"]  # requires tenure features first
    )

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between key variables.

    These capture non-linear relationships that individual features miss.
    """
    df = df.copy()

    # Contract x Tenure interaction
    df["contract_tenure"] = (
        df["is_month_to_month"] * (72 - df["Tenure Months"])
    )

    # Internet x Security gap (fiber without security)
    if "Online Security" in df.columns:
        df["security_gap"] = (
            (df["is_fiber"] == 1) &
            (df["Online Security"] == "No")
        ).astype(int)

    # High charge + low tenure (potential flight risk)
    median_charge = df["Monthly Charges"].median()
    df["high_charge_new_customer"] = (
        (df["Monthly Charges"] > median_charge) &
        (df["is_new_customer"] == 1)
    ).astype(int)

    return df


@timer
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Order matters: tenure features must come before risk indicators.
    """
    print("\n[FEATURE ENGINEERING PIPELINE]")
    print("=" * 50)

    n_original = len(df.columns)

    df = create_tenure_features(df)
    print(f"   + Tenure features (tenure_group, is_new_customer)")

    df = create_service_count(df)
    print(f"   + Service count")

    df = create_revenue_features(df)
    print(f"   + Revenue features (revenue_per_month, revenue_intensity)")

    df = create_customer_profile_features(df)
    print(f"   + Customer profile (has_family, tech_adoption, service_bundle)")

    df = create_risk_indicators(df)
    print(f"   + Risk indicators (m2m, fiber, e-check, risk_score)")

    df = create_interaction_features(df)
    print(f"   + Interaction features")

    n_new = len(df.columns) - n_original
    print(f"\n   Total new features created: {n_new}")
    print(f"   Final feature count: {len(df.columns)}")

    return df


def run_feature_pipeline() -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline: load clean -> engineer -> save.
    """
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # Load cleaned data
    df = pd.read_csv(CLEAN_FILE)
    df_summary(df, "Clean Data (Input)")

    # Engineer features
    df = engineer_features(df)
    df_summary(df, "Feature-Engineered Data (Output)")

    # Save
    save_dataframe(df, FEATURES_FILE)

    return df


if __name__ == "__main__":
    run_feature_pipeline()
