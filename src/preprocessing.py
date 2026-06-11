"""
Preprocessing module for the Customer Intelligence Platform.

Handles encoding, scaling, train/test splitting, and pipeline
construction for model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (
    TARGET,
    RANDOM_STATE,
    TEST_SIZE,
    BINARY_FEATURES,
    MULTI_CATEGORY_FEATURES,
    NUMERICAL_FEATURES,
)


def get_modeling_features(df: pd.DataFrame) -> list:
    """
    Get the list of features to use for modeling.

    Excludes target, non-numeric engineered categoricals that need
    encoding, and any identifier columns.
    """
    exclude = [TARGET, "tenure_group", "service_bundle"]
    features = [c for c in df.columns if c not in exclude]
    return features


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for modeling.

    - Binary features: Label encoded (0/1)
    - Multi-category features: One-hot encoded
    - tenure_group, service_bundle: One-hot encoded
    """
    df = df.copy()

    # Label encode binary features
    le = LabelEncoder()
    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # One-hot encode multi-category features
    multi_cat = [c for c in MULTI_CATEGORY_FEATURES if c in df.columns]
    # Also encode engineered categoricals
    for extra_cat in ["tenure_group", "service_bundle"]:
        if extra_cat in df.columns:
            multi_cat.append(extra_cat)

    if multi_cat:
        df = pd.get_dummies(df, columns=multi_cat, drop_first=True, dtype=int)

    return df


def split_data(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Split data into train and test sets with stratification.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[INFO] Train set: {X_train.shape[0]:,} samples")
    print(f"[INFO] Test set:  {X_test.shape[0]:,} samples")
    print(f"[INFO] Train churn rate: {y_train.mean():.2%}")
    print(f"[INFO] Test churn rate:  {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: list = None,
) -> tuple:
    """
    Apply StandardScaler to numerical features.

    Returns
    -------
    tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    if numerical_cols is None:
        numerical_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
        # Also scale engineered numerical features
        engineered_num = [
            "service_count", "revenue_per_month", "revenue_intensity",
            "tech_adoption_score", "risk_score", "contract_tenure",
        ]
        numerical_cols += [c for c in engineered_num if c in X_train.columns]

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, scaler


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline: encode -> split -> scale.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print("\n[PREPROCESSING PIPELINE]")
    print("=" * 50)

    # Encode
    df_encoded = encode_features(df)
    print(f"   Encoded features: {len(df_encoded.columns)} columns")

    # Split
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # Scale
    X_train, X_test, scaler = scale_features(X_train, X_test)
    print(f"   Scaled numerical features")

    feature_names = X_train.columns.tolist()
    print(f"   Final feature count: {len(feature_names)}")

    return X_train, X_test, y_train, y_test, scaler, feature_names
