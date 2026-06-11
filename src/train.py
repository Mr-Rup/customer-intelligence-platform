"""
Model training module for the Customer Intelligence Platform.

Trains multiple classifiers and serializes the best model.

Can be run as a standalone script for the DVC pipeline:
    python -m src.train
"""

import json
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score

from src.config import (
    FEATURES_FILE,
    BEST_MODEL_FILE,
    METRICS_FILE,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.preprocessing import prepare_data
from src.utils import timer


def get_models() -> dict:
    """
    Return a dictionary of model name -> model instance.
    All hyperparameters are set here (sourced from params.yaml values).
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=2.77,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            auto_class_weights="Balanced",
            random_state=RANDOM_STATE,
            verbose=0,
        ),
    }
    return models


@timer
def train_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """
    Train a single model and return its performance metrics.
    """
    print(f"\n   Training {name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"   {name}: ROC-AUC = {roc_auc:.4f}, F1 = {f1:.4f}")

    return {
        "name": name,
        "model": model,
        "roc_auc": roc_auc,
        "f1": f1,
    }


def train_all_models(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train all models and return results.

    Returns
    -------
    tuple of (results_list, best_model_name, best_model)
    """
    print("\n[MODEL TRAINING]")
    print("=" * 50)

    models = get_models()
    results = []

    for name, model in models.items():
        result = train_model(name, model, X_train, y_train, X_test, y_test)
        results.append(result)

    # Select best model by ROC-AUC
    best = max(results, key=lambda x: x["roc_auc"])
    print(f"\n   ** Best Model: {best['name']} (ROC-AUC = {best['roc_auc']:.4f}) **")

    return results, best["name"], best["model"]


def save_models(results: list, best_name: str, best_model) -> None:
    """
    Save the best model and all metrics to disk.
    """
    # Save best model
    joblib.dump(best_model, BEST_MODEL_FILE)
    print(f"[OK] Saved best model -> {BEST_MODEL_FILE.name}")

    # Save all models individually
    for r in results:
        model_path = MODELS_DIR / f"{r['name'].lower().replace(' ', '_')}.joblib"
        joblib.dump(r["model"], model_path)

    # Save metrics
    metrics = {
        "best_model": best_name,
        "models": {
            r["name"]: {"roc_auc": r["roc_auc"], "f1": r["f1"]}
            for r in results
        },
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved metrics -> {METRICS_FILE.name}")


def run_training_pipeline() -> None:
    """
    Execute the full training pipeline: load -> preprocess -> train -> save.
    """
    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load and preprocess
    df = pd.read_csv(FEATURES_FILE)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)

    # Save scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # Save feature names
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Train all models
    results, best_name, best_model = train_all_models(
        X_train, X_test, y_train, y_test
    )

    # Save
    save_models(results, best_name, best_model)

    # Save test data for evaluation
    X_test.to_csv(MODELS_DIR / "X_test.csv", index=False)
    y_test.to_csv(MODELS_DIR / "y_test.csv", index=False)
    print("[OK] Saved test data for evaluation")


if __name__ == "__main__":
    run_training_pipeline()
