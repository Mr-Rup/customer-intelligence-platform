"""
SHAP explainability module for the Customer Intelligence Platform.

Provides global and local model explanations using SHAP
(SHapley Additive exPlanations).
"""

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import BEST_MODEL_FILE, MODELS_DIR, FIGSIZE_MEDIUM, FIGSIZE_LARGE


def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 1000):
    """
    Compute SHAP values using TreeExplainer (for tree-based models)
    or KernelExplainer (fallback).

    Parameters
    ----------
    model : fitted model
    X : pd.DataFrame
        Feature matrix.
    max_samples : int
        Max samples for SHAP computation (for speed).

    Returns
    -------
    shap.Explanation
    """
    X_sample = X.sample(n=min(max_samples, len(X)), random_state=42)

    try:
        # TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
    except Exception:
        # Fallback to KernelExplainer for non-tree models
        explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:100])
        shap_values = explainer(X_sample)

    return shap_values, X_sample


def plot_shap_summary(shap_values, X_sample: pd.DataFrame,
                      max_display: int = 20) -> plt.Figure:
    """
    Plot SHAP summary (beeswarm) plot showing global feature importance.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
    # Handle multi-output SHAP values (take class 1 = churn)
    sv = shap_values
    if hasattr(sv, 'values') and sv.values.ndim == 3:
        sv = sv[:, :, 1]
    shap.summary_plot(sv, X_sample, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance (Global)", fontsize=14, fontweight="bold")
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_bar(shap_values, X_sample: pd.DataFrame,
                  max_display: int = 15) -> plt.Figure:
    """
    Plot SHAP bar chart of mean absolute SHAP values.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    sv = shap_values
    if hasattr(sv, 'values') and sv.values.ndim == 3:
        sv = sv[:, :, 1]
    shap.plots.bar(sv, max_display=max_display, show=False)
    plt.title("Mean |SHAP| - Top Churn Drivers", fontsize=14, fontweight="bold")
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_waterfall(shap_values, idx: int = 0) -> plt.Figure:
    """
    Plot SHAP waterfall for a single customer (local explanation).

    Parameters
    ----------
    shap_values : shap.Explanation
    idx : int
        Index of the customer in the SHAP values array.
    """
    sv = shap_values
    if hasattr(sv, 'values') and sv.values.ndim == 3:
        sv = sv[:, :, 1]
    fig = plt.figure(figsize=FIGSIZE_MEDIUM)
    shap.plots.waterfall(sv[idx], show=False)
    plt.title(f"SHAP Explanation - Customer {idx}", fontsize=14, fontweight="bold")
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_dependence(shap_values, X_sample: pd.DataFrame,
                         feature: str, interaction: str = None) -> plt.Figure:
    """
    Plot SHAP dependence plot for a specific feature.
    """
    sv = shap_values
    if hasattr(sv, 'values') and sv.values.ndim == 3:
        sv = sv[:, :, 1]
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    shap.dependence_plot(
        feature, sv.values, X_sample,
        interaction_index=interaction,
        ax=ax, show=False,
    )
    ax.set_title(f"SHAP Dependence: {feature}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def get_top_drivers(shap_values, X_sample: pd.DataFrame,
                    idx: int, top_n: int = 5) -> pd.DataFrame:
    """
    Get the top N SHAP drivers for a specific customer.

    Returns
    -------
    pd.DataFrame
        Columns: Feature, Value, SHAP Value, Direction
    """
    sv = shap_values
    if hasattr(sv, 'values') and sv.values.ndim == 3:
        sv = sv[:, :, 1]

    feature_names = X_sample.columns.tolist()
    shap_vals = sv.values[idx]
    feature_vals = X_sample.iloc[idx].values

    df = pd.DataFrame({
        "Feature": feature_names,
        "Value": feature_vals,
        "SHAP Value": shap_vals,
        "Direction": ["Increases Churn" if s > 0 else "Decreases Churn"
                      for s in shap_vals],
    })
    df["abs_shap"] = df["SHAP Value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n)
    return df.drop(columns=["abs_shap"]).reset_index(drop=True)


def explain_customer(model, X_customer: pd.DataFrame) -> dict:
    """
    Generate a full explanation for a single customer.

    Parameters
    ----------
    model : fitted model
    X_customer : pd.DataFrame
        Single-row DataFrame with the customer's features.

    Returns
    -------
    dict
        churn_probability, risk_level, top_drivers
    """
    prob = model.predict_proba(X_customer)[:, 1][0]

    if prob >= 0.7:
        risk = "High"
    elif prob >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    # SHAP for this customer
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_customer)
        if hasattr(sv, 'values') and sv.values.ndim == 3:
            sv_vals = sv.values[0, :, 1]
        else:
            sv_vals = sv.values[0]

        features = X_customer.columns.tolist()
        top_idx = np.argsort(np.abs(sv_vals))[-5:][::-1]
        top_drivers = [
            {
                "feature": features[i],
                "shap_value": round(float(sv_vals[i]), 4),
                "direction": "Increases Churn" if sv_vals[i] > 0 else "Decreases Churn",
            }
            for i in top_idx
        ]
    except Exception:
        top_drivers = []

    return {
        "churn_probability": round(float(prob), 4),
        "risk_level": risk,
        "top_drivers": top_drivers,
    }
