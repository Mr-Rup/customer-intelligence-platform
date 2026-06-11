"""
Model evaluation module for the Customer Intelligence Platform.

Computes comprehensive metrics, generates comparison plots,
and performs threshold optimization.

Can be run as a standalone script for the DVC pipeline:
    python -m src.evaluate
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from src.config import (
    MODELS_DIR,
    BEST_MODEL_FILE,
    EVALUATION_FILE,
    PALETTE_CHURN,
    FIGSIZE_MEDIUM,
    FIGSIZE_LARGE,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Compute all classification metrics.
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
    }


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> plt.Figure:
    """
    Plot a styled confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=ax, linewidths=0.5,
        annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_roc_curves(models_dict: dict, X_test, y_test) -> plt.Figure:
    """
    Plot ROC curves for all models on a single figure.

    Parameters
    ----------
    models_dict : dict
        {model_name: fitted_model}
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_dict)))

    for (name, model), color in zip(models_dict.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    return fig


def plot_precision_recall_curves(models_dict: dict, X_test, y_test) -> plt.Figure:
    """
    Plot Precision-Recall curves for all models.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_dict)))

    for (name, model), color in zip(models_dict.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"{name} (AP = {ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    return fig


def optimize_threshold(y_true, y_prob, metric: str = "f1") -> dict:
    """
    Find the optimal classification threshold.

    Parameters
    ----------
    metric : str
        'f1' for F1-optimal, 'youden' for Youden's J statistic.

    Returns
    -------
    dict
        optimal_threshold, best_score, and metrics at that threshold.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred)
        elif metric == "youden":
            # Youden's J = Sensitivity + Specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sens + spec - 1
        scores.append(score)

    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    # Compute full metrics at best threshold
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    metrics_at_threshold = compute_metrics(y_true, y_pred_opt, y_prob)

    return {
        "optimal_threshold": round(float(best_threshold), 2),
        "best_score": round(float(best_score), 4),
        "metric_optimized": metric,
        "metrics_at_threshold": metrics_at_threshold,
    }


def plot_threshold_analysis(y_true, y_prob) -> plt.Figure:
    """
    Plot how metrics change across classification thresholds.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)

    f1_scores = []
    precisions = []
    recalls = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    ax.plot(thresholds, f1_scores, linewidth=2, label="F1 Score", color="#e74c3c")
    ax.plot(thresholds, precisions, linewidth=2, label="Precision", color="#3498db")
    ax.plot(thresholds, recalls, linewidth=2, label="Recall", color="#2ecc71")

    # Mark optimal F1
    best_idx = np.argmax(f1_scores)
    ax.axvline(x=thresholds[best_idx], color="gray", linestyle="--", alpha=0.7)
    ax.scatter([thresholds[best_idx]], [f1_scores[best_idx]],
               color="#e74c3c", s=100, zorder=5)
    ax.annotate(
        f"Optimal: {thresholds[best_idx]:.2f}",
        xy=(thresholds[best_idx], f1_scores[best_idx]),
        xytext=(thresholds[best_idx] + 0.05, f1_scores[best_idx] + 0.05),
        fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Optimization", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


def model_comparison_table(results: list) -> pd.DataFrame:
    """
    Create a formatted comparison table of all model results.
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r["name"],
            "ROC-AUC": r.get("roc_auc", None),
            "F1": r.get("f1", None),
        })

    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    return df.reset_index(drop=True)


def run_evaluation_pipeline() -> None:
    """
    Load saved models and test data, compute comprehensive evaluation.
    """
    print("=" * 60)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 60)

    # Load test data
    X_test = pd.read_csv(MODELS_DIR / "X_test.csv")
    y_test = pd.read_csv(MODELS_DIR / "y_test.csv").squeeze()

    # Load best model
    best_model = joblib.load(BEST_MODEL_FILE)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    # Compute metrics at default threshold
    metrics_default = compute_metrics(y_test, y_pred, y_prob)
    print("\n[Metrics at threshold = 0.5]")
    for k, v in metrics_default.items():
        print(f"   {k}: {v}")

    # Optimize threshold
    threshold_result = optimize_threshold(y_test, y_prob, metric="f1")
    print(f"\n[Optimal Threshold: {threshold_result['optimal_threshold']}]")
    for k, v in threshold_result["metrics_at_threshold"].items():
        print(f"   {k}: {v}")

    # Save evaluation
    evaluation = {
        "default_threshold": metrics_default,
        "optimized_threshold": threshold_result,
    }
    with open(EVALUATION_FILE, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"\n[OK] Saved evaluation -> {EVALUATION_FILE.name}")


if __name__ == "__main__":
    run_evaluation_pipeline()
