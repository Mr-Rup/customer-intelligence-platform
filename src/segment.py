"""
Customer segmentation module for the Customer Intelligence Platform.

Uses K-Means clustering on behavioural and value features
to create business-meaningful customer segments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

from src.config import RANDOM_STATE, FIGSIZE_MEDIUM, FIGSIZE_LARGE, SEGMENT_NAMES


# Features used for segmentation
SEGMENTATION_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "service_count",
]


def prepare_segmentation_features(
    df: pd.DataFrame,
    extra_features: list = None,
) -> tuple:
    """
    Prepare and scale features for clustering.

    Parameters
    ----------
    df : pd.DataFrame
    extra_features : list, optional
        Additional features to include (e.g., churn_probability, clv).

    Returns
    -------
    tuple of (X_scaled, scaler, feature_names)
    """
    features = SEGMENTATION_FEATURES.copy()
    if extra_features:
        features += [f for f in extra_features if f in df.columns]

    # Only include features that exist
    features = [f for f in features if f in df.columns]

    X = df[features].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, features


def find_optimal_clusters(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
) -> tuple:
    """
    Find optimal number of clusters using elbow method and silhouette scores.

    Returns
    -------
    tuple of (inertias, silhouette_scores, optimal_k)
    """
    inertias = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    # Optimal k by silhouette score
    optimal_k = list(k_range)[np.argmax(sil_scores)]

    return inertias, sil_scores, optimal_k


def plot_elbow_silhouette(
    k_range: range,
    inertias: list,
    sil_scores: list,
    optimal_k: int,
) -> plt.Figure:
    """
    Plot elbow curve and silhouette scores side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    # Elbow
    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (k)", fontsize=12)
    axes[0].set_ylabel("Inertia", fontsize=12)
    axes[0].set_title("Elbow Method", fontsize=14, fontweight="bold")

    # Silhouette
    axes[1].plot(list(k_range), sil_scores, "ro-", linewidth=2, markersize=8)
    axes[1].axvline(x=optimal_k, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Number of Clusters (k)", fontsize=12)
    axes[1].set_ylabel("Silhouette Score", fontsize=12)
    axes[1].set_title("Silhouette Analysis", fontsize=14, fontweight="bold")
    axes[1].annotate(
        f"Optimal k={optimal_k}",
        xy=(optimal_k, sil_scores[optimal_k - list(k_range)[0]]),
        fontsize=11, fontweight="bold",
    )

    fig.tight_layout()
    return fig


def fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
) -> tuple:
    """
    Fit K-Means clustering.

    Returns
    -------
    tuple of (kmeans_model, labels)
    """
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"[INFO] K-Means (k={n_clusters}): Silhouette Score = {sil:.4f}")
    return km, labels


def assign_segment_names(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Assign business-meaningful segment names based on cluster characteristics.

    The naming logic examines cluster centroids on key dimensions
    (Tenure, Charges, Service Count) to assign appropriate labels.
    """
    df = df.copy()
    df["cluster"] = labels

    # Compute cluster profiles
    profiles = df.groupby("cluster").agg({
        "Tenure Months": "mean",
        "Monthly Charges": "mean",
        "Total Charges": "mean",
    })

    if "service_count" in df.columns:
        profiles["service_count"] = df.groupby("cluster")["service_count"].mean()
    if "churn_probability" in df.columns:
        profiles["churn_probability"] = df.groupby("cluster")["churn_probability"].mean()

    # Sort clusters by Total Charges (proxy for value) descending
    profiles = profiles.sort_values("Total Charges", ascending=False)

    # Assign names based on ranking
    name_map = {}
    sorted_clusters = profiles.index.tolist()

    default_names = list(SEGMENT_NAMES.values())
    for i, cluster_id in enumerate(sorted_clusters):
        if i < len(default_names):
            name_map[cluster_id] = default_names[i]
        else:
            name_map[cluster_id] = f"Segment {cluster_id}"

    df["segment"] = df["cluster"].map(name_map)
    print(f"[INFO] Segments assigned: {df['segment'].value_counts().to_dict()}")

    return df


def segment_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for each customer segment.
    """
    agg_dict = {
        "Tenure Months": ["mean", "median"],
        "Monthly Charges": ["mean", "median"],
        "Total Charges": ["mean", "sum"],
    }

    # Add optional columns if they exist
    if "service_count" in df.columns:
        agg_dict["service_count"] = "mean"
    if "churn_probability" in df.columns:
        agg_dict["churn_probability"] = "mean"
    if "clv" in df.columns:
        agg_dict["clv"] = ["mean", "sum"]

    col = "segment" if "segment" in df.columns else "cluster"
    profiles = df.groupby(col).agg(agg_dict)
    profiles.columns = ["_".join(c).strip("_") for c in profiles.columns]

    # Add customer count
    profiles["customer_count"] = df.groupby(col).size()

    return profiles


def plot_segment_radar(df: pd.DataFrame, features: list = None) -> plt.Figure:
    """
    Plot radar charts for each segment's profile.
    """
    if features is None:
        features = ["Tenure Months", "Monthly Charges", "service_count"]
        features = [f for f in features if f in df.columns]

    col = "segment" if "segment" in df.columns else "cluster"
    segments = df[col].unique()
    n_segments = len(segments)

    # Normalize features
    normalized = df.groupby(col)[features].mean()
    for f in features:
        fmin = normalized[f].min()
        fmax = normalized[f].max()
        if fmax > fmin:
            normalized[f] = (normalized[f] - fmin) / (fmax - fmin)
        else:
            normalized[f] = 0.5

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, n_segments))

    for seg, color in zip(segments, colors):
        values = normalized.loc[seg].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=seg, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=10)
    ax.set_title("Segment Profiles", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    return fig
