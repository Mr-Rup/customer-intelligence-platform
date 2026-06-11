"""
Reusable EDA plotting functions for the Customer Intelligence Platform.

All visualization functions follow a consistent pattern:
- Accept a DataFrame and optional parameters
- Return a matplotlib Figure (or display in-place for notebooks)
- Use the project color palette from config.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PALETTE_CHURN,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    FIGSIZE_SMALL,
    FIGSIZE_MEDIUM,
    FIGSIZE_LARGE,
    FIGSIZE_WIDE,
    TARGET,
)

# Set consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_churn_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Plot the overall churn distribution as bar + pie charts.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    # Bar chart
    churn_counts = df[TARGET].value_counts()
    labels = ["Retained", "Churned"]
    colors = [PALETTE_CHURN[0], PALETTE_CHURN[1]]

    axes[0].bar(labels, churn_counts.values, color=colors, edgecolor="white", width=0.6)
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold", fontsize=12)
    axes[0].set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")

    # Pie chart
    axes[1].pie(
        churn_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0, 0.05),
        textprops={"fontsize": 12},
    )
    axes[1].set_title("Churn Rate", fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_categorical_vs_churn(
    df: pd.DataFrame,
    column: str,
    title: str = None,
    figsize: tuple = None,
) -> plt.Figure:
    """
    Plot a categorical feature against churn with grouped bars and churn rate line.
    """
    if figsize is None:
        figsize = FIGSIZE_MEDIUM
    if title is None:
        title = f"{column} vs Churn"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Grouped bar chart
    ct = pd.crosstab(df[column], df[TARGET])
    ct.columns = ["Retained", "Churned"]
    ct.plot(kind="bar", ax=axes[0], color=[PALETTE_CHURN[0], PALETTE_CHURN[1]],
            edgecolor="white", rot=0)
    axes[0].set_title(f"{title} - Counts", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=True)

    # Churn rate by category
    churn_rate = df.groupby(column)[TARGET].mean().sort_values(ascending=False)
    bars = axes[1].bar(
        range(len(churn_rate)),
        churn_rate.values,
        color=COLOR_PRIMARY,
        edgecolor="white",
        width=0.6,
    )
    axes[1].set_xticks(range(len(churn_rate)))
    axes[1].set_xticklabels(churn_rate.index, rotation=0 if len(churn_rate) <= 5 else 45,
                             ha="right" if len(churn_rate) > 5 else "center")
    axes[1].set_title(f"{title} - Churn Rate", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Churn Rate")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Add value labels
    for bar, val in zip(bars, churn_rate.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.1%}", ha="center", fontsize=10,
        )

    fig.suptitle("", fontsize=0)  # Remove default suptitle space
    fig.tight_layout()
    return fig


def plot_numerical_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
) -> plt.Figure:
    """
    Plot numerical feature distribution split by churn status.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    # Histogram with KDE
    for label, color in PALETTE_CHURN.items():
        subset = df[df[TARGET] == label][column]
        tag = "Churned" if label == 1 else "Retained"
        axes[0].hist(subset, bins=bins, alpha=0.5, color=color, label=tag, density=True)
        subset.plot.kde(ax=axes[0], color=color, linewidth=2)
    axes[0].set_title(f"{column} Distribution by Churn", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Box plot
    data_retained = df[df[TARGET] == 0][column]
    data_churned = df[df[TARGET] == 1][column]
    bp = axes[1].boxplot(
        [data_retained, data_churned],
        labels=["Retained", "Churned"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], [PALETTE_CHURN[0], PALETTE_CHURN[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title(f"{column} Box Plot by Churn", fontsize=13, fontweight="bold")
    axes[1].set_ylabel(column)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, columns: list = None) -> plt.Figure:
    """
    Plot a correlation heatmap for numerical columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_service_analysis(df: pd.DataFrame) -> plt.Figure:
    """
    Plot churn rate by service subscription.
    """
    service_cols = [
        "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies",
    ]
    # Only include columns that exist
    service_cols = [c for c in service_cols if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()

    for idx, col in enumerate(service_cols):
        churn_rate = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
        colors = [PALETTE_CHURN[1] if v > df[TARGET].mean() else COLOR_PRIMARY
                  for v in churn_rate.values]
        axes[idx].bar(range(len(churn_rate)), churn_rate.values,
                      color=colors, edgecolor="white", width=0.5)
        axes[idx].set_xticks(range(len(churn_rate)))
        axes[idx].set_xticklabels(churn_rate.index, fontsize=9)
        axes[idx].set_title(col, fontsize=12, fontweight="bold")
        axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        axes[idx].axhline(y=df[TARGET].mean(), color="gray", linestyle="--",
                          alpha=0.7, linewidth=1)

    fig.suptitle("Churn Rate by Service Subscription", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_contract_analysis(df: pd.DataFrame) -> plt.Figure:
    """
    Plot contract, billing, and payment method vs churn.
    """
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)

    contract_cols = ["Contract", "Paperless Billing", "Payment Method"]
    for idx, col in enumerate(contract_cols):
        if col not in df.columns:
            continue
        churn_rate = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
        colors = [PALETTE_CHURN[1] if v > df[TARGET].mean() else PALETTE_CHURN[0]
                  for v in churn_rate.values]
        bars = axes[idx].barh(range(len(churn_rate)), churn_rate.values,
                              color=colors, edgecolor="white", height=0.5)
        axes[idx].set_yticks(range(len(churn_rate)))
        axes[idx].set_yticklabels(churn_rate.index, fontsize=10)
        axes[idx].set_title(col, fontsize=13, fontweight="bold")
        axes[idx].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        axes[idx].axvline(x=df[TARGET].mean(), color="gray", linestyle="--", alpha=0.7)

        for bar, val in zip(bars, churn_rate.values):
            axes[idx].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                           f"{val:.1%}", va="center", fontsize=10)

    fig.suptitle("Contract & Payment Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_financial_analysis(df: pd.DataFrame) -> plt.Figure:
    """
    Plot Monthly Charges and Total Charges distributions by churn.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

    for idx, col in enumerate(["Monthly Charges", "Total Charges"]):
        if col not in df.columns:
            continue
        # Histogram
        for label, color in PALETTE_CHURN.items():
            subset = df[df[TARGET] == label][col]
            tag = "Churned" if label == 1 else "Retained"
            axes[0, idx].hist(subset, bins=30, alpha=0.5, color=color,
                              label=tag, density=True)
        axes[0, idx].set_title(f"{col} Distribution", fontsize=12, fontweight="bold")
        axes[0, idx].legend()
        axes[0, idx].set_xlabel(col)

        # Box plot
        data = [df[df[TARGET] == 0][col], df[df[TARGET] == 1][col]]
        bp = axes[1, idx].boxplot(data, labels=["Retained", "Churned"],
                                   patch_artist=True)
        for patch, color in zip(bp["boxes"], [PALETTE_CHURN[0], PALETTE_CHURN[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, idx].set_title(f"{col} by Churn", fontsize=12, fontweight="bold")

    fig.suptitle("Financial Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_tenure_analysis(df: pd.DataFrame) -> plt.Figure:
    """
    Plot tenure distribution and survival-style analysis.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    # Tenure histogram by churn
    for label, color in PALETTE_CHURN.items():
        subset = df[df[TARGET] == label]["Tenure Months"]
        tag = "Churned" if label == 1 else "Retained"
        axes[0].hist(subset, bins=72, alpha=0.5, color=color, label=tag)
    axes[0].set_title("Tenure Distribution by Churn", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Tenure (Months)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Churn rate by tenure bucket
    df_temp = df.copy()
    df_temp["Tenure Group"] = pd.cut(
        df_temp["Tenure Months"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48-72"],
    )
    churn_by_tenure = df_temp.groupby("Tenure Group", observed=True)[TARGET].mean()
    axes[1].plot(churn_by_tenure.index, churn_by_tenure.values,
                 marker="o", linewidth=2, color=PALETTE_CHURN[1], markersize=8)
    axes[1].fill_between(range(len(churn_by_tenure)), churn_by_tenure.values,
                         alpha=0.2, color=PALETTE_CHURN[1])
    axes[1].set_title("Churn Rate by Tenure Group", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Tenure Group (Months)")
    axes[1].set_ylabel("Churn Rate")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.tight_layout()
    return fig


def plot_geographic_churn(df: pd.DataFrame, geo_col: str = "State",
                          top_n: int = 15) -> plt.Figure:
    """
    Plot churn rate by geographic region.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the geographic column and Target.
    geo_col : str
        Geographic column to analyze (State, City, etc.)
    top_n : int
        Number of top regions to display.
    """
    churn_by_geo = (
        df.groupby(geo_col)[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Churn Rate", "count": "Customers"})
        .sort_values("Churn Rate", ascending=False)
    )

    # Filter to regions with sufficient sample size
    min_customers = 10
    churn_by_geo = churn_by_geo[churn_by_geo["Customers"] >= min_customers]

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    top = churn_by_geo.head(top_n)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top)))
    bars = ax.barh(range(len(top)), top["Churn Rate"].values,
                   color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_title(f"Top {top_n} {geo_col}s by Churn Rate (min {min_customers} customers)",
                 fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.invert_yaxis()

    for bar, val, count in zip(bars, top["Churn Rate"].values, top["Customers"].values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%} (n={count})", va="center", fontsize=9)

    fig.tight_layout()
    return fig
