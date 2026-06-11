"""
Page 1: Executive Dashboard
============================
KPIs, churn overview, revenue metrics, and segment distribution.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_INTERIM, DATA_PROCESSED, MODELS_DIR, TARGET


st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")
st.title("📊 Executive Dashboard")
st.markdown("---")


@st.cache_data
def load_data():
    """Load the latest available data."""
    # Try processed first, fall back to interim
    processed = DATA_PROCESSED / "telco_features.csv"
    interim = DATA_INTERIM / "telco_clean.csv"

    if processed.exists():
        df = pd.read_csv(processed)
    elif interim.exists():
        df = pd.read_csv(interim)
    else:
        st.error("No data files found. Please run the pipeline first.")
        st.stop()
    return df


@st.cache_data
def load_metrics():
    """Load model metrics if available."""
    metrics_file = MODELS_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


df = load_data()
metrics = load_metrics()

# ── KPI Cards ──
st.markdown("### Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
churn_rate = df[TARGET].mean()
churned = df[TARGET].sum()
monthly_revenue = df["Monthly Charges"].sum() if "Monthly Charges" in df.columns else 0

with col1:
    st.metric("Total Customers", f"{total_customers:,}")

with col2:
    st.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"-{churned:,} customers", delta_color="inverse")

with col3:
    st.metric("Monthly Revenue", f"${monthly_revenue:,.0f}")

with col4:
    revenue_at_risk = df[df[TARGET] == 1]["Monthly Charges"].sum() if "Monthly Charges" in df.columns else 0
    st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}", delta=f"{revenue_at_risk/monthly_revenue:.1%}" if monthly_revenue > 0 else "N/A", delta_color="inverse")

st.markdown("---")

# ── Charts Row ──
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Churn Distribution")
    churn_counts = df[TARGET].value_counts().reset_index()
    churn_counts.columns = ["Status", "Count"]
    churn_counts["Status"] = churn_counts["Status"].map({0: "Retained", 1: "Churned"})

    fig = px.pie(
        churn_counts, values="Count", names="Status",
        color="Status",
        color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("### Churn by Contract Type")
    if "Contract" in df.columns:
        contract_churn = df.groupby("Contract")[TARGET].agg(["mean", "count"]).reset_index()
        contract_churn.columns = ["Contract", "Churn Rate", "Customers"]

        fig = px.bar(
            contract_churn, x="Contract", y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale="RdYlGn_r",
            text=contract_churn["Churn Rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(
            yaxis_tickformat=".0%",
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Revenue Analysis ──
st.markdown("### Revenue Analysis")
col1, col2 = st.columns(2)

with col1:
    if "Monthly Charges" in df.columns:
        fig = px.histogram(
            df, x="Monthly Charges", color=df[TARGET].map({0: "Retained", 1: "Churned"}),
            barmode="overlay", opacity=0.7,
            color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
            labels={"color": "Status"},
        )
        fig.update_layout(
            title="Monthly Charges Distribution",
            margin=dict(t=40, b=20),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if "Tenure Months" in df.columns:
        fig = px.histogram(
            df, x="Tenure Months", color=df[TARGET].map({0: "Retained", 1: "Churned"}),
            barmode="overlay", opacity=0.7,
            color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
            labels={"color": "Status"},
        )
        fig.update_layout(
            title="Tenure Distribution",
            margin=dict(t=40, b=20),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Model Performance ──
if metrics:
    st.markdown("---")
    st.markdown("### Model Performance")

    best_model = metrics.get("best_model", "N/A")
    st.success(f"**Best Model:** {best_model}")

    model_data = metrics.get("models", {})
    if model_data:
        perf_df = pd.DataFrame([
            {"Model": name, "ROC-AUC": m["roc_auc"], "F1 Score": m["f1"]}
            for name, m in model_data.items()
        ]).sort_values("ROC-AUC", ascending=False)

        st.dataframe(perf_df, use_container_width=True, hide_index=True)
