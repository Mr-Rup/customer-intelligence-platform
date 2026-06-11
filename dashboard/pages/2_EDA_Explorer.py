"""
Page 2: EDA Explorer
=====================
Interactive exploration with filters and dynamic charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_INTERIM, DATA_PROCESSED, TARGET

st.set_page_config(page_title="EDA Explorer", page_icon="🔍", layout="wide")
st.title("🔍 EDA Explorer")
st.markdown("Explore customer data with interactive filters and visualizations.")
st.markdown("---")


@st.cache_data
def load_data():
    processed = DATA_PROCESSED / "telco_features.csv"
    interim = DATA_INTERIM / "telco_clean.csv"
    if processed.exists():
        return pd.read_csv(processed)
    elif interim.exists():
        return pd.read_csv(interim)
    st.error("No data found.")
    st.stop()


df = load_data()

# ── Sidebar Filters ──
st.sidebar.markdown("### Filters")

# Contract filter
if "Contract" in df.columns:
    contracts = st.sidebar.multiselect(
        "Contract Type",
        options=df["Contract"].unique(),
        default=df["Contract"].unique(),
    )
    df = df[df["Contract"].isin(contracts)]

# Internet service filter
if "Internet Service" in df.columns:
    internet = st.sidebar.multiselect(
        "Internet Service",
        options=df["Internet Service"].unique(),
        default=df["Internet Service"].unique(),
    )
    df = df[df["Internet Service"].isin(internet)]

# Tenure range
if "Tenure Months" in df.columns:
    tenure_range = st.sidebar.slider(
        "Tenure (Months)",
        min_value=int(df["Tenure Months"].min()),
        max_value=int(df["Tenure Months"].max()),
        value=(int(df["Tenure Months"].min()), int(df["Tenure Months"].max())),
    )
    df = df[(df["Tenure Months"] >= tenure_range[0]) & (df["Tenure Months"] <= tenure_range[1])]

# Monthly charges range
if "Monthly Charges" in df.columns:
    charge_range = st.sidebar.slider(
        "Monthly Charges ($)",
        min_value=float(df["Monthly Charges"].min()),
        max_value=float(df["Monthly Charges"].max()),
        value=(float(df["Monthly Charges"].min()), float(df["Monthly Charges"].max())),
    )
    df = df[(df["Monthly Charges"] >= charge_range[0]) & (df["Monthly Charges"] <= charge_range[1])]

st.sidebar.markdown(f"**Filtered:** {len(df):,} customers")

# ── Analysis Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Services", "Financial", "Correlations"])

with tab1:
    st.markdown("### Demographic Analysis")
    col1, col2 = st.columns(2)

    demo_features = ["Gender", "Senior Citizen", "Partner", "Dependents"]

    for idx, feat in enumerate(demo_features):
        if feat not in df.columns:
            continue
        with col1 if idx % 2 == 0 else col2:
            churn_by_feat = df.groupby(feat)[TARGET].mean().reset_index()
            churn_by_feat.columns = [feat, "Churn Rate"]
            fig = px.bar(
                churn_by_feat, x=feat, y="Churn Rate",
                color="Churn Rate",
                color_continuous_scale="RdYlGn_r",
                text=churn_by_feat["Churn Rate"].apply(lambda x: f"{x:.1%}"),
            )
            fig.update_layout(
                title=f"Churn Rate by {feat}",
                yaxis_tickformat=".0%",
                showlegend=False,
                height=300,
                margin=dict(t=40, b=20),
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Service Analysis")

    service_features = [
        "Internet Service", "Online Security", "Online Backup",
        "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    ]
    available_services = [s for s in service_features if s in df.columns]

    selected_service = st.selectbox("Select Service", available_services)

    if selected_service:
        col1, col2 = st.columns(2)

        with col1:
            service_churn = df.groupby(selected_service)[TARGET].mean().reset_index()
            service_churn.columns = [selected_service, "Churn Rate"]
            fig = px.bar(
                service_churn, x=selected_service, y="Churn Rate",
                color="Churn Rate", color_continuous_scale="RdYlGn_r",
                text=service_churn["Churn Rate"].apply(lambda x: f"{x:.1%}"),
            )
            fig.update_layout(
                title=f"Churn Rate by {selected_service}",
                yaxis_tickformat=".0%", showlegend=False,
                height=350, margin=dict(t=40, b=20),
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            service_counts = df[selected_service].value_counts().reset_index()
            service_counts.columns = [selected_service, "Count"]
            fig = px.pie(
                service_counts, values="Count", names=selected_service,
                hole=0.3,
            )
            fig.update_layout(
                title=f"{selected_service} Distribution",
                height=350, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Financial Analysis")
    col1, col2 = st.columns(2)

    with col1:
        if "Monthly Charges" in df.columns:
            fig = px.box(
                df, x=df[TARGET].map({0: "Retained", 1: "Churned"}),
                y="Monthly Charges",
                color=df[TARGET].map({0: "Retained", 1: "Churned"}),
                color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
            )
            fig.update_layout(
                title="Monthly Charges by Churn Status",
                xaxis_title="", showlegend=False,
                height=400, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Total Charges" in df.columns:
            fig = px.box(
                df, x=df[TARGET].map({0: "Retained", 1: "Churned"}),
                y="Total Charges",
                color=df[TARGET].map({0: "Retained", 1: "Churned"}),
                color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
            )
            fig.update_layout(
                title="Total Charges by Churn Status",
                xaxis_title="", showlegend=False,
                height=400, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Feature Correlations")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if TARGET in num_cols:
        corr = df[num_cols].corr()
        # Show correlation with target
        target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False).reset_index()
        target_corr.columns = ["Feature", "Correlation with Churn"]

        fig = px.bar(
            target_corr, x="Correlation with Churn", y="Feature",
            orientation="h",
            color="Correlation with Churn",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
        )
        fig.update_layout(
            title="Feature Correlation with Churn",
            height=max(400, len(target_corr) * 25),
            margin=dict(t=40, b=20, l=150),
        )
        st.plotly_chart(fig, use_container_width=True)
