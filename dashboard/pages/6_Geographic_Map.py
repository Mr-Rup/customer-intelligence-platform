"""
Page 6: Geographic Churn Map
==============================
Interactive map showing regional churn patterns.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW, TARGET

st.set_page_config(page_title="Geographic Map", page_icon="🗺️", layout="wide")
st.title("🗺️ Geographic Churn Map")
st.markdown("Explore regional churn patterns with interactive maps.")
st.markdown("---")


@st.cache_data
def load_geo_data():
    """Load raw data with geographic columns."""
    path = DATA_RAW / "Telco_customer_churn.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df
    return None


df = load_geo_data()

if df is None:
    st.error("No data found.")
    st.stop()

# Ensure required columns exist
required = ["Latitude", "Longitude", "Churn Value"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ── Sidebar Filters ──
st.sidebar.markdown("### Map Filters")

if "State" in df.columns:
    states = st.sidebar.multiselect(
        "Filter by State",
        options=sorted(df["State"].unique()),
        default=[],
    )
    if states:
        df = df[df["State"].isin(states)]

churn_filter = st.sidebar.radio("Show", ["All", "Churned Only", "Retained Only"])
if churn_filter == "Churned Only":
    df = df[df["Churn Value"] == 1]
elif churn_filter == "Retained Only":
    df = df[df["Churn Value"] == 0]

st.sidebar.markdown(f"**Showing:** {len(df):,} customers")

# ── City-Level Churn Map ──
st.markdown("### Churn Heatmap by City")

if "City" in df.columns:
    city_stats = (
        df.groupby(["City", "Latitude", "Longitude"])
        .agg(
            total_customers=("Churn Value", "count"),
            churned=("Churn Value", "sum"),
            churn_rate=("Churn Value", "mean"),
            avg_monthly=("Monthly Charges", "mean"),
        )
        .reset_index()
    )

    # Filter to cities with enough data
    min_customers = st.slider("Min customers per city", 1, 50, 5)
    city_stats = city_stats[city_stats["total_customers"] >= min_customers]

    fig = px.scatter_mapbox(
        city_stats,
        lat="Latitude",
        lon="Longitude",
        size="total_customers",
        color="churn_rate",
        color_continuous_scale="RdYlGn_r",
        hover_name="City",
        hover_data={
            "total_customers": True,
            "churned": True,
            "churn_rate": ":.1%",
            "avg_monthly": ":$.0f",
            "Latitude": False,
            "Longitude": False,
        },
        size_max=20,
        zoom=5,
        mapbox_style="carto-positron",
        title="City-Level Churn Rate",
    )

    fig.update_layout(
        height=600,
        margin=dict(t=40, b=0, l=0, r=0),
        coloraxis_colorbar_title="Churn Rate",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── State-Level Analysis ──
if "State" in df.columns:
    st.markdown("### State-Level Churn Analysis")

    state_stats = (
        df.groupby("State")
        .agg(
            total_customers=("Churn Value", "count"),
            churned=("Churn Value", "sum"),
            churn_rate=("Churn Value", "mean"),
        )
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 10 Highest Churn States**")
        top_states = state_stats.head(10)
        fig = px.bar(
            top_states, x="churn_rate", y="State",
            orientation="h",
            color="churn_rate", color_continuous_scale="Reds",
            text=top_states["churn_rate"].apply(lambda x: f"{x:.1%}"),
            hover_data={"total_customers": True, "churned": True},
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_tickformat=".0%",
            showlegend=False, height=400,
            margin=dict(t=20, b=20),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top 10 Best Retention States**")
        bottom_states = state_stats.tail(10).sort_values("churn_rate", ascending=True)
        fig = px.bar(
            bottom_states, x="churn_rate", y="State",
            orientation="h",
            color="churn_rate", color_continuous_scale="Greens_r",
            text=bottom_states["churn_rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_tickformat=".0%",
            showlegend=False, height=400,
            margin=dict(t=20, b=20),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Full state table
    with st.expander("View Full State Data"):
        state_stats["churn_rate_fmt"] = state_stats["churn_rate"].apply(lambda x: f"{x:.2%}")
        st.dataframe(
            state_stats[["State", "total_customers", "churned", "churn_rate_fmt"]],
            use_container_width=True,
            hide_index=True,
        )
