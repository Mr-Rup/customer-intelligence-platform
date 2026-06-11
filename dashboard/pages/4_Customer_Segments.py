"""
Page 4: Customer Segments
==========================
Explore customer segments with profiles and comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED, DATA_INTERIM, TARGET

st.set_page_config(page_title="Customer Segments", page_icon="👥", layout="wide")
st.title("👥 Customer Segments")
st.markdown("Explore customer clusters and their behavioral profiles.")
st.markdown("---")


@st.cache_data
def load_data():
    processed = DATA_PROCESSED / "telco_features.csv"
    interim = DATA_INTERIM / "telco_clean.csv"
    if processed.exists():
        df = pd.read_csv(processed)
    elif interim.exists():
        df = pd.read_csv(interim)
    else:
        st.error("No data found.")
        st.stop()
    return df


df = load_data()

# Create segments if not already present
if "segment" not in df.columns:
    # Create simple segments based on available features
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    seg_features = ["Tenure Months", "Monthly Charges"]
    if "Total Charges" in df.columns:
        seg_features.append("Total Charges")
    if "service_count" in df.columns:
        seg_features.append("service_count")

    X_seg = df[seg_features].fillna(df[seg_features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # Name segments by value
    profiles = df.groupby("cluster")["Total Charges"].mean().sort_values(ascending=False)
    names = ["VIP Loyal", "VIP At Risk", "New Customers", "Low Value", "Growth Potential"]
    name_map = {cluster: names[i] for i, cluster in enumerate(profiles.index)}
    df["segment"] = df["cluster"].map(name_map)

# ── Segment Overview ──
st.markdown("### Segment Overview")

segments = df["segment"].unique()

cols = st.columns(len(segments))
for i, seg in enumerate(segments):
    seg_data = df[df["segment"] == seg]
    with cols[i % len(cols)]:
        st.markdown(f"**{seg}**")
        st.metric("Customers", f"{len(seg_data):,}")
        st.metric("Churn Rate", f"{seg_data[TARGET].mean():.1%}")
        if "Monthly Charges" in seg_data.columns:
            st.metric("Avg Monthly", f"${seg_data['Monthly Charges'].mean():.0f}")

st.markdown("---")

# ── Segment Comparison ──
st.markdown("### Segment Comparison")

tab1, tab2, tab3 = st.tabs(["Distribution", "Profiles", "Details"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        seg_counts = df["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.pie(seg_counts, values="Count", names="Segment", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title="Segment Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_churn = df.groupby("segment")[TARGET].mean().reset_index()
        seg_churn.columns = ["Segment", "Churn Rate"]
        fig = px.bar(seg_churn.sort_values("Churn Rate", ascending=False),
                     x="Segment", y="Churn Rate",
                     color="Churn Rate", color_continuous_scale="RdYlGn_r",
                     text=seg_churn.sort_values("Churn Rate", ascending=False)["Churn Rate"].apply(lambda x: f"{x:.1%}"))
        fig.update_layout(title="Churn Rate by Segment", yaxis_tickformat=".0%",
                          showlegend=False, height=400)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Radar chart
    profile_features = ["Tenure Months", "Monthly Charges"]
    if "service_count" in df.columns:
        profile_features.append("service_count")
    if "Total Charges" in df.columns:
        profile_features.append("Total Charges")

    profiles = df.groupby("segment")[profile_features].mean()

    # Normalize
    for col in profile_features:
        col_min = profiles[col].min()
        col_max = profiles[col].max()
        if col_max > col_min:
            profiles[col] = (profiles[col] - col_min) / (col_max - col_min)

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, (seg, row) in enumerate(profiles.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=row.values.tolist() + [row.values[0]],
            theta=profile_features + [profile_features[0]],
            fill="toself",
            name=seg,
            opacity=0.6,
            line_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Segment Profiles (Normalized)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Detailed segment table
    selected_segment = st.selectbox("Select Segment", sorted(df["segment"].unique()))
    seg_data = df[df["segment"] == selected_segment]

    st.markdown(f"### {selected_segment} Segment Details")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", f"{len(seg_data):,}")
    with col2:
        st.metric("Churn Rate", f"{seg_data[TARGET].mean():.1%}")
    with col3:
        st.metric("Avg Tenure", f"{seg_data['Tenure Months'].mean():.0f} months")
    with col4:
        st.metric("Avg Monthly", f"${seg_data['Monthly Charges'].mean():.0f}")

    # Show data
    st.dataframe(
        seg_data.head(50),
        use_container_width=True,
        hide_index=True,
    )
