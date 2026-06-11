"""
Customer Intelligence Platform - Streamlit Dashboard
=====================================================

Main application entry point with multipage navigation.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
from pathlib import Path

# ── Page Configuration ──
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ──
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar Branding ──
with st.sidebar:
    st.markdown("# Customer Intelligence Platform")
    st.markdown("---")
    st.markdown(
        """
        **Navigation**

        Use the sidebar to explore different sections
        of the Customer Intelligence Platform.

        ---
        *Built with Streamlit*
        """
    )

# ── Landing Page ──
st.title("Customer Intelligence Platform")
st.markdown("#### End-to-End Analytics for Customer Retention")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        ### Predict
        Identify customers at risk of churning using
        machine learning models trained on behavioral data.
        """
    )

with col2:
    st.markdown(
        """
        ### Understand
        Explain why customers churn with SHAP-based
        model interpretability and statistical analysis.
        """
    )

with col3:
    st.markdown(
        """
        ### Act
        Get actionable retention recommendations
        tailored to each customer segment and risk level.
        """
    )

st.markdown("---")

st.info(
    "Navigate to different pages using the sidebar. "
    "Start with the **Executive Dashboard** for a high-level overview."
)
