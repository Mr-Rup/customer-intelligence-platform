"""
Page 5: Customer Lookup
========================
Look up individual customers and view their full profile.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR,
    BEST_MODEL_FILE, TARGET,
)

st.set_page_config(page_title="Customer Lookup", page_icon="🔎", layout="wide")
st.title("🔎 Customer Lookup")
st.markdown("Search for a customer and view their full intelligence profile.")
st.markdown("---")


@st.cache_data
def load_raw():
    path = DATA_RAW / "Telco_customer_churn.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_clean():
    interim = DATA_INTERIM / "telco_clean.csv"
    processed = DATA_PROCESSED / "telco_features.csv"
    if processed.exists():
        return pd.read_csv(processed)
    elif interim.exists():
        return pd.read_csv(interim)
    return None


df_raw = load_raw()
df_clean = load_clean()

if df_raw is None:
    st.error("No data found.")
    st.stop()

# ── Customer Search ──
st.markdown("### Search Customer")

search_method = st.radio("Search by", ["Index", "Customer ID"], horizontal=True)

if search_method == "Index":
    customer_idx = st.number_input(
        "Customer Index", min_value=0, max_value=len(df_raw) - 1, value=0, step=1,
    )
    customer = df_raw.iloc[customer_idx]
elif search_method == "Customer ID":
    if "CustomerID" in df_raw.columns:
        customer_id = st.selectbox("Customer ID", df_raw["CustomerID"].tolist()[:100])
        customer = df_raw[df_raw["CustomerID"] == customer_id].iloc[0]
        customer_idx = df_raw[df_raw["CustomerID"] == customer_id].index[0]
    else:
        st.warning("CustomerID column not found in raw data.")
        st.stop()

st.markdown("---")

# ── Customer Profile Display ──
st.markdown(f"### Customer #{customer_idx}")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    st.write(f"- Gender: **{customer.get('Gender', 'N/A')}**")
    st.write(f"- Senior Citizen: **{customer.get('Senior Citizen', 'N/A')}**")
    st.write(f"- Partner: **{customer.get('Partner', 'N/A')}**")
    st.write(f"- Dependents: **{customer.get('Dependents', 'N/A')}**")

with col2:
    st.markdown("**Account**")
    st.write(f"- Tenure: **{customer.get('Tenure Months', 'N/A')} months**")
    st.write(f"- Contract: **{customer.get('Contract', 'N/A')}**")
    st.write(f"- Payment: **{customer.get('Payment Method', 'N/A')}**")
    st.write(f"- Billing: **{customer.get('Paperless Billing', 'N/A')}**")

with col3:
    st.markdown("**Financial**")
    st.write(f"- Monthly: **${customer.get('Monthly Charges', 0):.2f}**")
    st.write(f"- Total: **${float(customer.get('Total Charges', 0)) if str(customer.get('Total Charges', '')).strip() else 0:.2f}**")
    actual_churn = customer.get("Churn Value", customer.get("Churn Label", "N/A"))
    st.write(f"- Actual Churn: **{actual_churn}**")

# ── Services ──
st.markdown("---")
st.markdown("### Services")

services = {
    "Phone Service": customer.get("Phone Service", "N/A"),
    "Multiple Lines": customer.get("Multiple Lines", "N/A"),
    "Internet Service": customer.get("Internet Service", "N/A"),
    "Online Security": customer.get("Online Security", "N/A"),
    "Online Backup": customer.get("Online Backup", "N/A"),
    "Device Protection": customer.get("Device Protection", "N/A"),
    "Tech Support": customer.get("Tech Support", "N/A"),
    "Streaming TV": customer.get("Streaming TV", "N/A"),
    "Streaming Movies": customer.get("Streaming Movies", "N/A"),
}

service_cols = st.columns(3)
for i, (service, value) in enumerate(services.items()):
    with service_cols[i % 3]:
        icon = "+" if value == "Yes" else "-"
        st.write(f"{icon} {service}: **{value}**")

# ── Churn Prediction ──
st.markdown("---")
st.markdown("### Churn Intelligence")

model_path = BEST_MODEL_FILE
if model_path.exists() and df_clean is not None:
    model = joblib.load(model_path)
    fn_path = MODELS_DIR / "feature_names.json"

    if fn_path.exists():
        with open(fn_path) as f:
            feature_names = json.load(f)

        # Get the features for this customer
        if customer_idx < len(df_clean):
            X_customer = df_clean.iloc[[customer_idx]].copy()

            # Ensure all columns exist
            for feat in feature_names:
                if feat not in X_customer.columns:
                    X_customer[feat] = 0

            # Only keep model features
            available_feats = [f for f in feature_names if f in X_customer.columns]
            if len(available_feats) == len(feature_names):
                X_pred = X_customer[feature_names]

                try:
                    churn_prob = model.predict_proba(X_pred)[:, 1][0]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Churn Probability", f"{churn_prob:.1%}")

                    with col2:
                        if churn_prob >= 0.7:
                            st.error("HIGH RISK")
                        elif churn_prob >= 0.4:
                            st.warning("MEDIUM RISK")
                        else:
                            st.success("LOW RISK")

                    with col3:
                        # Basic CLV
                        tenure = customer.get("Tenure Months", 0)
                        monthly = customer.get("Monthly Charges", 0)
                        clv = monthly * tenure
                        st.metric("Est. CLV", f"${clv:,.0f}")

                    # Recommendation
                    from src.recommend import generate_recommendation, classify_risk
                    rec_input = {
                        "churn_probability": churn_prob,
                        "clv": clv,
                        "clv_tier": "High" if clv > 3000 else "Medium" if clv > 1000 else "Low",
                        "is_month_to_month": 1 if customer.get("Contract") == "Month-to-month" else 0,
                        "is_new_customer": 1 if tenure < 12 else 0,
                        "is_electronic_check": 1 if customer.get("Payment Method") == "Electronic check" else 0,
                        "security_gap": 1 if (customer.get("Internet Service") == "Fiber optic" and customer.get("Online Security") == "No") else 0,
                        "service_count": sum(1 for s in services.values() if s == "Yes"),
                    }
                    rec = generate_recommendation(rec_input)

                    st.markdown("---")
                    st.markdown("### Recommendation")
                    st.info(f"**{rec['recommendation']}**\n\n{rec['action']}")

                except Exception as e:
                    st.warning(f"Could not generate prediction: {e}")
            else:
                st.warning("Feature mismatch. Run the full pipeline to enable predictions.")
        else:
            st.warning("Customer index out of range for processed data.")
else:
    st.info("Train the model to enable churn predictions here.")
