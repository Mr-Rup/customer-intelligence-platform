"""
Page 3: Churn Prediction
=========================
Interactive customer profile input with real-time churn prediction.
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

from src.config import MODELS_DIR, BEST_MODEL_FILE

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide")
st.title("🎯 Churn Prediction")
st.markdown("Enter a customer profile to predict their churn risk.")
st.markdown("---")


@st.cache_resource
def load_model():
    if BEST_MODEL_FILE.exists():
        return joblib.load(BEST_MODEL_FILE)
    return None


@st.cache_data
def load_feature_names():
    fn_path = MODELS_DIR / "feature_names.json"
    if fn_path.exists():
        with open(fn_path) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_scaler():
    sc_path = MODELS_DIR / "scaler.joblib"
    if sc_path.exists():
        return joblib.load(sc_path)
    return None


model = load_model()
feature_names = load_feature_names()
scaler = load_scaler()

if model is None or feature_names is None:
    st.warning(
        "Model not found. Please run the training pipeline first:\n\n"
        "```bash\npython -m src.train\n```"
    )
    st.stop()

# ── Customer Profile Input ──
st.markdown("### Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    st.markdown("**Account**")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
    )

with col3:
    st.markdown("**Services**")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

col4, col5 = st.columns(2)
with col4:
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col5:
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, 5.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure, 50.0)

st.markdown("---")

# ── Predict Button ──
if st.button("Predict Churn Risk", type="primary", use_container_width=True):

    # Build feature dict matching the model's expected features
    input_data = {
        "Gender": 1 if gender == "Male" else 0,
        "Senior Citizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "Tenure Months": tenure,
        "Phone Service": 1 if phone == "Yes" else 0,
        "Paperless Billing": 1 if billing == "Yes" else 0,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
    }

    # Engineered features
    service_count = sum(1 for s in [phone, security, backup, protection,
                                      tech_support, streaming_tv, streaming_movies, multiple_lines]
                       if s == "Yes")
    input_data["service_count"] = service_count
    input_data["is_new_customer"] = 1 if tenure < 12 else 0
    input_data["revenue_per_month"] = total_charges / tenure if tenure > 0 else monthly_charges
    input_data["revenue_intensity"] = total_charges / (monthly_charges * tenure) if (monthly_charges * tenure) > 0 else 1.0
    input_data["has_family"] = 1 if partner == "Yes" or dependents == "Yes" else 0
    input_data["tech_adoption_score"] = sum(1 for s in [security, backup, protection, tech_support] if s == "Yes")
    input_data["is_month_to_month"] = 1 if contract == "Month-to-month" else 0
    input_data["is_fiber"] = 1 if internet == "Fiber optic" else 0
    input_data["is_electronic_check"] = 1 if payment == "Electronic check" else 0
    input_data["risk_score"] = input_data["is_month_to_month"] + input_data["is_fiber"] + input_data["is_electronic_check"] + input_data["is_new_customer"]
    input_data["contract_tenure"] = input_data["is_month_to_month"] * (72 - tenure)
    input_data["security_gap"] = 1 if internet == "Fiber optic" and security == "No" else 0
    input_data["high_charge_new_customer"] = 1 if monthly_charges > 65 and input_data["is_new_customer"] == 1 else 0

    # One-hot encode multi-category features
    multi_cats = {
        "Multiple Lines": multiple_lines,
        "Internet Service": internet,
        "Online Security": security,
        "Online Backup": backup,
        "Device Protection": protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Payment Method": payment,
    }

    # Tenure group
    if tenure <= 12:
        tg = "0-12"
    elif tenure <= 24:
        tg = "12-24"
    elif tenure <= 48:
        tg = "24-48"
    else:
        tg = "48+"

    # Service bundle
    if service_count <= 1:
        sb = "Basic"
    elif service_count <= 3:
        sb = "Standard"
    elif service_count <= 5:
        sb = "Premium"
    else:
        sb = "All-In"

    multi_cats["tenure_group"] = tg
    multi_cats["service_bundle"] = sb

    # Create DataFrame with all expected features
    df_input = pd.DataFrame([input_data])

    # Add one-hot columns
    for col_name, col_value in multi_cats.items():
        for feat in feature_names:
            if feat.startswith(f"{col_name}_"):
                category = feat[len(col_name) + 1:]
                df_input[feat] = 1 if col_value == category else 0

    # Ensure all feature columns exist
    for feat in feature_names:
        if feat not in df_input.columns:
            df_input[feat] = 0

    # Order columns to match model
    df_input = df_input[feature_names]

    # Scale numerical features
    if scaler is not None:
        num_cols = ["Tenure Months", "Monthly Charges", "Total Charges",
                    "service_count", "revenue_per_month", "revenue_intensity",
                    "tech_adoption_score", "risk_score", "contract_tenure"]
        scale_cols = [c for c in num_cols if c in df_input.columns]
        df_input[scale_cols] = scaler.transform(df_input[scale_cols])

    # Predict
    try:
        churn_prob = model.predict_proba(df_input)[:, 1][0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # ── Display Results ──
    st.markdown("---")
    st.markdown("### Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{churn_prob:.1%}")

    with col2:
        if churn_prob >= 0.7:
            risk = "🔴 HIGH RISK"
            color = "red"
        elif churn_prob >= 0.4:
            risk = "🟡 MEDIUM RISK"
            color = "orange"
        else:
            risk = "🟢 LOW RISK"
            color = "green"
        st.markdown(f"### {risk}")

    with col3:
        st.metric("Risk Score", f"{input_data['risk_score']}/4")

    # Risk gauge
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Churn Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ef4444" if churn_prob >= 0.5 else "#10b981"},
            "steps": [
                {"range": [0, 40], "color": "#d1fae5"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": churn_prob * 100,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
