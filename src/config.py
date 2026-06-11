"""
Centralized configuration for the Customer Intelligence Platform.

All paths, constants, feature lists, and visualization settings
are defined here to ensure consistency across notebooks and modules.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Directory Structure
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
SQL_DIR = ROOT_DIR / "sql"

# Ensure directories exist
for d in [DATA_INTERIM, DATA_PROCESSED, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Data Files
# ──────────────────────────────────────────────
RAW_FILE = DATA_RAW / "Telco_customer_churn.csv"
CLEAN_FILE = DATA_INTERIM / "telco_clean.csv"
FEATURES_FILE = DATA_PROCESSED / "telco_features.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
EVALUATION_FILE = MODELS_DIR / "evaluation.json"

# ──────────────────────────────────────────────
# Target Variable
# ──────────────────────────────────────────────
TARGET = "Churn Value"

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ──────────────────────────────────────────────
# Feature Classification
# ──────────────────────────────────────────────

# Columns to drop due to leakage or irrelevance
LEAKAGE_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "Lat Long",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

# Geographic columns — loaded separately when needed, excluded from ML
GEOGRAPHIC_COLUMNS = [
    "State",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
]

# Demographic features
DEMOGRAPHIC_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
]

# Account / contract features
ACCOUNT_FEATURES = [
    "Tenure Months",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

# Service features
SERVICE_FEATURES = [
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

# Financial features
FINANCIAL_FEATURES = [
    "Monthly Charges",
    "Total Charges",
]

# All features approved for modeling (after cleaning)
CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

# Binary features (for label encoding)
BINARY_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Paperless Billing",
]

# Multi-category features (for one-hot encoding)
MULTI_CATEGORY_FEATURES = [
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Payment Method",
]

# ──────────────────────────────────────────────
# Visualization Settings
# ──────────────────────────────────────────────
PALETTE_CHURN = {0: "#2ecc71", 1: "#e74c3c"}
PALETTE_CHURN_LABELS = {"No": "#2ecc71", "Yes": "#e74c3c"}
COLOR_PRIMARY = "#3498db"
COLOR_SECONDARY = "#9b59b6"
COLOR_ACCENT = "#e67e22"
COLOR_BG = "#f8f9fa"

FIGSIZE_SMALL = (8, 5)
FIGSIZE_MEDIUM = (12, 6)
FIGSIZE_LARGE = (16, 8)
FIGSIZE_WIDE = (18, 6)
FIGSIZE_SQUARE = (8, 8)

# ──────────────────────────────────────────────
# Segment Names
# ──────────────────────────────────────────────
SEGMENT_NAMES = {
    0: "VIP Loyal",
    1: "VIP At Risk",
    2: "New Customers",
    3: "Low Value",
    4: "Growth Potential",
}
