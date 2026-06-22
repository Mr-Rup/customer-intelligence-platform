# Customer Intelligence Platform

> An end-to-end analytics system that predicts customer churn, explains why customers leave,
> estimates customer lifetime value, segments customers, and recommends data-driven retention actions.

---

## Architecture

```
Raw Data
    |
    v
Data Engineering  ──>  data/interim/telco_clean.csv
    |
    v
Business Analytics  ──>  SQL queries, EDA reports
    |
    v
Statistical Analysis  ──>  Hypothesis testing, effect sizes
    |
    v
Feature Engineering  ──>  data/processed/telco_features.csv
    |
    v
Machine Learning  ──>  5 models (LR, RF, XGBoost, LightGBM, CatBoost)
    |
    v
Explainability  ──>  SHAP global + local explanations
    |
    v
Customer Intelligence  ──>  Segments, CLV, Recommendations
    |
    v
Interactive Dashboard  ──>  6-page Streamlit application
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | 7,043 customers, 33 features |
| Churn Rate | 26.54% |
| Best Model ROC-AUC | ~0.85 |
| Optimized Threshold | 0.36 (F1-optimized) |
| Customer Segments | 5 (K-Means) |
| SQL Queries | 25+ business analytics queries |
| Engineered Features | 15 new features |
| Dashboard Pages | 6 interactive pages |

---

## Project Structure

```
customer-intelligence-platform/
|
+-- data/
|   +-- raw/                    # Original IBM Telco dataset
|   +-- interim/                # Cleaned data
|   +-- processed/              # Feature-engineered data
|
+-- notebooks/
|   +-- 01_data_audit.ipynb
|   +-- 02_data_cleaning.ipynb
|   +-- 03_eda.ipynb
|   +-- 04_statistical_analysis.ipynb
|   +-- 05_sql_analytics.ipynb
|   +-- 06_feature_engineering.ipynb
|   +-- 07_modeling.ipynb
|   +-- 08_explainability.ipynb
|   +-- 09_segmentation.ipynb
|   +-- 10_clv_analysis.ipynb
|   +-- 11_business_recommendations.ipynb
|
+-- src/
|   +-- config.py               # Centralized configuration
|   +-- load_data.py            # Data loading utilities
|   +-- cleaning.py             # Data cleaning pipeline
|   +-- eda.py                  # EDA visualization functions
|   +-- statistics.py           # Hypothesis testing
|   +-- feature_engineering.py  # Feature creation
|   +-- preprocessing.py        # Encoding, scaling, splitting
|   +-- train.py                # Model training pipeline
|   +-- evaluate.py             # Metrics and evaluation
|   +-- explain.py              # SHAP explainability
|   +-- segment.py              # K-Means segmentation
|   +-- recommend.py            # Rule-based recommendations
|   +-- utils.py                # Shared utilities
|
+-- sql/
|   +-- business_queries.sql    # 25+ SQL analytics queries
|
+-- dashboard/
|   +-- app.py                  # Streamlit main application
|   +-- pages/                  # 6 dashboard pages
|   +-- assets/                 # Custom CSS
|
+-- reports/
|   +-- executive_summary.md
|   +-- data_audit.md
|   +-- statistical_report.md
|   +-- model_report.md
|   +-- eda_findings.md
|   +-- feature_governance.md
|   +-- data_quality_report.md
|   +-- business_recommendations.md
|
+-- models/                     # Serialized models and metrics
+-- dvc.yaml                    # DVC pipeline definition
+-- params.yaml                 # Hyperparameters
+-- requirements.txt            # Python dependencies
```

---

## Methodology

### 1. Data Governance
- Identified and removed 8 leakage/irrelevant columns
- Fixed data types (Total Charges string -> numeric)
- Imputed 11 missing values (new customers)
- Separated geographic data for map-only usage

### 2. Exploratory Data Analysis
Extensive analysis across 7 dimensions:
- Churn distribution, demographics, services, contracts, financials, tenure, geography

### 3. Statistical Analysis
- Chi-Square tests + Cramer's V for categorical features
- Welch's t-tests + Cohen's d for numerical features
- Bonferroni and Benjamini-Hochberg multiple testing corrections

### 4. SQL Analytics
- 25+ business queries in SQLite
- Revenue analysis, cohort analysis, risk factor accumulation

### 5. Feature Engineering
15 new features across 6 groups:
- Tenure groups, service count, revenue ratios
- Customer profiles, risk indicators, interaction features

### 6. Predictive Modeling
5 classifiers with class imbalance handling:
- Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Threshold optimization (F1-based)

### 7. Explainability
- SHAP global explanations (feature importance rankings)
- SHAP local explanations (individual customer "why")
- Dependence plots for top features

### 8. Customer Segmentation
- K-Means clustering (5 segments)
- Elbow method + silhouette analysis
- Business-meaningful segment naming

### 9. Customer Lifetime Value
- Basic CLV: Monthly Charges x Tenure
- Enhanced CLV: Adjusted by churn probability
- High Value + High Risk customer identification

### 10. Retention Recommendations
- Rule-based engine with 9 prioritized strategies
- Personalized recommendations per customer
- Expected ROI estimation

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Mr-Rup/customer-intelligence-platform.git
cd customer-intelligence-platform

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the ML Pipeline (DVC)
```bash
dvc repro
```

### Run Individual Pipeline Stages
```bash
python -m src.cleaning
python -m src.feature_engineering
python -m src.train
python -m src.evaluate
```

### Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

### Run Notebooks
```bash
jupyter notebook
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn, Plotly |
| Statistics | SciPy |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Pipeline | DVC |
| Database | SQLite |
| Version Control | Git + DVC |

---

## Future Work

- [ ] Hyperparameter tuning with Optuna
- [ ] Time-series churn prediction with survival analysis
- [ ] A/B testing framework for retention strategies
- [ ] Real-time prediction API (FastAPI)
- [ ] Automated model retraining pipeline
- [ ] Customer satisfaction (NPS) integration
- [ ] Cost-sensitive threshold optimization

---

## License

MIT License