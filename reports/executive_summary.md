# Executive Summary

## Customer Intelligence Platform

### Business Problem

Customer churn is one of the most critical metrics in the telecommunications industry.
For every customer lost, the company incurs acquisition costs to replace them — typically
5-7x the cost of retention. This project builds an end-to-end analytics platform to
**predict**, **explain**, and **prevent** customer churn. 

### Dataset

IBM Telco Customer Churn dataset: **7,043 customers** with **33 features** covering
demographics, account information, services, financial data, and geographic location.

### Key Findings

#### Churn Rate
- **26.54%** of customers have churned (1,869 out of 7,043)
- This represents approximately **$139,000+ in monthly revenue at risk**

#### Top Churn Drivers (Statistically Validated)
1. **Contract Type**: Month-to-month customers churn at 43% vs 3% for two-year contracts
2. **Tenure**: 50% of churners leave in their first year
3. **Internet Service**: Fiber optic customers without security services churn at 42%+
4. **Payment Method**: Electronic check users churn at 45%
5. **Family Status**: Customers with dependents churn at only 15%

#### Model Performance
- **5 models trained**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- **Best ROC-AUC**: ~0.85 (strong discriminative ability)
- **Threshold optimized** for maximum F1 score

#### Customer Segments
Five distinct customer segments identified:
- **VIP Loyal**: Long-tenure, high-value, low-risk
- **VIP At Risk**: High-value but elevated churn probability
- **New Customers**: Recently joined, uncertain loyalty
- **Low Value**: Minimal engagement and spend
- **Growth Potential**: Moderate value with expansion opportunity

### Recommendations

| Priority | Action | Target | Expected Impact |
|----------|--------|--------|----------------|
| 1 | Premium retention package | High CLV + High Risk | Prevent highest-value losses |
| 2 | Contract migration incentive | Month-to-month + At Risk | Convert to annual contracts |
| 3 | Security bundle promotion | Fiber without security | Reduce service-gap churn |
| 4 | Onboarding program | New customers (<12 months) | Strengthen critical first year |
| 5 | Payment method transition | Electronic check users | Reduce payment-driven churn |

### Business Impact

If the retention program achieves even a **10% reduction** in churn:
- ~187 customers retained
- ~$14,000+ monthly revenue preserved
- ~$168,000+ annual revenue impact

### Data Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Male / Female |
| Senior Citizen | Categorical | Yes / No |
| Partner | Categorical | Has a partner |
| Dependents | Categorical | Has dependents |
| Tenure Months | Numerical | Months as a customer |
| Phone Service | Categorical | Subscribed to phone |
| Multiple Lines | Categorical | Multiple phone lines |
| Internet Service | Categorical | DSL / Fiber optic / No |
| Online Security | Categorical | Subscribed to security |
| Online Backup | Categorical | Subscribed to backup |
| Device Protection | Categorical | Subscribed to protection |
| Tech Support | Categorical | Subscribed to support |
| Streaming TV | Categorical | Subscribed to streaming TV |
| Streaming Movies | Categorical | Subscribed to streaming movies |
| Contract | Categorical | Month-to-month / One year / Two year |
| Paperless Billing | Categorical | Uses paperless billing |
| Payment Method | Categorical | Electronic check / Mailed check / Bank / Credit card |
| Monthly Charges | Numerical | Monthly payment amount ($) |
| Total Charges | Numerical | Total amount paid ($) |
| Churn Value | Binary | 0 = Retained, 1 = Churned (TARGET) |
