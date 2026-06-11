# Exploratory Data Analysis Findings

## Churn Distribution

The dataset contains 7,043 customers, of which 1,869 customers have churned.

The overall churn rate is 26.54%, indicating that approximately one out of every four customers leaves the service.

From a business perspective, this level of customer attrition may have a substantial impact on recurring revenue and customer lifetime value. Retention-focused interventions could therefore generate significant business value.

From a modeling perspective, the target variable exhibits moderate class imbalance but remains suitable for standard classification approaches.


## Demographic Factors and Churn

A Chi-Square test of independence was conducted to evaluate the relationship between demographic variables and customer churn.

Gender was not significantly associated with churn (p = 0.487, Cramer's V = 0.008), indicating negligible practical influence.

Senior Citizen status demonstrated a statistically significant relationship with churn (p < 0.001, Cramer's V = 0.151). Senior customers exhibited substantially higher churn rates than non-senior customers.

Partner status was also significantly associated with churn (p < 0.001, Cramer's V = 0.150). Customers with partners showed lower churn rates, suggesting greater service stability among household customers.

Dependents exhibited the strongest demographic association with churn (p < 0.001, Cramer's V = 0.248). Customers with dependents had markedly lower churn rates, indicating that family-oriented households are substantially more likely to remain with the company.

Among demographic variables, Dependents emerged as the most influential predictor of customer retention.
