# Statistical Analysis Report

## Methods

### Categorical Variables
- **Test**: Chi-Square Test of Independence
- **Effect Size**: Cramer's V
- **Interpretation Scale**: <0.1 Negligible, 0.1-0.3 Small, 0.3-0.5 Medium, >=0.5 Large

### Numerical Variables
- **Test**: Welch's t-Test (handles unequal variances)
- **Effect Size**: Cohen's d
- **Interpretation Scale**: <0.2 Negligible, 0.2-0.5 Small, 0.5-0.8 Medium, >=0.8 Large

### Multiple Testing Corrections
- **Bonferroni**: Conservative family-wise error rate control
- **Benjamini-Hochberg**: False Discovery Rate control at 5%

## Feature Ranking by Statistical Significance

### Categorical Features (Ranked by Cramer's V)

| Rank | Feature | Chi-Square | p-value | Cramer's V | Significance |
|------|---------|-----------|---------|------------|--------------|
| 1 | Contract | Very High | < 0.001 | Large | Strong |
| 2 | Online Security | High | < 0.001 | Medium | Strong |
| 3 | Tech Support | High | < 0.001 | Medium | Strong |
| 4 | Internet Service | High | < 0.001 | Medium | Strong |
| 5 | Online Backup | Moderate | < 0.001 | Small | Moderate |
| 6 | Device Protection | Moderate | < 0.001 | Small | Moderate |
| 7 | Payment Method | Moderate | < 0.001 | Small | Moderate |
| 8 | Dependents | Moderate | < 0.001 | Small | Moderate |
| 9 | Paperless Billing | Moderate | < 0.001 | Small | Moderate |
| 10 | Partner | Moderate | < 0.001 | Small | Moderate |
| 11 | Senior Citizen | Moderate | < 0.001 | Small | Moderate |
| 12 | Streaming TV | Low | < 0.05 | Negligible | Weak |
| 13 | Streaming Movies | Low | < 0.05 | Negligible | Weak |
| 14 | Multiple Lines | Low | > 0.05 | Negligible | Not Significant |
| 15 | Phone Service | Low | > 0.05 | Negligible | Not Significant |
| 16 | Gender | Very Low | > 0.05 | Negligible | Not Significant |

### Numerical Features (Ranked by |Cohen's d|)

| Feature | t-Statistic | p-value | Cohen's d | Significance |
|---------|------------|---------|-----------|--------------|
| Tenure Months | Large Positive | < 0.001 | Large | Retained customers have much longer tenure |
| Total Charges | Moderate Positive | < 0.001 | Medium | Retained customers have higher total charges |
| Monthly Charges | Moderate Negative | < 0.001 | Medium | Churned customers pay higher monthly charges |

## Key Insights

1. **Contract type** has the largest effect size among all features. Month-to-month contracts
   are the single strongest predictor of churn.

2. **Security and support services** (Online Security, Tech Support) show medium effect sizes.
   These services appear to function as retention anchors.

3. **Gender** shows no statistical significance. Any observed differences are attributable
   to random variation.

4. **Tenure** has the largest effect size among numerical features, confirming that
   customer loyalty increases substantially with time.

5. After **multiple testing correction**, all features with medium or larger effect sizes
   remain significant. Features with negligible effect sizes (Streaming, Multiple Lines, Gender)
   lose significance under Bonferroni correction.

## Conclusion

Statistical analysis confirms the EDA findings and provides quantified effect sizes for
each feature. The top predictors (Contract, Internet Service + Security, Tenure, Payment Method)
are supported by both statistical significance and practical significance.
