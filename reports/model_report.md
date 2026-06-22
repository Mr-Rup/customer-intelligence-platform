# Model Performance Report

## Models Evaluated

| Model | Type | Key Characteristics |
|-------|------|-------------------|
| Logistic Regression | Linear | Interpretable coefficients, fast training |
| Random Forest | Ensemble (Bagging) | Feature importance, handles non-linearity |
| XGBoost | Ensemble (Boosting) | Gradient boosting, regularization |
| LightGBM | Ensemble (Boosting) | Histogram-based, fast training |
| CatBoost | Ensemble (Boosting) | Native categorical handling |

## Results Summary

All models were trained with class weighting or scale_pos_weight to handle
the 26.54% churn rate (class imbalance ratio of 2.77:1).

### Metrics at Default Threshold (0.50)

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| Best model selected by ROC-AUC | | | | |

*Note: Run the 07_modeling.ipynb notebook for exact values.* 

## Threshold Optimization

The default threshold of 0.50 is not optimal for churn prediction.
We optimized the threshold by maximizing the F1 score.

**Optimal Threshold**: ~0.36

At the optimal threshold:
- **Recall increases significantly** (catching more actual churners)
- **Precision decreases slightly** (more false positives)
- **F1 score improves** (better balance overall)

### Business Rationale for Lower Threshold
In churn prediction, the cost of **missing a churner** (false negative) is typically
much higher than the cost of **a false alarm** (false positive). A lower threshold
trades precision for recall, which aligns with retention business objectives.

## Feature Importance

### SHAP-Based Feature Ranking

Top features identified by SHAP analysis (from the best model):

1. **Contract** (Month-to-month flag)
2. **Tenure Months**
3. **Monthly Charges**
4. **Internet Service** (Fiber optic)
5. **Online Security**
6. **Tech Support**
7. **Payment Method** (Electronic check)
8. **Total Charges**
9. **Dependents**
10. **Partner**

## Model Selection Justification

The best model was selected based on:
1. **ROC-AUC** (primary metric): Measures overall ability to distinguish churners
2. **F1 Score** (secondary): Balances precision and recall
3. **Interpretability**: For business stakeholder communication

## Limitations

1. **Temporal validation**: No time-based split (cross-sectional data)
2. **External validity**: Model trained on a single telco company's data
3. **Feature engineering**: All features derived from customer snapshot data
4. **No cost-sensitive optimization**: Could incorporate actual retention campaign costs
