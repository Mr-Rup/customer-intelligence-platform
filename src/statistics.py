"""
Statistical analysis module for the Customer Intelligence Platform.

Provides hypothesis testing, effect size calculations, and
multiple testing corrections for categorical and numerical features.
"""

import pandas as pd
import numpy as np
from scipy import stats


def chi_square_test(df: pd.DataFrame, feature: str, target: str) -> dict:
    """
    Perform Chi-Square test of independence between a categorical
    feature and the target variable.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
        Categorical feature column name.
    target : str
        Binary target column name.

    Returns
    -------
    dict
        Keys: feature, chi2, p_value, dof, cramers_v, n, interpretation
    """
    contingency = pd.crosstab(df[feature], df[target])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    k = min(contingency.shape) - 1
    v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0

    # Interpret Cramer's V
    if v < 0.1:
        interpretation = "Negligible"
    elif v < 0.3:
        interpretation = "Small"
    elif v < 0.5:
        interpretation = "Medium"
    else:
        interpretation = "Large"

    return {
        "feature": feature,
        "test": "Chi-Square",
        "chi2": round(chi2, 4),
        "p_value": p_value,
        "dof": dof,
        "cramers_v": round(v, 4),
        "n": n,
        "significant": p_value < 0.05,
        "effect_interpretation": interpretation,
    }


def cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Calculate Cramer's V from a contingency table.
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * k)) if k > 0 else 0.0


def welch_t_test(df: pd.DataFrame, feature: str, target: str) -> dict:
    """
    Perform Welch's t-test comparing a numerical feature between
    churn groups (handles unequal variances).

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
        Numerical feature column name.
    target : str
        Binary target column name.

    Returns
    -------
    dict
        Keys: feature, t_stat, p_value, cohens_d, mean_0, mean_1,
              std_0, std_1, interpretation
    """
    group_0 = df[df[target] == 0][feature].dropna()
    group_1 = df[df[target] == 1][feature].dropna()

    t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)

    # Cohen's d
    d = cohens_d(group_0, group_1)

    # Interpret Cohen's d
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "Negligible"
    elif abs_d < 0.5:
        interpretation = "Small"
    elif abs_d < 0.8:
        interpretation = "Medium"
    else:
        interpretation = "Large"

    return {
        "feature": feature,
        "test": "Welch's t-test",
        "t_stat": round(t_stat, 4),
        "p_value": p_value,
        "cohens_d": round(d, 4),
        "mean_retained": round(group_0.mean(), 2),
        "mean_churned": round(group_1.mean(), 2),
        "std_retained": round(group_0.std(), 2),
        "std_churned": round(group_1.std(), 2),
        "significant": p_value < 0.05,
        "effect_interpretation": interpretation,
    }


def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Calculate Cohen's d effect size between two groups.

    Uses pooled standard deviation.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def multiple_testing_correction(p_values: list, method: str = "bonferroni") -> list:
    """
    Apply multiple testing correction to a list of p-values.

    Parameters
    ----------
    p_values : list of float
    method : str
        'bonferroni' or 'bh' (Benjamini-Hochberg).

    Returns
    -------
    list of float
        Adjusted p-values.
    """
    n = len(p_values)
    if method == "bonferroni":
        return [min(p * n, 1.0) for p in p_values]
    elif method == "bh":
        # Benjamini-Hochberg procedure
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * n
        prev_adjusted = 0.0
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            adj_p = min(p * n / rank, 1.0)
            adjusted[orig_idx] = adj_p
        # Enforce monotonicity (from largest to smallest rank)
        sorted_indices = [x[0] for x in sorted(enumerate(p_values), key=lambda x: x[1])]
        running_min = 1.0
        for i in reversed(range(n)):
            orig_idx = sorted_indices[i]
            adjusted[orig_idx] = min(adjusted[orig_idx], running_min)
            running_min = adjusted[orig_idx]
        return adjusted
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bonferroni' or 'bh'.")


def run_all_categorical_tests(
    df: pd.DataFrame,
    features: list,
    target: str,
) -> pd.DataFrame:
    """
    Run Chi-Square tests on all categorical features and return
    a summary DataFrame sorted by effect size.
    """
    results = [chi_square_test(df, feat, target) for feat in features]
    results_df = pd.DataFrame(results)

    # Apply multiple testing corrections
    p_vals = results_df["p_value"].tolist()
    results_df["p_bonferroni"] = multiple_testing_correction(p_vals, "bonferroni")
    results_df["p_bh"] = multiple_testing_correction(p_vals, "bh")

    return results_df.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def run_all_numerical_tests(
    df: pd.DataFrame,
    features: list,
    target: str,
) -> pd.DataFrame:
    """
    Run Welch's t-tests on all numerical features and return
    a summary DataFrame sorted by effect size.
    """
    results = [welch_t_test(df, feat, target) for feat in features]
    results_df = pd.DataFrame(results)

    # Apply multiple testing corrections
    p_vals = results_df["p_value"].tolist()
    results_df["p_bonferroni"] = multiple_testing_correction(p_vals, "bonferroni")
    results_df["p_bh"] = multiple_testing_correction(p_vals, "bh")

    return results_df.sort_values(
        "cohens_d", key=abs, ascending=False
    ).reset_index(drop=True)


def summarize_all_tests(
    df: pd.DataFrame,
    categorical_features: list,
    numerical_features: list,
    target: str,
) -> tuple:
    """
    Run all statistical tests and return summary DataFrames.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (categorical_results, numerical_results)
    """
    cat_results = run_all_categorical_tests(df, categorical_features, target)
    num_results = run_all_numerical_tests(df, numerical_features, target)
    return cat_results, num_results
