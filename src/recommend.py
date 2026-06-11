"""
Rule-based recommendation engine for the Customer Intelligence Platform.

Generates actionable retention recommendations based on customer
attributes, churn risk, and segment membership.
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# Recommendation Rules
# ──────────────────────────────────────────────

RULES = [
    {
        "name": "Premium Retention Package",
        "priority": 1,
        "conditions": lambda r: r.get("clv_tier") == "High" and r.get("risk_level") == "High",
        "action": "Assign dedicated account manager. Offer personalized retention "
                  "incentive (15-20% discount on annual contract). Schedule quarterly "
                  "service review calls.",
        "expected_impact": "High",
        "estimated_cost": "$$$",
    },
    {
        "name": "Contract Migration Offer",
        "priority": 2,
        "conditions": lambda r: r.get("is_month_to_month") == 1 and r.get("risk_level") in ("High", "Medium"),
        "action": "Offer 15% discount for switching to annual contract. "
                  "Highlight cost savings and added benefits. Send comparison "
                  "email showing total annual savings.",
        "expected_impact": "High",
        "estimated_cost": "$$",
    },
    {
        "name": "Security Bundle Upgrade",
        "priority": 3,
        "conditions": lambda r: r.get("security_gap") == 1 and r.get("risk_level") in ("High", "Medium"),
        "action": "Offer free 3-month trial of Online Security + Tech Support bundle. "
                  "Emphasize protection value for fiber customers.",
        "expected_impact": "Medium",
        "estimated_cost": "$$",
    },
    {
        "name": "New Customer Onboarding",
        "priority": 4,
        "conditions": lambda r: r.get("is_new_customer") == 1 and r.get("risk_level") in ("High", "Medium"),
        "action": "Trigger guided onboarding program. Assign onboarding specialist "
                  "for first 90 days. Schedule check-in calls at 30, 60, 90 days.",
        "expected_impact": "Medium",
        "estimated_cost": "$",
    },
    {
        "name": "Payment Method Transition",
        "priority": 5,
        "conditions": lambda r: r.get("is_electronic_check") == 1 and r.get("risk_level") == "High",
        "action": "Offer $5/month discount for switching to automatic bank transfer "
                  "or credit card payment. Emphasize convenience and autopay benefits.",
        "expected_impact": "Medium",
        "estimated_cost": "$",
    },
    {
        "name": "Service Enrichment",
        "priority": 6,
        "conditions": lambda r: r.get("service_count", 0) <= 2 and r.get("risk_level") == "Medium",
        "action": "Recommend adding complementary services (Streaming, Online Backup). "
                  "Offer first month free for any new service addition.",
        "expected_impact": "Medium",
        "estimated_cost": "$",
    },
    {
        "name": "Loyalty Recognition",
        "priority": 7,
        "conditions": lambda r: r.get("clv_tier") == "High" and r.get("risk_level") == "Low",
        "action": "Send loyalty appreciation communication. Offer exclusive upgrade "
                  "or early access to new services. Include in VIP loyalty program.",
        "expected_impact": "Low",
        "estimated_cost": "$",
    },
    {
        "name": "Growth Nurture",
        "priority": 8,
        "conditions": lambda r: r.get("clv_tier") in ("Medium", "Low") and r.get("risk_level") == "Low",
        "action": "Include in automated nurture email sequence. Offer targeted "
                  "promotions based on usage patterns. Recommend service upgrades.",
        "expected_impact": "Low",
        "estimated_cost": "$",
    },
    {
        "name": "Standard Monitoring",
        "priority": 9,
        "conditions": lambda r: True,  # Catch-all
        "action": "Continue standard service monitoring. Include in regular "
                  "satisfaction surveys. No immediate intervention required.",
        "expected_impact": "Low",
        "estimated_cost": "-",
    },
]


def classify_risk(churn_probability: float) -> str:
    """Classify churn probability into risk levels."""
    if churn_probability >= 0.7:
        return "High"
    elif churn_probability >= 0.4:
        return "Medium"
    else:
        return "Low"


def classify_clv_tier(clv: float, clv_median: float, clv_q75: float) -> str:
    """Classify CLV into tiers."""
    if clv >= clv_q75:
        return "High"
    elif clv >= clv_median:
        return "Medium"
    else:
        return "Low"


def generate_recommendation(customer: dict) -> dict:
    """
    Generate a retention recommendation for a single customer.

    Parameters
    ----------
    customer : dict
        Customer attributes including:
        - churn_probability (float)
        - clv (float)
        - clv_tier (str)
        - is_month_to_month (int)
        - is_new_customer (int)
        - is_electronic_check (int)
        - security_gap (int)
        - service_count (int)

    Returns
    -------
    dict
        name, action, priority, expected_impact, estimated_cost
    """
    customer["risk_level"] = classify_risk(customer.get("churn_probability", 0))

    for rule in RULES:
        if rule["conditions"](customer):
            return {
                "recommendation": rule["name"],
                "action": rule["action"],
                "priority": rule["priority"],
                "expected_impact": rule["expected_impact"],
                "estimated_cost": rule["estimated_cost"],
                "risk_level": customer["risk_level"],
            }

    # Should never reach here due to catch-all rule
    return {"recommendation": "Standard Monitoring", "risk_level": "Low"}


def batch_recommend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate recommendations for all customers in a DataFrame.

    The DataFrame must contain the necessary columns for the rules.
    Returns the original DataFrame with recommendation columns added.
    """
    df = df.copy()

    # Ensure CLV tiers exist
    if "clv" in df.columns and "clv_tier" not in df.columns:
        clv_median = df["clv"].median()
        clv_q75 = df["clv"].quantile(0.75)
        df["clv_tier"] = df["clv"].apply(
            lambda x: classify_clv_tier(x, clv_median, clv_q75)
        )

    # Ensure risk level exists
    if "churn_probability" in df.columns and "risk_level" not in df.columns:
        df["risk_level"] = df["churn_probability"].apply(classify_risk)

    recommendations = []
    for _, row in df.iterrows():
        rec = generate_recommendation(row.to_dict())
        recommendations.append(rec)

    rec_df = pd.DataFrame(recommendations)
    result = pd.concat([df.reset_index(drop=True), rec_df], axis=1)

    return result


def recommendation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize recommendations by type.
    """
    if "recommendation" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("recommendation")
        .agg(
            customer_count=("recommendation", "size"),
            avg_churn_prob=("churn_probability", "mean") if "churn_probability" in df.columns else ("recommendation", "size"),
        )
        .sort_values("customer_count", ascending=False)
    )

    if "clv" in df.columns:
        summary["total_clv_at_risk"] = df.groupby("recommendation")["clv"].sum()

    return summary
