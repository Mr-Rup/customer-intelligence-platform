-- ============================================================
-- Customer Intelligence Platform - Business SQL Queries
-- ============================================================
-- Database: SQLite (created from telco_clean.csv)
-- Target: Churn Value (0 = Retained, 1 = Churned)
-- ============================================================


-- ============================================================
-- 1. OVERVIEW QUERIES
-- ============================================================

-- 1.1 Overall churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM("Churn Value") AS churned_customers,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers;


-- 1.2 Customer distribution by churn status
SELECT
    CASE WHEN "Churn Value" = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage  -- upto 2 decimal place
FROM customers
GROUP BY "Churn Value"
ORDER BY "Churn Value";


-- 1.3 Revenue summary
SELECT
    ROUND(SUM("Monthly Charges"), 2) AS total_monthly_revenue,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly_charge,
    ROUND(SUM("Total Charges"), 2) AS total_lifetime_revenue,
    ROUND(AVG("Total Charges"), 2) AS avg_total_charge
FROM customers;


-- ============================================================
-- 2. DEMOGRAPHIC ANALYSIS
-- ============================================================

-- 2.1 Churn rate by gender
SELECT
    "Gender",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY "Gender"
ORDER BY churn_rate_pct DESC;


-- 2.2 Churn rate by senior citizen status
SELECT
    "Senior Citizen",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY "Senior Citizen"
ORDER BY churn_rate_pct DESC;


-- 2.3 Churn rate by partner and dependents
SELECT
    "Partner",
    "Dependents",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY "Partner", "Dependents"
ORDER BY churn_rate_pct DESC;


-- 2.4 Family status impact on churn
SELECT
    CASE
        WHEN "Partner" = 'Yes' AND "Dependents" = 'Yes' THEN 'Full Family'
        WHEN "Partner" = 'Yes' AND "Dependents" = 'No' THEN 'Partner Only'
        WHEN "Partner" = 'No' AND "Dependents" = 'Yes' THEN 'Single Parent'
        ELSE 'Single'
    END AS family_status,
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly_charge
FROM customers
GROUP BY family_status
ORDER BY churn_rate_pct DESC;


-- ============================================================
-- 3. SERVICE ANALYSIS
-- ============================================================

-- 3.1 Churn rate by internet service type
SELECT
    "Internet Service",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly_charge
FROM customers
GROUP BY "Internet Service"
ORDER BY churn_rate_pct DESC;


-- 3.2 Online security impact on churn
SELECT
    "Online Security",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY "Online Security"
ORDER BY churn_rate_pct DESC;


-- 3.3 Tech support impact on churn
SELECT
    "Tech Support",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY "Tech Support"
ORDER BY churn_rate_pct DESC;


-- 3.4 Service adoption analysis - count services per customer
SELECT
    service_count,
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM (
    SELECT
        *,
        (CASE WHEN "Phone Service" = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN "Multiple Lines" = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN "Online Security" = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN "Online Backup" = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN "Device Protection" = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN "Tech Support" = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN "Streaming TV" = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN "Streaming Movies" = 'Yes' THEN 1 ELSE 0 END) AS service_count
    FROM customers
) subq
GROUP BY service_count
ORDER BY service_count;


-- 3.5 Streaming services and churn
SELECT
    "Streaming TV",
    "Streaming Movies",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY "Streaming TV", "Streaming Movies"
ORDER BY churn_rate_pct DESC;


-- 3.6 Fiber without security (risk segment)
SELECT
    CASE
        WHEN "Internet Service" = 'Fiber optic' AND "Online Security" = 'No' THEN 'Fiber, No Security'
        WHEN "Internet Service" = 'Fiber optic' AND "Online Security" = 'Yes' THEN 'Fiber, With Security'
        WHEN "Internet Service" = 'DSL' THEN 'DSL'
        ELSE 'No Internet'
    END AS service_profile,
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY service_profile
ORDER BY churn_rate_pct DESC;


-- ============================================================
-- 4. CONTRACT & PAYMENT ANALYSIS
-- ============================================================

-- 4.1 Churn rate by contract type
SELECT
    "Contract",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly,
    ROUND(SUM("Monthly Charges"), 2) AS total_monthly_revenue
FROM customers
GROUP BY "Contract"
ORDER BY churn_rate_pct DESC;


-- 4.2 Churn rate by payment method
SELECT
    "Payment Method",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly
FROM customers
GROUP BY "Payment Method"
ORDER BY churn_rate_pct DESC;


-- 4.3 Paperless billing impact
SELECT
    "Paperless Billing",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY "Paperless Billing"
ORDER BY churn_rate_pct DESC;


-- 4.4 Contract and payment method cross-analysis
SELECT
    "Contract",
    "Payment Method",
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY "Contract", "Payment Method"
HAVING COUNT(*) >= 50
ORDER BY churn_rate_pct DESC
LIMIT 15;


-- ============================================================
-- 5. FINANCIAL ANALYSIS
-- ============================================================

-- 5.1 Revenue at risk from churning customers
SELECT
    CASE WHEN "Churn Value" = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    ROUND(SUM("Monthly Charges"), 2) AS monthly_revenue,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly,
    ROUND(SUM("Total Charges"), 2) AS total_revenue,
    ROUND(AVG("Total Charges"), 2) AS avg_total
FROM customers
GROUP BY "Churn Value";


-- 5.2 Revenue by charge brackets
SELECT
    CASE
        WHEN "Monthly Charges" < 30 THEN 'Low ($0-30)'
        WHEN "Monthly Charges" < 60 THEN 'Medium ($30-60)'
        WHEN "Monthly Charges" < 90 THEN 'High ($60-90)'
        ELSE 'Premium ($90+)'
    END AS charge_bracket,
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(SUM("Monthly Charges"), 2) AS total_monthly_revenue
FROM customers
GROUP BY charge_bracket
ORDER BY charge_bracket;


-- 5.3 High-value customer identification
SELECT
    COUNT(*) AS high_value_count,
    SUM("Churn Value") AS high_value_churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(SUM("Monthly Charges"), 2) AS monthly_revenue_at_risk
FROM customers
WHERE "Monthly Charges" >= (SELECT AVG("Monthly Charges") + STDEV("Monthly Charges") FROM customers)
  AND "Tenure Months" >= 24;


-- ============================================================
-- 6. TENURE & LOYALTY ANALYSIS
-- ============================================================

-- 6.1 Churn rate by tenure cohort
SELECT
    CASE
        WHEN "Tenure Months" <= 12 THEN '0-12 months'
        WHEN "Tenure Months" <= 24 THEN '12-24 months'
        WHEN "Tenure Months" <= 48 THEN '24-48 months'
        ELSE '48+ months'
    END AS tenure_cohort,
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly_charge,
    ROUND(AVG("Total Charges"), 2) AS avg_total_charge
FROM customers
GROUP BY tenure_cohort
ORDER BY tenure_cohort;


-- 6.2 New customer risk (first year)
SELECT
    'New Customers (0-12 months)' AS segment,
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(SUM("Monthly Charges"), 2) AS monthly_revenue
FROM customers
WHERE "Tenure Months" <= 12

UNION ALL

SELECT
    'Established Customers (12+ months)' AS segment,
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(SUM("Monthly Charges"), 2) AS monthly_revenue
FROM customers
WHERE "Tenure Months" > 12;


-- 6.3 Loyal customer profile
SELECT
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly,
    ROUND(AVG("Total Charges"), 2) AS avg_total,
    COUNT(DISTINCT "Contract") AS contract_types,
    ROUND(AVG(CASE WHEN "Partner" = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS pct_with_partner,
    ROUND(AVG(CASE WHEN "Dependents" = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS pct_with_dependents
FROM customers
WHERE "Tenure Months" >= 48 AND "Churn Value" = 0;


-- ============================================================
-- 7. RETENTION INSIGHTS
-- ============================================================

-- 7.1 Service combinations with lowest churn rate
SELECT
    "Internet Service",
    "Online Security",
    "Tech Support",
    COUNT(*) AS total,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
WHERE "Internet Service" != 'No'
GROUP BY "Internet Service", "Online Security", "Tech Support"
HAVING COUNT(*) >= 50
ORDER BY churn_rate_pct ASC
LIMIT 10;


-- 7.2 Best retention contract + payment combination
SELECT
    "Contract",
    "Payment Method",
    COUNT(*) AS total,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly
FROM customers
GROUP BY "Contract", "Payment Method"
HAVING COUNT(*) >= 30
ORDER BY churn_rate_pct ASC
LIMIT 10;


-- ============================================================
-- 8. ADVANCED ANALYTICS
-- ============================================================

-- 8.1 Cumulative revenue by tenure (customer lifetime summary)
SELECT
    "Tenure Months",
    COUNT(*) AS customers_at_tenure,
    SUM("Churn Value") AS churned_at_tenure,
    ROUND(AVG("Total Charges"), 2) AS avg_total_charges,
    ROUND(SUM("Total Charges"), 2) AS cumulative_revenue
FROM customers
GROUP BY "Tenure Months"
ORDER BY "Tenure Months";


-- 8.2 Risk scoring - customers with multiple risk factors
SELECT
    risk_factors,
    COUNT(*) AS total,
    SUM("Churn Value") AS churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM (
    SELECT
        *,
        (CASE WHEN "Contract" = 'Month-to-month' THEN 1 ELSE 0 END +
         CASE WHEN "Internet Service" = 'Fiber optic' THEN 1 ELSE 0 END +
         CASE WHEN "Payment Method" = 'Electronic check' THEN 1 ELSE 0 END +
         CASE WHEN "Tenure Months" <= 12 THEN 1 ELSE 0 END) AS risk_factors
    FROM customers
) subq
GROUP BY risk_factors
ORDER BY risk_factors;


-- 8.3 Monthly revenue churn impact by contract
SELECT
    "Contract",
    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN "Monthly Charges" ELSE 0 END), 2) AS revenue_lost,
    ROUND(SUM("Monthly Charges"), 2) AS total_revenue,
    ROUND(
        SUM(CASE WHEN "Churn Value" = 1 THEN "Monthly Charges" ELSE 0 END) * 100.0 /
        SUM("Monthly Charges"),
    2) AS revenue_loss_pct
FROM customers
GROUP BY "Contract"
ORDER BY revenue_loss_pct DESC;


-- 8.4 Top 10 highest-risk customer profiles
SELECT
    "Contract",
    "Internet Service",
    "Payment Method",
    "Senior Citizen",
    CASE
        WHEN "Tenure Months" <= 12 THEN 'New'
        WHEN "Tenure Months" <= 48 THEN 'Established'
        ELSE 'Veteran'
    END AS tenure_category,
    COUNT(*) AS total,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY "Contract", "Internet Service", "Payment Method", "Senior Citizen", tenure_category
HAVING COUNT(*) >= 20
ORDER BY churn_rate_pct DESC
LIMIT 10;


-- 8.5 Monthly charges percentile analysis by churn
SELECT
    CASE WHEN "Churn Value" = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    ROUND(MIN("Monthly Charges"), 2) AS min_charge,
    ROUND(AVG("Monthly Charges"), 2) AS avg_charge,
    ROUND(MAX("Monthly Charges"), 2) AS max_charge
FROM customers
GROUP BY "Churn Value";
