# Data Quality Report

## Dataset Overview

Records: 7,043

Original Features: 33

Features Retained for Modeling: 26

---

## Data Leakage Assessment

The following variables were excluded:

* Churn Label
* Churn Score
* CLTV
* Churn Reason

Reason:

These variables either directly encode the target variable or contain information unavailable at prediction time.

---

## Constant Features

Count was removed because it contains the same value for every observation and provides no predictive information.

---

## Data Type Issues

Total Charges was stored as a string variable and requires conversion to a numeric format before analysis and modeling.
