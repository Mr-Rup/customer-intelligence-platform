# Data Audit Report

Dataset Name:
IBM Telco Customer Churn (Enhanced Version)

Total Records:
7043

Total Features:
33

Initial Findings:

1. No missing values were observed in most variables.

2. Churn Reason contains 5,174 missing values. This is expected because churn reasons are only available for customers who have churned.

3. Additional columns such as Churn Score, Churn Label, Churn Value, and CLTV require investigation for potential target leakage before model development.

4. Geographic information including City, State, Latitude, Longitude, and Zip Code is available and may be useful for exploratory analysis and customer segmentation.

5. Further feature screening is required before model training.
