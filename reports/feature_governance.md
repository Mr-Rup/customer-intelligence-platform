# Feature Governance Report

## Target Variable

Target:

* Churn Value

Description:

* 1 = Customer churned
* 0 = Customer retained

---

## Features Approved for Analysis

Customer Demographics:

* Gender
* Senior Citizen
* Partner
* Dependents

Account Information:

* Tenure Months
* Contract
* Paperless Billing
* Payment Method

Services:

* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

Financial Variables:

* Monthly Charges
* Total Charges

Geographic Variables:

* State
* City
* Zip Code
* Latitude
* Longitude

---

## Features Excluded From Modeling

CustomerID
Reason:

* Unique identifier.

Count
Reason:

* Constant value.

Lat Long
Reason:

* Duplicate geographic information.

Churn Label
Reason:

* Direct representation of target.

Churn Score
Reason:

* Potential target leakage.

CLTV
Reason:

* Derived business metric not available at prediction time.

Churn Reason
Reason:

* Available only after churn occurs.
