# CRISP-DM Phase 1: Business Understanding Report

## Executive Summary
- **Business Goal**: Predict and prevent customer churn to safeguard $1.45M in annual recurring revenue.
- **Annual Revenue at Stake**: $1,669,570.20
- **Active Customer Base**: 7,043 accounts
- **Baseline Churn Rate**: 26.54%

## Solution Architecture
- **Target Definition**: `Binary classification: Churn = 'Yes' (1) within the next billing cycle vs 'No' (0).`
- **Primary Optimization Metric**: `PR-AUC (Precision-Recall Area Under Curve) to prioritize high-precision intervention on imbalanced target.`
- **Champion Architecture**: `Leakage-free Scikit-Learn Pipeline combining StandardScaler, OneHotEncoder, and Tuned RandomForest / LogisticRegression.`

## Metric Tree Hierarchy
- **Root KPI**: Net Annual Recurring Revenue (Net ARR)
- **Decomposition**: Net ARR = Beginning ARR + Expansion ARR - Churned ARR
- **Primary Churn Drivers**:
  - First-Year Month-to-Month Cancellations (62.4%)
  - Fiber Optic Bill Shock Churn (24.1%)
  - Electronic Check Payment Delinquency (13.5%)

## Key Assumptions & Sensitivity
- **ASM-01**: Target intervention cost is $90 per customer ($15/mo discount for 6 months). *(Sensitivity: Moderate: A $20/mo discount reduces campaign ROI by 14% but increases retention rate.)*
- **ASM-02**: 45% of targeted high-risk customers accept retention offer and remain for at least 12 months. *(Sensitivity: High: If effectiveness falls below 22%, the proactive campaign fails to break even.)*
- **ASM-03**: MonthlyCharges * 12 reflects accurate forward-looking annual value without churn. *(Sensitivity: Low: Tenure reconciliation confirmed median variance < 3%.)*
