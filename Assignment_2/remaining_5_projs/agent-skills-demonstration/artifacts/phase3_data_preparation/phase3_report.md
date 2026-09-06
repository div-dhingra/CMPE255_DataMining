# CRISP-DM Phase 3: Data Preparation Report

## Leakage-Safe Cleaning
- **Imputation Strategy**: `train_set_median`
- **Imputed Value**: `$1398.12` (learned strictly from training split)
- **Train Imputed Missing**: 8 records
- **Test Imputed Missing**: 3 records
- **Target Encoding**: `No` -> 0, `Yes` -> 1

## Feature Engineering
- **Total Features Prepared**: 26
- **Domain Features Added**:
  - `TenureCohort`: Binned customer lifecycle stages (`0-12m`, `13-24m`, `25-48m`, `49-72m`)
  - `ServiceBundleCount`: Aggregated security & streaming add-ons count (0 to 6)
  - `HasFiberOptic`: InternetService indicator (high churn risk factor)
  - `IsMonthToMonth`: Short-term contract indicator
  - `IsElectronicCheck`: Manual payment indicator
  - `MonthlyToTotalRatio`: Ratio of monthly bill to cumulative lifetime billing
  - `HighSpendRisk`: Interaction flag (Month-to-month and MonthlyCharges > $70)

## Imbalanced Data Management
- **Original Imbalance**: 2.77:1 (Retained: 4,139, Churned: 1,495)
- **Class Weights**: {0: 0.6806, 1: 1.8843}
- **SMOTE Oversampling**: Synthesized 2,644 minority samples for training.
