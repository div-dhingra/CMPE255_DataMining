# CRISP-DM Phase 2: Data Understanding Report

## Dataset Health & Quality Audit
- **Total Records**: 7,043
- **Primary Key Unique**: True
- **Data Health Score**: 74.0 / 100
- **Whitespace Corruption**: 11 empty strings in `TotalCharges` (tenure=0 accounts)

## EDA & Target Distribution
- **Class Balance**: Retained: 5,174 | Churned: 1,869 (26.54%)
- **Numerical Summary**:
  - `tenure`: Mean = 32.37m, Median = 29.0m
  - `MonthlyCharges`: Mean = $64.76, Median = $70.35
  - `TotalCharges`: Mean = $2283.3, Median = $1397.47

## Metric Reconciliation
- **Discrepancy Status**: `RECONCILED_WITH_EXPLAINED_VARIANCE`
- **Mean Variance**: 3.21% (explained by promotional tenure discounts)

## Vectorized Pandas Patterns
- **Memory Footprint**: Reduced from 7975.08 KB to 7494.46 KB (6.03% reduction).
