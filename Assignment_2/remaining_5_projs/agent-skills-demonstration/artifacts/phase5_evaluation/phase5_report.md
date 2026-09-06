# CRISP-DM Phase 5: Evaluation Report

## Model Evaluation Metrics (Holdout Test Set)
- **ROC-AUC**: 0.8477
- **PR-AUC**: 0.6682
- **Brier Calibration Score**: 0.1647
- **Optimal Decision Threshold**: 0.577 (F1 = 0.6341)
- **Precision**: 0.5514 | **Recall**: 0.746
- **Confusion Matrix**: True Negatives: 808, False Positives: 227, False Negatives: 95, True Positives: 279

## Cohort Analysis & Retention Dynamics
- **High Risk Cohort**: `Tenure 0-12m with Month-to-Month Contract (~47% Churn)`
- **Low Risk Cohort**: `Tenure 49-72m with Two Year Contract (<3% Churn)`
- **Key Takeaway**: `First-year Month-to-Month accounts require immediate automated onboarding and intervention.`

## Root Cause 5-Whys Diagnostic
- **Core Problem**: Elevated churn rate of 26.5% predominantly concentrated in the first 12 months.
- **5-Whys Progression**:
  - **Why_1**: Why is customer churn disproportionately high? -> Customers on month-to-month contracts cancel within 90 days.
  - **Why_2**: Why do month-to-month users cancel so quickly? -> High bill shock ($70+/mo) on Fiber Optic plans.
  - **Why_3**: Why does bill shock cause cancellation? -> Fiber optic plans were purchased without Tech Support or Device Protection.
  - **Why_4**: Why did they not subscribe to Tech Support? -> Bundled options were not presented during initial unassisted checkout.
  - **Why_5**: Root Cause: High-speed fiber customers experience technical friction with no proactive onboarding or discounted support bundling.

## A/B Test Results (Proactive $15/mo Credit)
- **Control Churn**: 38.0%
- **Treatment Churn**: 24.5%
- **Absolute Risk Reduction**: 13.5%
- **Relative Risk Reduction**: 35.53%
- **p-value**: 7.38e-11 (Statistically Significant: True)

## GenAI Evaluation (LLM-as-Judge, RAGAS, LoRA)
- **LLM Judge Approval Rate**: 50.0%
- **RAGAS Retrieval Score**: 0.956 (Faithfulness: 0.99)
- **LoRA Adapter Footprint**: 16,777,216 params (0.2089% of Llama-3-8B)
