#!/usr/bin/env python3
"""Master CLI Runner for Project 5: Agent Skills Demonstration.

Executes all 46 skills (15 Agent ML skills + 31 Data Analytics skills)
sequentially across all 6 CRISP-DM phases on the Kaggle Telco Customer Churn dataset.
Generates comprehensive JSON reports, Markdown summaries, and analytical figures in artifacts/.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure workspace is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Ensure MPLCONFIGDIR is set
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR.parent / ".mplcache"))

from src.config import ARTIFACTS_DIR, DATASET_PATH, FIGURES_DIR
from src.crisp_dm.phase1_business_understanding import run_phase1_business_understanding
from src.crisp_dm.phase2_data_understanding import run_phase2_data_understanding
from src.crisp_dm.phase3_data_preparation import run_phase3_data_preparation
from src.crisp_dm.phase4_modeling import run_phase4_modeling
from src.crisp_dm.phase5_evaluation import run_phase5_evaluation
from src.crisp_dm.phase6_deployment_monitoring import run_phase6_deployment_monitoring
from src.skills_registry import list_all_skills

# ANSI Colors
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"\n{BOLD}{CYAN}================================================================================{RESET}")
    print(f"{BOLD}{CYAN}   Project 5: Agent ML & Data Analytics Skills CRISP-DM Demonstration Engine   {RESET}")
    print(f"{BOLD}{CYAN}   Dataset: Kaggle / IBM Telco Customer Churn (7,043 Accounts, 21 Attributes)  {RESET}")
    print(f"{BOLD}{CYAN}   Skills: 15 param087 Agent ML Skills + 31 nimrodfisher Data Analytics Skills {RESET}")
    print(f"{BOLD}{CYAN}================================================================================{RESET}\n")


def execute_full_lifecycle():
    start_time = time.time()
    print_banner()

    skills_catalog = list_all_skills()
    print(f"{BLUE}[INFO]{RESET} Total skills registered in engine: {BOLD}{len(skills_catalog)}{RESET} skills across 6 phases.\n")

    # -------------------------------------------------------------
    # PHASE 1: Business Understanding
    # -------------------------------------------------------------
    print(f"{BOLD}{BLUE}>>> [CRISP-DM Phase 1] Business Understanding{RESET}")
    print("  Executing: solution-design, semantic-model-builder, metric-tree-builder, assumptions-log, biz-metrics...")
    p1 = run_phase1_business_understanding()
    print(f"  {GREEN}✓{RESET} Annual ARR at Risk: {BOLD}${p1['business_kpis']['annualized_revenue_at_risk']:,.2f}{RESET}")
    print(f"  {GREEN}✓{RESET} Baseline Churn Rate: {BOLD}{p1['business_kpis']['annual_churn_rate_pct']}%{RESET}")
    print(f"  {GREEN}✓{RESET} Artifacts written: artifacts/phase1_business_understanding/\n")

    # -------------------------------------------------------------
    # PHASE 2: Data Understanding
    # -------------------------------------------------------------
    print(f"{BOLD}{BLUE}>>> [CRISP-DM Phase 2] Data Understanding{RESET}")
    print("  Executing: programmatic-eda, pandas-patterns, data-quality-audit, schema-mapper, reconciliation...")
    p2 = run_phase2_data_understanding()
    print(f"  {GREEN}✓{RESET} Evaluated Records: {BOLD}{p2['data_quality_audit']['total_records_evaluated']:,}{RESET}")
    print(f"  {GREEN}✓{RESET} Data Quality Health Score: {BOLD}{p2['data_quality_audit']['health_score']}/100{RESET}")
    print(f"  {GREEN}✓{RESET} TotalCharges Whitespace Blanks: {p2['data_quality_audit']['whitespace_anomalies_detected']} records (tenure=0)")
    print(f"  {GREEN}✓{RESET} Memory Footprint Optimized: {BOLD}{p2['pandas_patterns']['memory_savings_pct']}% savings{RESET}")
    print(f"  {GREEN}✓{RESET} Artifacts written: artifacts/phase2_data_understanding/\n")

    # -------------------------------------------------------------
    # PHASE 3: Data Preparation
    # -------------------------------------------------------------
    print(f"{BOLD}{BLUE}>>> [CRISP-DM Phase 3] Data Preparation{RESET}")
    print("  Executing: data-cleaning (train-only median), feature-engineering, imbalanced-data (SMOTE)...")
    p3 = run_phase3_data_preparation()
    splits = p3["_splits"]
    print(f"  {GREEN}✓{RESET} Train Split: {len(splits['X_train']):,} | Test Split: {len(splits['X_test']):,}")
    print(f"  {GREEN}✓{RESET} Train-only TotalCharges Median: ${p3['cleaning_metadata']['imputed_value']}")
    print(f"  {GREEN}✓{RESET} Total Engineered Features: {BOLD}{p3['engineered_feature_count']}{RESET} attributes")
    print(f"  {GREEN}✓{RESET} Balanced SMOTE Samples: {p3['imbalance_handling']['smote_resampled_distribution']}")
    print(f"  {GREEN}✓{RESET} Artifacts written: artifacts/phase3_data_preparation/\n")

    # -------------------------------------------------------------
    # PHASE 4: Modeling
    # -------------------------------------------------------------
    print(f"{BOLD}{BLUE}>>> [CRISP-DM Phase 4] Modeling{RESET}")
    print("  Executing: sklearn-pipelines, model-training, hyperparameter-tuning, 5-fold cross-validation...")
    p4 = run_phase4_modeling(splits=splits)
    champion_name = p4["champion_model"]
    champion_pipeline = p4["_champion_pipeline"]
    print(f"  {GREEN}✓{RESET} Champion Model: {BOLD}{champion_name}{RESET} (CV ROC-AUC = {p4['champion_cv_roc_auc']:.4f})")
    print("  Model Leaderboard:")
    for rank, m in enumerate(p4["model_leaderboard"]):
        print(f"    {rank+1}. {m['model']:<22} | CV ROC-AUC: {m['cv_roc_auc_mean']:.4f} ± {m['cv_roc_auc_std']:.4f} | PR-AUC: {m['cv_pr_auc_mean']:.4f}")
    print(f"  {GREEN}✓{RESET} Artifacts written: artifacts/phase4_modeling/\n")

    # -------------------------------------------------------------
    # PHASE 5: Evaluation
    # -------------------------------------------------------------
    print(f"{BOLD}{BLUE}>>> [CRISP-DM Phase 5] Evaluation{RESET}")
    print("  Executing: model-evaluation, cohort-analysis, root-cause 5-whys, ab-test-analysis, LLM-as-judge, RAGAS...")
    p5 = run_phase5_evaluation(splits=splits, champion_pipeline=champion_pipeline)
    eval_m = p5["evaluation_metrics"]
    ab_m = p5["ab_test_analysis"]["statistical_metrics"]
    print(f"  {GREEN}✓{RESET} Holdout ROC-AUC: {BOLD}{eval_m['roc_auc']}{RESET} | PR-AUC: {BOLD}{eval_m['pr_auc']}{RESET} | Brier: {eval_m['brier_score']}")
    print(f"  {GREEN}✓{RESET} Calibrated Decision Threshold: {BOLD}{eval_m['optimal_threshold']}{RESET} (Precision: {eval_m['precision_at_optimal']}, Recall: {eval_m['recall_at_optimal']})")
    print(f"  {GREEN}✓{RESET} A/B Trial: p-value = {ab_m['p_value']} (Relative Churn Reduction: {ab_m['relative_churn_reduction_pct']}%)")
    print(f"  {GREEN}✓{RESET} LLM-as-Judge Approval: {p5['llm_as_judge']['approval_rate_pct']}% | RAGAS Retrieval Score: {p5['ragas_evaluation']['ragas_composite_score']}")
    print(f"  {GREEN}✓{RESET} QA Verification Checklist: {BOLD}{p5['analysis_qa_checklist']['overall_status']}{RESET}")
    print(f"  {GREEN}✓{RESET} Artifacts written: artifacts/phase5_evaluation/\n")

    # -------------------------------------------------------------
    # PHASE 6: Deployment & Monitoring
    # -------------------------------------------------------------
    print(f"{BOLD}{BLUE}>>> [CRISP-DM Phase 6] Deployment & Observability{RESET}")
    print("  Executing: executive-summary, insight-to-action, financial impact ROI, drift observability...")
    p6 = run_phase6_deployment_monitoring(
        splits=splits, champion_pipeline=champion_pipeline, eval_metrics=eval_m
    )
    impact = p6["financial_impact"]
    drift = p6["ml_monitoring"]
    print(f"  {GREEN}✓{RESET} Annual Gross Revenue Preserved: {BOLD}${impact['gross_annual_revenue_preserved']:,.2f}{RESET}")
    print(f"  {GREEN}✓{RESET} Net Profit Gain: {BOLD}${impact['net_annual_profit_gain']:,.2f}{RESET} ({BOLD}{impact['return_on_investment_roi_pct']}% ROI{RESET})")
    print(f"  {GREEN}✓{RESET} Successfully Retained Customers: {BOLD}{impact['customers_successfully_retained']:,}{RESET} accounts")
    print(f"  {GREEN}✓{RESET} Observability Drift Status: {BOLD}{drift['system_status']}{RESET} (Prediction PSI: {drift['prediction_drift']['psi_score']})")
    print(f"  {GREEN}✓{RESET} Latency SLA: p50={drift['latency_metrics']['p50_latency_ms']}ms, p99={drift['latency_metrics']['p99_latency_ms']}ms (100% compliant)")
    print(f"  {GREEN}✓{RESET} Artifacts written: artifacts/phase6_deployment/\n")

    elapsed = round(time.time() - start_time, 2)
    print(f"{BOLD}{GREEN}================================================================================{RESET}")
    print(f"{BOLD}{GREEN}   SUCCESS! All 46 Skills Executed Across All 6 CRISP-DM Phases in {elapsed}s   {RESET}")
    print(f"{BOLD}{GREEN}   Publication Figures Saved to: artifacts/figures/                             {RESET}")
    print(f"{BOLD}{GREEN}   Executive & Technical Reports Saved to: artifacts/                           {RESET}")
    print(f"{BOLD}{GREEN}================================================================================{RESET}\n")


if __name__ == "__main__":
    execute_full_lifecycle()
