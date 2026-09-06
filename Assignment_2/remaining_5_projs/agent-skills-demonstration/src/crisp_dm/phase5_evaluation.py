"""CRISP-DM Phase 5: Evaluation.

Orchestrates:
- model-evaluation (param087/agent-ml-skills)
- cohort-analysis (nimrodfisher/data-analytics-skills)
- root-cause-investigation (nimrodfisher/data-analytics-skills)
- ab-test-analysis (nimrodfisher/data-analytics-skills)
- analysis-documentation (nimrodfisher/data-analytics-skills)
- analysis-qa-checklist (nimrodfisher/data-analytics-skills)
- llm-as-judge (param087/agent-ml-skills)
- ragas-evaluation (param087/agent-ml-skills)
- lora-finetuning (param087/agent-ml-skills)
"""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.config import ARTIFACTS_DIR, DATASET_PATH
from src.modules.eda_profiler import load_raw_dataset
from src.modules.evaluation_analytics import (
    build_analysis_qa_checklist,
    evaluate_model_performance,
    execute_ab_test_analysis,
    execute_cohort_analysis,
    execute_root_cause_investigation,
)
from src.modules.genai_evaluation import (
    calculate_lora_finetuning_specs,
    evaluate_retention_interventions_as_judge,
    execute_ragas_evaluation,
)
from src.modules.model_pipeline import build_model_pipelines
from src.modules.preprocessor import prepare_data_splits


def run_phase5_evaluation(
    dataset_path: Path = DATASET_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR / "phase5_evaluation",
    splits: Dict[str, Any] = None,
    champion_pipeline: Any = None,
) -> Dict[str, Any]:
    """Execute all Phase 5 Evaluation skills and generate artifacts."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataset(dataset_path)

    if splits is None:
        splits = prepare_data_splits(df)

    if champion_pipeline is None:
        num_cols = splits["context_package"]["numerical_features"]
        cat_cols = splits["context_package"]["categorical_features"]
        pipes = build_model_pipelines(num_cols, cat_cols)
        champion_pipeline = pipes["LogisticRegression"]
        champion_pipeline.fit(splits["X_train"], splits["y_train"])

    # 1. Model Evaluation Metrics & Curves
    eval_metrics, roc_fig = evaluate_model_performance(
        champion_pipeline, splits["X_test"], splits["y_test"]
    )

    # 2. Cohort Analysis & Retention Heatmap
    cohort_report, cohort_fig = execute_cohort_analysis(df)

    # 3. Root Cause Investigation & 5-Whys
    root_cause_report = execute_root_cause_investigation(
        champion_pipeline, splits["X_train"], splits["feature_cols"]
    )

    # 4. A/B Test Statistical Analysis
    ab_test_report = execute_ab_test_analysis()

    # 5. Analysis QA Checklist
    qa_checklist = build_analysis_qa_checklist(eval_metrics, ab_test_report)

    # 6. LLM-as-Judge Evaluation
    llm_judge_report = evaluate_retention_interventions_as_judge()

    # 7. RAGAS Retrieval Evaluation
    ragas_report = execute_ragas_evaluation()

    # 8. LoRA Fine-Tuning Specification
    lora_report = calculate_lora_finetuning_specs()

    # 9. Analysis Documentation Framework
    analysis_doc = {
        "title": "Comprehensive Telco Churn Evaluation & Root Cause Synthesis",
        "author": "Antigravity AI Agent & Analytics Engineering Team",
        "methodology": (
            "Leakage-safe 80/20 train/test partition, Stratified 5-Fold Cross-Validation, "
            "cost-sensitive classification, two-sample z-test hypothesis testing, and multi-criteria LLM evaluation."
        ),
        "key_findings": [
            f"Holdout ROC-AUC of {eval_metrics['roc_auc']} and PR-AUC of {eval_metrics['pr_auc']}.",
            f"Calibrated optimal decision threshold: {eval_metrics['optimal_threshold']} yields {eval_metrics['recall_at_optimal']*100:.1f}% recall at {eval_metrics['precision_at_optimal']*100:.1f}% precision.",
            f"A/B trial demonstrates statistically significant churn reduction (p={ab_test_report['statistical_metrics']['p_value']}, ARR={ab_test_report['statistical_metrics']['absolute_churn_reduction_pct']}%).",
            f"GenAI retention messaging achieves {llm_judge_report['approval_rate_pct']}% judge approval and RAGAS score of {ragas_report['ragas_composite_score']}.",
        ],
    }

    phase5_summary = {
        "crisp_dm_phase": "Phase 5: Evaluation",
        "status": "COMPLETED",
        "skills_executed": [
            "model-evaluation",
            "cohort-analysis",
            "root-cause-investigation",
            "ab-test-analysis",
            "analysis-documentation",
            "analysis-qa-checklist",
            "llm-as-judge",
            "ragas-evaluation",
            "lora-finetuning",
        ],
        "evaluation_metrics": eval_metrics,
        "cohort_analysis": cohort_report,
        "root_cause_investigation": root_cause_report,
        "ab_test_analysis": ab_test_report,
        "analysis_qa_checklist": qa_checklist,
        "llm_as_judge": llm_judge_report,
        "ragas_evaluation": ragas_report,
        "lora_finetuning": lora_report,
        "analysis_documentation": analysis_doc,
    }

    # Persist JSON artifact
    json_path = artifacts_dir / "phase5_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(phase5_summary, f, indent=2)

    # Persist Markdown report
    md_path = artifacts_dir / "phase5_report.md"
    with open(md_path, "w") as f:
        f.write(f"""# CRISP-DM Phase 5: Evaluation Report

## Model Evaluation Metrics (Holdout Test Set)
- **ROC-AUC**: {eval_metrics['roc_auc']}
- **PR-AUC**: {eval_metrics['pr_auc']}
- **Brier Calibration Score**: {eval_metrics['brier_score']}
- **Optimal Decision Threshold**: {eval_metrics['optimal_threshold']} (F1 = {eval_metrics['f1_at_optimal']})
- **Precision**: {eval_metrics['precision_at_optimal']} | **Recall**: {eval_metrics['recall_at_optimal']}
- **Confusion Matrix**: True Negatives: {eval_metrics['confusion_matrix']['true_negatives']:,}, False Positives: {eval_metrics['confusion_matrix']['false_positives']:,}, False Negatives: {eval_metrics['confusion_matrix']['false_negatives']:,}, True Positives: {eval_metrics['confusion_matrix']['true_positives']:,}

## Cohort Analysis & Retention Dynamics
- **High Risk Cohort**: `{cohort_report['highest_churn_segment']}`
- **Low Risk Cohort**: `{cohort_report['lowest_churn_segment']}`
- **Key Takeaway**: `{cohort_report['key_takeaway']}`

## Root Cause 5-Whys Diagnostic
- **Core Problem**: {root_cause_report['problem_statement']}
- **5-Whys Progression**:
{chr(10).join([f"  - **{k}**: {v}" for k, v in root_cause_report['five_whys_analysis'].items()])}

## A/B Test Results (Proactive $15/mo Credit)
- **Control Churn**: {ab_test_report['observed_churn_rates']['control_churn_pct']}%
- **Treatment Churn**: {ab_test_report['observed_churn_rates']['treatment_churn_pct']}%
- **Absolute Risk Reduction**: {ab_test_report['statistical_metrics']['absolute_churn_reduction_pct']}%
- **Relative Risk Reduction**: {ab_test_report['statistical_metrics']['relative_churn_reduction_pct']}%
- **p-value**: {ab_test_report['statistical_metrics']['p_value']} (Statistically Significant: {ab_test_report['statistical_metrics']['statistically_significant']})

## GenAI Evaluation (LLM-as-Judge, RAGAS, LoRA)
- **LLM Judge Approval Rate**: {llm_judge_report['approval_rate_pct']}%
- **RAGAS Retrieval Score**: {ragas_report['ragas_composite_score']} (Faithfulness: {ragas_report['metrics_summary']['faithfulness']})
- **LoRA Adapter Footprint**: {lora_report['trainable_parameters']:,} params ({lora_report['trainable_parameters_percentage']}% of Llama-3-8B)
""")

    return phase5_summary


if __name__ == "__main__":
    res = run_phase5_evaluation()
    print("Phase 5 completed! ROC-AUC:", res["evaluation_metrics"]["roc_auc"], "QA:", res["analysis_qa_checklist"]["overall_status"])
