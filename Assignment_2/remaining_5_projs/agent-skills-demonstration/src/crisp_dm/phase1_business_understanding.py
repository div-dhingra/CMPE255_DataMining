"""CRISP-DM Phase 1: Business Understanding.

Orchestrates:
- solution-design (param087/agent-ml-skills)
- semantic-model-builder (nimrodfisher/data-analytics-skills)
- metric-tree-builder (nimrodfisher/data-analytics-skills)
- analysis-assumptions-log (nimrodfisher/data-analytics-skills)
- business-metrics-calculator (nimrodfisher/data-analytics-skills)
- stakeholder-requirements-gathering (nimrodfisher/data-analytics-skills)
- analysis-planning (nimrodfisher/data-analytics-skills)
"""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.config import ARTIFACTS_DIR, DATASET_PATH
from src.modules.business_strategy import (
    build_metric_tree,
    build_semantic_model,
    build_solution_design,
    calculate_business_metrics,
    get_analysis_assumptions_log,
)
from src.modules.eda_profiler import load_raw_dataset


def run_phase1_business_understanding(
    dataset_path: Path = DATASET_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR / "phase1_business_understanding",
) -> Dict[str, Any]:
    """Execute all Phase 1 Business Understanding skills and generate artifacts."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataset(dataset_path)

    # 1. Solution Design
    solution_design = build_solution_design()

    # 2. Semantic Model
    semantic_model = build_semantic_model()

    # 3. Metric Tree
    metric_tree = build_metric_tree()

    # 4. Assumptions Log
    assumptions_log = get_analysis_assumptions_log()

    # 5. Business Recurring Metrics
    business_metrics = calculate_business_metrics(df)

    # 6. Stakeholder Requirements
    stakeholder_reqs = {
        "primary_stakeholders": ["VP of Customer Success", "Head of Growth", "Retention Operations Team"],
        "core_questions_to_answer": [
            "Which customer segments are driving the majority of churn?",
            "Can we reliably identify high-risk churners at least 30 days in advance?",
            "What proactive interventions provide the highest positive ROI?",
        ],
        "acceptance_criteria": [
            "Model achieves PR-AUC >= 0.60 on holdout validation data.",
            "Explainable top 3 churn risk drivers delivered with every predicted probability.",
            "Proactive campaign ROI exceeds 100% net of intervention costs.",
        ],
    }

    # 7. Analysis Planning (Work Breakdown Structure)
    analysis_plan = {
        "phase_1": "Business Understanding: Formulate KPIs, metric tree, and requirements (Completed)",
        "phase_2": "Data Understanding: Profile data, assess quality, map schema, reconcile charges",
        "phase_3": "Data Preparation: Zero-leakage cleaning, feature engineering, handle imbalanced classes",
        "phase_4": "Modeling: Train, tune, and cross-validate Scikit-Learn pipelines",
        "phase_5": "Evaluation: Evaluate PR/ROC curves, cohorts, root cause, A/B test, GenAI evaluation",
        "phase_6": "Deployment & Observability: Executive summary, dashboard specs, drift monitoring",
    }

    phase1_summary = {
        "crisp_dm_phase": "Phase 1: Business Understanding",
        "status": "COMPLETED",
        "skills_executed": [
            "solution-design",
            "semantic-model-builder",
            "metric-tree-builder",
            "analysis-assumptions-log",
            "business-metrics-calculator",
            "stakeholder-requirements-gathering",
            "analysis-planning",
        ],
        "business_kpis": business_metrics,
        "solution_design": solution_design,
        "semantic_model": semantic_model,
        "metric_tree": metric_tree,
        "assumptions": assumptions_log,
        "stakeholder_requirements": stakeholder_reqs,
        "analysis_plan": analysis_plan,
    }

    # Persist JSON artifact
    json_path = artifacts_dir / "phase1_business_understanding.json"
    with open(json_path, "w") as f:
        json.dump(phase1_summary, f, indent=2)

    # Persist Markdown artifact report
    md_path = artifacts_dir / "phase1_report.md"
    with open(md_path, "w") as f:
        f.write(f"""# CRISP-DM Phase 1: Business Understanding Report

## Executive Summary
- **Business Goal**: {solution_design['business_objective']}
- **Annual Revenue at Stake**: ${business_metrics['annualized_revenue_at_risk']:,.2f}
- **Active Customer Base**: {business_metrics['total_active_customers']:,} accounts
- **Baseline Churn Rate**: {business_metrics['annual_churn_rate_pct']}%

## Solution Architecture
- **Target Definition**: `{solution_design['target_definition']}`
- **Primary Optimization Metric**: `{solution_design['primary_metric']}`
- **Champion Architecture**: `{solution_design['champion_model_architecture']}`

## Metric Tree Hierarchy
- **Root KPI**: {metric_tree['root_kpi']}
- **Decomposition**: {metric_tree['decomposition']['formula']}
- **Primary Churn Drivers**:
{chr(10).join([f"  - {d['driver']} ({d['share_pct']}%)" for d in metric_tree['decomposition']['branches'][2]['sub_drivers']])}

## Key Assumptions & Sensitivity
{chr(10).join([f"- **{a['id']}**: {a['assumption']} *(Sensitivity: {a['sensitivity']})*" for a in assumptions_log])}
""")

    return phase1_summary


if __name__ == "__main__":
    res = run_phase1_business_understanding()
    print("Phase 1 completed! Revenue at risk: $", res["business_kpis"]["annualized_revenue_at_risk"])
