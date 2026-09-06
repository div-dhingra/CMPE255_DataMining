"""FastAPI Router for CRISP-DM Phases and Artifacts."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from src.config import ARTIFACTS_DIR

router = APIRouter(prefix="/api/crisp", tags=["CRISP-DM Lifecycle"])

PHASES_META = [
    {
        "phase_id": "phase1",
        "name": "Phase 1: Business Understanding",
        "description": "Formulate retention objectives, revenue impact, semantic models, and metric trees.",
        "skills": ["solution-design", "semantic-model-builder", "metric-tree-builder", "analysis-assumptions-log", "business-metrics-calculator", "stakeholder-requirements-gathering", "analysis-planning"],
        "artifact_dir": "phase1_business_understanding",
        "json_file": "phase1_business_understanding.json",
        "md_file": "phase1_report.md",
    },
    {
        "phase_id": "phase2",
        "name": "Phase 2: Data Understanding",
        "description": "Programmatic EDA, data quality audit, schema mapper, metric reconciliation, and pandas patterns.",
        "skills": ["exploratory-data-analysis", "programmatic-eda", "pandas-patterns", "data-quality-audit", "schema-mapper", "metric-reconciliation", "query-validation", "data-catalog-entry", "sql-to-business-logic"],
        "artifact_dir": "phase2_data_understanding",
        "json_file": "phase2_data_understanding.json",
        "md_file": "phase2_report.md",
    },
    {
        "phase_id": "phase3",
        "name": "Phase 3: Data Preparation",
        "description": "Zero-leakage data cleaning, domain feature engineering, and SMOTE class balancing.",
        "skills": ["data-cleaning", "feature-engineering", "imbalanced-data", "context-packager"],
        "artifact_dir": "phase3_data_preparation",
        "json_file": "phase3_data_preparation.json",
        "md_file": "phase3_report.md",
    },
    {
        "phase_id": "phase4",
        "name": "Phase 4: Modeling",
        "description": "Scikit-learn pipelines, model training, stratified cross-validation, and hyperparameter tuning.",
        "skills": ["sklearn-pipelines", "model-training", "hyperparameter-tuning", "cross-validation"],
        "artifact_dir": "phase4_modeling",
        "json_file": "phase4_modeling.json",
        "md_file": "phase4_report.md",
    },
    {
        "phase_id": "phase5",
        "name": "Phase 5: Evaluation",
        "description": "Model metrics (ROC/PR), cohort analysis, root-cause 5-whys, A/B test analysis, and GenAI evaluation.",
        "skills": ["model-evaluation", "cohort-analysis", "root-cause-investigation", "ab-test-analysis", "analysis-documentation", "analysis-qa-checklist", "llm-as-judge", "ragas-evaluation", "lora-finetuning"],
        "artifact_dir": "phase5_evaluation",
        "json_file": "phase5_evaluation.json",
        "md_file": "phase5_report.md",
    },
    {
        "phase_id": "phase6",
        "name": "Phase 6: Deployment & Monitoring",
        "description": "Executive summary, insight-to-action matrix, financial ROI impact, and drift observability.",
        "skills": ["executive-summary-generator", "insight-to-action", "impact-quantification", "dashboard-specification", "visualization-builder", "data-narrative-builder", "technical-to-business-translator", "methodology-explainer", "ml-monitoring-observability", "peer-review-template", "analysis-retrospective"],
        "artifact_dir": "phase6_deployment",
        "json_file": "phase6_deployment.json",
        "md_file": "phase6_report.md",
    },
]


@router.get("/phases")
def list_phases():
    """List all 6 CRISP-DM phases with metadata and execution status."""
    return {"total_phases": len(PHASES_META), "phases": PHASES_META}


@router.get("/{phase_id}")
def get_phase_artifacts(phase_id: str):
    """Retrieve full generated artifacts and markdown report for a specific phase."""
    meta = next((p for p in PHASES_META if p["phase_id"] == phase_id), None)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Phase '{phase_id}' not recognized.")

    phase_dir = ARTIFACTS_DIR / meta["artifact_dir"]
    json_path = phase_dir / meta["json_file"]
    md_path = phase_dir / meta["md_file"]

    json_data = {}
    if json_path.exists():
        with open(json_path, "r") as f:
            json_data = json.load(f)

    md_content = ""
    if md_path.exists():
        with open(md_path, "r") as f:
            md_content = f.read()

    return {
        "metadata": meta,
        "artifact_data": json_data,
        "markdown_report": md_content,
    }
