"""Test suite for the 31 skills from nimrodfisher/data-analytics-skills across 6 categories."""

from src.skills_registry import execute_skill_by_id


# ----------------------------------------------------------------------
# Category 01: Data Quality & Validation (5 skills)
# ----------------------------------------------------------------------
def test_programmatic_eda():
    res = execute_skill_by_id("programmatic-eda")
    assert res["status"] == "SUCCESS"
    assert "missing_values" in res["output"]


def test_data_quality_audit():
    res = execute_skill_by_id("data-quality-audit")
    assert res["status"] == "SUCCESS"
    assert res["output"]["primary_key_unique"] is True
    assert res["output"]["whitespace_anomalies_detected"] == 11


def test_query_validation():
    res = execute_skill_by_id("query-validation")
    assert res["status"] == "SUCCESS"
    assert res["output"]["all_valid"] is True


def test_schema_mapper():
    res = execute_skill_by_id("schema-mapper")
    assert res["status"] == "SUCCESS"
    assert "Customer" in res["output"]["entities"]


def test_metric_reconciliation():
    res = execute_skill_by_id("metric-reconciliation")
    assert res["status"] == "SUCCESS"
    assert "RECONCILED" in res["output"]["status"]


# ----------------------------------------------------------------------
# Category 02: Documentation & Knowledge (5 skills)
# ----------------------------------------------------------------------
def test_semantic_model_builder():
    res = execute_skill_by_id("semantic-model-builder")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["measures"]) >= 3


def test_analysis_documentation():
    res = execute_skill_by_id("analysis-documentation")
    assert res["status"] == "SUCCESS"
    assert res["output"]["status"] == "APPROVED"


def test_data_catalog_entry():
    res = execute_skill_by_id("data-catalog-entry")
    assert res["status"] == "SUCCESS"
    assert res["output"]["grain"] == "customerID"


def test_sql_to_business_logic():
    res = execute_skill_by_id("sql-to-business-logic")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["translated_expressions"]) >= 2


def test_analysis_assumptions_log():
    res = execute_skill_by_id("analysis-assumptions-log")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["assumptions"]) >= 3


# ----------------------------------------------------------------------
# Category 03: Data Analysis & Investigation (7 skills)
# ----------------------------------------------------------------------
def test_cohort_analysis():
    res = execute_skill_by_id("cohort-analysis")
    assert res["status"] == "SUCCESS"
    assert "cohort_churn_matrix_pct" in res["output"]


def test_segmentation_analysis():
    res = execute_skill_by_id("segmentation-analysis")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["segments_identified"]) >= 3


def test_funnel_analysis():
    res = execute_skill_by_id("funnel-analysis")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["funnel_stages"]) == 4


def test_time_series_analysis():
    res = execute_skill_by_id("time-series-analysis")
    assert res["status"] == "SUCCESS"
    assert "hazard_rate_curve" in res["output"]


def test_root_cause_investigation():
    res = execute_skill_by_id("root-cause-investigation")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["five_whys_analysis"]) == 5


def test_ab_test_analysis():
    res = execute_skill_by_id("ab-test-analysis")
    assert res["status"] == "SUCCESS"
    assert res["output"]["statistical_metrics"]["statistically_significant"] is True
    assert res["output"]["statistical_metrics"]["p_value"] < 0.05


def test_business_metrics_calculator():
    res = execute_skill_by_id("business-metrics-calculator")
    assert res["status"] == "SUCCESS"
    assert res["output"]["annualized_revenue_at_risk"] > 1_000_000


# ----------------------------------------------------------------------
# Category 04: Data Storytelling & Visualization (5 skills)
# ----------------------------------------------------------------------
def test_insight_synthesis():
    res = execute_skill_by_id("insight-synthesis")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["top_insights"]) >= 3


def test_visualization_builder():
    res = execute_skill_by_id("visualization-builder")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["visualizations_built"]) >= 6


def test_executive_summary_generator():
    res = execute_skill_by_id("executive-summary-generator")
    assert res["status"] == "SUCCESS"
    assert res["output"]["status"] == "DEPLOYMENT_RECOMMENDED"


def test_dashboard_specification():
    res = execute_skill_by_id("dashboard-specification")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["layout_hierarchy"]) == 3


def test_data_narrative_builder():
    res = execute_skill_by_id("data-narrative-builder")
    assert res["status"] == "SUCCESS"
    assert "hook" in res["output"]["narrative_arc"]


# ----------------------------------------------------------------------
# Category 05: Stakeholder Communication (5 skills)
# ----------------------------------------------------------------------
def test_technical_to_business_translator():
    res = execute_skill_by_id("technical-to-business-translator")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["translations"]) >= 2


def test_stakeholder_requirements_gathering():
    res = execute_skill_by_id("stakeholder-requirements-gathering")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["stakeholders"]) >= 3


def test_analysis_qa_checklist():
    res = execute_skill_by_id("analysis-qa-checklist")
    assert res["status"] == "SUCCESS"
    assert res["output"]["overall_status"] == "READY_FOR_DEPLOYMENT"


def test_methodology_explainer():
    res = execute_skill_by_id("methodology-explainer")
    assert res["status"] == "SUCCESS"
    assert "Logistic Regression" in res["output"]["models_explained"]


def test_impact_quantification():
    res = execute_skill_by_id("impact-quantification")
    assert res["status"] == "SUCCESS"
    assert res["output"]["net_annual_profit_gain"] > 200_000
    assert res["output"]["return_on_investment_roi_pct"] > 100.0


# ----------------------------------------------------------------------
# Category 06: Workflow Optimization (4 skills)
# ----------------------------------------------------------------------
def test_analysis_planning():
    res = execute_skill_by_id("analysis-planning")
    assert res["status"] == "SUCCESS"
    assert res["output"]["phases_completed"] == 6


def test_context_packager():
    res = execute_skill_by_id("context-packager")
    assert res["status"] == "SUCCESS"
    assert "train_samples" in res["output"]


def test_peer_review_template():
    res = execute_skill_by_id("peer-review-template")
    assert res["status"] == "SUCCESS"
    assert res["output"]["data_leakage_check"] == "PASS"


def test_analysis_retrospective():
    res = execute_skill_by_id("analysis-retrospective")
    assert res["status"] == "SUCCESS"
    assert len(res["output"]["retrospective_findings"]) >= 2
