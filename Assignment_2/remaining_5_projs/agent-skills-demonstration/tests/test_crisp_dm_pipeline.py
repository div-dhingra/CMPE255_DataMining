"""Test suite verifying end-to-end CRISP-DM 6-Phase execution."""

from pathlib import Path
from src.config import ARTIFACTS_DIR
from src.crisp_dm.phase1_business_understanding import run_phase1_business_understanding
from src.crisp_dm.phase2_data_understanding import run_phase2_data_understanding
from src.crisp_dm.phase3_data_preparation import run_phase3_data_preparation
from src.crisp_dm.phase4_modeling import run_phase4_modeling
from src.crisp_dm.phase5_evaluation import run_phase5_evaluation
from src.crisp_dm.phase6_deployment_monitoring import run_phase6_deployment_monitoring


def test_crisp_dm_phase1():
    res = run_phase1_business_understanding()
    assert res["status"] == "COMPLETED"
    assert (ARTIFACTS_DIR / "phase1_business_understanding" / "phase1_report.md").exists()
    assert (ARTIFACTS_DIR / "phase1_business_understanding" / "phase1_business_understanding.json").exists()


def test_crisp_dm_phase2():
    res = run_phase2_data_understanding()
    assert res["status"] == "COMPLETED"
    assert (ARTIFACTS_DIR / "phase2_data_understanding" / "phase2_report.md").exists()
    assert (ARTIFACTS_DIR / "phase2_data_understanding" / "phase2_data_understanding.json").exists()


def test_crisp_dm_phase3():
    res = run_phase3_data_preparation()
    assert res["status"] == "COMPLETED"
    assert res["engineered_feature_count"] >= 20
    assert (ARTIFACTS_DIR / "phase3_data_preparation" / "phase3_report.md").exists()


def test_crisp_dm_phase4():
    res = run_phase4_modeling()
    assert res["status"] == "COMPLETED"
    assert res["champion_cv_roc_auc"] > 0.80
    assert (ARTIFACTS_DIR / "phase4_modeling" / "phase4_report.md").exists()


def test_crisp_dm_phase5():
    res = run_phase5_evaluation()
    assert res["status"] == "COMPLETED"
    assert res["evaluation_metrics"]["roc_auc"] > 0.80
    assert (ARTIFACTS_DIR / "phase5_evaluation" / "phase5_report.md").exists()


def test_crisp_dm_phase6():
    res = run_phase6_deployment_monitoring()
    assert res["status"] == "COMPLETED"
    assert res["financial_impact"]["return_on_investment_roi_pct"] > 100.0
    assert (ARTIFACTS_DIR / "phase6_deployment" / "phase6_report.md").exists()
