"""Tests for the CRISP-DM framework lifecycle manager and audit trail."""

from backend.src.core.crisp_dm import CrispDMLifecycle


def test_crisp_dm_initialization():
    lifecycle = CrispDMLifecycle()
    summary = lifecycle.get_lifecycle_summary()

    assert summary["total_phases"] == 6
    assert summary["completed_phases"] == 6
    assert len(summary["phases"]) == 6

    phase_names = [p["name"] for p in summary["phases"]]
    assert "Business Understanding" in phase_names
    assert "Data Understanding" in phase_names
    assert "Data Preparation" in phase_names
    assert "Modeling" in phase_names
    assert "Evaluation" in phase_names
    assert "Deployment" in phase_names


def test_crisp_dm_update_phase():
    lifecycle = CrispDMLifecycle()
    lifecycle.update_phase(phase_id=4, status="IN_PROGRESS", metrics={"active_run": "apriori"})

    summary = lifecycle.get_lifecycle_summary()
    phase4 = [p for p in summary["phases"] if p["phase_id"] == 4][0]

    assert phase4["status"] == "IN_PROGRESS"
    assert phase4["key_metrics"]["active_run"] == "apriori"
    assert summary["completed_phases"] == 5
