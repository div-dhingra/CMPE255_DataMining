"""Test suite verifying all FastAPI REST API endpoints."""

from fastapi.testclient import TestClient


def test_api_health(api_client: TestClient):
    res = api_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["skills_total"] == 46
    assert data["crisp_dm_phases"] == 6


def test_api_list_skills(api_client: TestClient):
    res = api_client.get("/api/skills")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 46
    assert len(data["skills"]) == 46


def test_api_get_skill_spec(api_client: TestClient):
    res = api_client.get("/api/skills/imbalanced-data")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "imbalanced-data"
    assert "markdown_specification" in data
    assert len(data["markdown_specification"]) > 100


def test_api_run_skill_live(api_client: TestClient):
    res = api_client.post("/api/skills/solution-design/run")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "business_objective" in data["output"]


def test_api_crisp_phases(api_client: TestClient):
    res = api_client.get("/api/crisp/phases")
    assert res.status_code == 200
    data = res.json()
    assert data["total_phases"] == 6


def test_api_crisp_phase_detail(api_client: TestClient):
    res = api_client.get("/api/crisp/phase1")
    assert res.status_code == 200
    data = res.json()
    assert "metadata" in data
    assert "markdown_report" in data


def test_api_predict_churn_high_risk(api_client: TestClient):
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1.0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 95.0,
    }
    res = api_client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_tier"] in ["CRITICAL", "HIGH"]
    assert data["churn_probability"] > 0.50
    assert len(data["top_contributing_risk_factors"]) >= 1
    assert "recommended_retention_action" in data


def test_api_predict_churn_low_risk(api_client: TestClient):
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60.0,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.0,
        "TotalCharges": 3900.0,
    }
    res = api_client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_tier"] in ["LOW", "MEDIUM"]
    assert data["churn_probability"] < 0.40


def test_api_monitoring_drift(api_client: TestClient):
    res = api_client.get("/api/monitoring/drift")
    assert res.status_code == 200
    data = res.json()
    assert "system_status" in data
    assert "prediction_drift" in data
