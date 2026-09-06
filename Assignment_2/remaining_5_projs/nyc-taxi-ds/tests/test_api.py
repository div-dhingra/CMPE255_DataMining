"""
API Integration Tests for FastAPI backend endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_landmarks_endpoint():
    response = client.get("/api/landmarks")
    assert response.status_code == 200
    data = response.json()
    assert "landmarks" in data
    assert len(data["landmarks"]) >= 5


def test_routes_endpoint():
    response = client.get("/api/routes")
    assert response.status_code == 200
    data = response.json()
    assert "routes" in data
    assert len(data["routes"]) >= 3


def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "gradient_boosting" in data["available_models"]
    assert "comparison" in data


def test_stats_endpoint():
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "temporal" in data


def test_predict_endpoint_success():
    payload = {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7061,
        "dropoff_longitude": -73.9969,
        "passenger_count": 2,
        "model_name": "gradient_boosting",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_duration_seconds" in data
    assert "estimated_fare_usd" in data
    assert "distance_km" in data
    assert "congestion_factor" in data
    assert "route_geometry" in data
    assert len(data["route_geometry"]) > 0


def test_predict_endpoint_out_of_bounds():
    # Out of NYC latitude
    payload = {
        "pickup_latitude": 30.0,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7061,
        "dropoff_longitude": -73.9969,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422  # Validation error from Pydantic bounds


def test_frontend_serving():
    response = client.get("/")
    assert response.status_code == 200
    assert "NYC Taxi" in response.text
