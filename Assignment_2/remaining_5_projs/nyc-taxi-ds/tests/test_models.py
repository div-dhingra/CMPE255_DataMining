"""
Unit tests for model prediction, evaluation metrics, and explanation breakdown.
"""
import pytest
import numpy as np
from src.models.evaluate import compute_metrics, compute_residuals
from src.models.predictor import get_predictor


def test_evaluation_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["rmse"] == 10.0
    assert metrics["mae"] == 10.0
    assert metrics["r2"] > 0.95
    assert metrics["rmsle"] > 0


def test_residuals_computation():
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred = np.array([105.0, 195.0, 305.0, 390.0, 510.0])
    res = compute_residuals(y_true, y_pred)
    assert "residual_mean" in res
    assert "histogram" in res
    assert len(res["sample_points"]) == 5


def test_predictor_inference():
    predictor = get_predictor()
    predictor.ensure_loaded()

    # Times Square to Brooklyn Bridge
    result = predictor.predict_trip(
        pickup_lat=40.7580,
        pickup_lon=-73.9855,
        dropoff_lat=40.7061,
        dropoff_lon=-73.9969,
        passenger_count=2,
        model_name="gradient_boosting",
    )

    assert result["predicted_duration_seconds"] > 180
    assert result["estimated_fare_usd"] >= 5.0
    assert result["distance_miles"] > 1.0
    assert result["congestion_factor"] >= 1.0
    assert "explanation" in result
    assert "fare_components" in result["explanation"]
    assert len(result["route_geometry"]) > 5
