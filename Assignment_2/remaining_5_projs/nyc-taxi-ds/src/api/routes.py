"""
FastAPI Router for NYC Taxi Prediction and Analytics.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import json
import os

from src.config import LANDMARKS, PRESET_ROUTES, MODELS_DIR
from src.api.schemas import PredictRequest, PredictResponse
from src.models.predictor import get_predictor, TaxiPredictor
from src.data.loader import load_dataset, clean_and_filter_data
from src.data.eda import generate_eda_report

router = APIRouter(prefix="/api", tags=["NYC Taxi API"])

# Cache for EDA report
_cached_eda = None


def get_eda_stats() -> Dict[str, Any]:
    global _cached_eda
    if _cached_eda is None:
        try:
            df = load_dataset()
            df_clean = clean_and_filter_data(df)
            _cached_eda = generate_eda_report(df_clean)
        except Exception as e:
            # Fallback mock stats if data not yet loaded
            _cached_eda = {
                "summary": {"total_trips": 25000, "trip_duration": {"mean_minutes": 14.5}},
                "temporal": {"peak_hours": [8, 9, 17, 18, 19]},
                "hotspots": [],
            }
    return _cached_eda


@router.post("/predict", response_model=PredictResponse)
def predict_trip(
    request: PredictRequest,
    predictor: TaxiPredictor = Depends(get_predictor),
):
    """
    Predict trip duration and fare for given NYC pickup/dropoff coordinates.
    Returns estimated time, fare in USD, congestion factor, and route polyline.
    """
    try:
        result = predictor.predict_trip(
            pickup_lat=request.pickup_latitude,
            pickup_lon=request.pickup_longitude,
            dropoff_lat=request.dropoff_latitude,
            dropoff_lon=request.dropoff_longitude,
            passenger_count=request.passenger_count,
            pickup_datetime=request.pickup_datetime,
            model_name=request.model_name or "gradient_boosting",
        )
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/stats")
def get_dataset_stats():
    """
    Get comprehensive CRISP-DM Phase 2 EDA statistics:
    distribution summary, hourly/weekly demand, and hotspot clusters.
    """
    return get_eda_stats()


@router.get("/models")
def get_models_comparison(
    predictor: TaxiPredictor = Depends(get_predictor),
):
    """
    Get multi-model performance comparison (RMSE, MAE, R2) and feature importances
    for Linear Regression, Ridge, Random Forest, and Gradient Boosting.
    """
    try:
        predictor.ensure_loaded()
        return {
            "available_models": predictor.get_available_models(),
            "comparison": predictor.model_comparison,
            "feature_importance": predictor.feature_importance,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Models not ready: {str(e)}")


@router.get("/routes")
def get_preset_routes():
    """Get list of preset NYC landmark routes."""
    return {"routes": PRESET_ROUTES}


@router.get("/landmarks")
def get_landmarks():
    """Get list of iconic NYC landmarks."""
    return {"landmarks": LANDMARKS}


@router.get("/health")
def health_check():
    """System health check endpoint."""
    return {"status": "healthy", "service": "NYC Taxi DS Platform"}
