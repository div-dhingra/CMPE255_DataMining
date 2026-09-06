"""
Pydantic Schemas for FastAPI NYC Taxi Endpoints.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    pickup_latitude: float = Field(..., ge=40.40, le=41.10, description="Pickup latitude in NYC")
    pickup_longitude: float = Field(..., ge=-74.40, le=-73.50, description="Pickup longitude in NYC")
    dropoff_latitude: float = Field(..., ge=40.40, le=41.10, description="Dropoff latitude in NYC")
    dropoff_longitude: float = Field(..., ge=-74.40, le=-73.50, description="Dropoff longitude in NYC")
    passenger_count: int = Field(default=1, ge=1, le=6, description="Number of passengers")
    pickup_datetime: Optional[str] = Field(default=None, description="ISO or standard datetime string")
    model_name: Optional[str] = Field(default="gradient_boosting", description="Model identifier")


class PredictResponse(BaseModel):
    model_used: str
    predicted_duration_seconds: int
    predicted_duration_minutes: float
    formatted_duration: str
    estimated_fare_usd: float
    distance_km: float
    distance_miles: float
    manhattan_distance_km: float
    congestion_factor: float
    pickup_borough: str
    dropoff_borough: str
    explanation: Dict[str, Any]
    route_geometry: List[List[float]]


class ModelMetricDetail(BaseModel):
    model_name: str
    train_time_sec: float
    cv_rmse_mean: float
    cv_mae_mean: float
    cv_r2_mean: float
    test_rmse: float
    test_rmsle: float
    test_mae: float
    test_r2: float


class ModelsResponse(BaseModel):
    available_models: List[str]
    dataset_info: Dict[str, Any]
    duration_models: Dict[str, ModelMetricDetail]
    fare_models: Dict[str, ModelMetricDetail]
    feature_importance: Dict[str, Any]
