"""
Trip Predictor and Model Explanation Service.
Loads trained models, executes live predictions, and generates feature contribution
breakdowns for duration and fare estimation.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from src.config import MODELS_DIR, FEATURE_COLS, AIRPORTS
from src.features.pipeline import TaxiFeaturePipeline, extract_features
from src.features.geospatial import (
    haversine_distance,
    manhattan_distance,
    bearing,
    classify_borough,
    generate_route_geometry,
)


class TaxiPredictor:
    """
    Inference and Explanation engine for NYC Taxi duration and fare predictions.
    """

    def __init__(self):
        self.preprocessor: Optional[TaxiFeaturePipeline] = None
        self.duration_models: Dict[str, Any] = {}
        self.fare_models: Dict[str, Any] = {}
        self.model_comparison: Dict[str, Any] = {}
        self.feature_importance: Dict[str, Any] = {}
        self.is_loaded = False

    def load_artifacts(self):
        """Load serialized models and metadata."""
        prep_path = MODELS_DIR / "preprocessor.joblib"
        dur_path = MODELS_DIR / "duration_models.joblib"
        fare_path = MODELS_DIR / "fare_models.joblib"

        if not (prep_path.exists() and dur_path.exists() and fare_path.exists()):
            raise FileNotFoundError(
                f"Model artifacts not found in {MODELS_DIR}. Please run training pipeline first."
            )

        self.preprocessor = TaxiFeaturePipeline.load(prep_path)
        self.duration_models = joblib.load(dur_path)
        self.fare_models = joblib.load(fare_path)

        comp_path = MODELS_DIR / "model_comparison.json"
        if comp_path.exists():
            import json
            with open(comp_path, "r") as f:
                self.model_comparison = json.load(f)

        imp_path = MODELS_DIR / "feature_importance.json"
        if imp_path.exists():
            import json
            with open(imp_path, "r") as f:
                self.feature_importance = json.load(f)

        self.is_loaded = True

    def ensure_loaded(self):
        if not self.is_loaded:
            self.load_artifacts()

    def get_available_models(self) -> list:
        self.ensure_loaded()
        return list(self.duration_models.keys())

    def predict_trip(
        self,
        pickup_lat: float,
        pickup_lon: float,
        dropoff_lat: float,
        dropoff_lon: float,
        passenger_count: int = 1,
        pickup_datetime: Optional[str] = None,
        model_name: str = "gradient_boosting",
    ) -> Dict[str, Any]:
        """
        Predict duration and fare for a single trip and provide feature breakdown.
        """
        self.ensure_loaded()

        if model_name not in self.duration_models:
            model_name = "gradient_boosting"

        if not pickup_datetime:
            pickup_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        trip_dict = {
            "pickup_latitude": pickup_lat,
            "pickup_longitude": pickup_lon,
            "dropoff_latitude": dropoff_lat,
            "dropoff_longitude": dropoff_lon,
            "passenger_count": passenger_count,
            "pickup_datetime": pickup_datetime,
        }

        # Feature extraction
        df_trip = pd.DataFrame([trip_dict])
        df_feat = extract_features(df_trip)

        # Scale features
        X = self.preprocessor.scaler.transform(df_feat[FEATURE_COLS])

        # Model predictions
        dur_model = self.duration_models[model_name]
        fare_model = self.fare_models[model_name]

        raw_duration_sec = float(dur_model.predict(X)[0])
        raw_fare_usd = float(fare_model.predict(X)[0])

        # Enforce realistic physical floors
        h_dist_km = float(df_feat["haversine_distance_km"].iloc[0])
        m_dist_km = float(df_feat["manhattan_distance_km"].iloc[0])
        dist_miles = round(h_dist_km * 0.621371, 2)

        # Minimum physical duration (assuming max highway speed 80 km/h)
        min_sec = max(60.0, (h_dist_km / 80.0) * 3600.0 + 30.0)
        duration_sec = int(round(max(raw_duration_sec, min_sec)))

        # Minimum fare: $3.00 initial + $1.50 surcharges
        fare_usd = round(max(raw_fare_usd, 3.50), 2)

        # Minutes and seconds format
        dur_min = duration_sec // 60
        dur_rem_sec = duration_sec % 60
        formatted_duration = f"{dur_min}m {dur_rem_sec}s"

        # Calculate Congestion Factor
        # Average free-flow city speed ~32 km/h; actual trip speed = (h_dist_km / duration_sec) * 3600
        actual_speed_kmh = (h_dist_km / max(duration_sec, 1)) * 3600.0
        free_flow_speed = 32.0 if h_dist_km < 8.0 else 50.0
        congestion_ratio = max(1.0, round(free_flow_speed / max(actual_speed_kmh, 4.0), 2))
        congestion_factor = min(congestion_ratio, 3.2)

        # Borough identification
        p_borough = classify_borough(pickup_lat, pickup_lon)
        d_borough = classify_borough(dropoff_lat, dropoff_lon)

        # Explanation breakdown
        is_rush = bool(df_feat["is_rush_hour"].iloc[0])
        is_jfk = bool(df_feat["is_jfk_trip"].iloc[0])
        is_lga = bool(df_feat["is_lga_trip"].iloc[0])
        is_ewr = bool(df_feat["is_ewr_trip"].iloc[0])
        is_overnight = bool(df_feat["is_overnight"].iloc[0])

        # Detailed fare component breakdown
        base_meter = 3.00
        dist_component = round(m_dist_km * 2.15, 2)
        traffic_component = round(max(0.0, fare_usd - (base_meter + dist_component + (20.0 if is_jfk else 0.0))), 2)
        airport_surcharge = 20.00 if is_jfk else (5.00 if is_lga else (17.50 if is_ewr else 0.0))
        rush_surcharge = 1.00 if is_rush else (0.50 if is_overnight else 0.0)

        # Time component breakdown (duration)
        free_flow_duration_sec = int((m_dist_km / free_flow_speed) * 3600.0)
        delay_sec = max(0, duration_sec - free_flow_duration_sec)

        explanation = {
            "fare_components": {
                "base_meter_fare": base_meter,
                "distance_cost": dist_component,
                "traffic_delay_charge": traffic_component,
                "rush_or_overnight_surcharge": rush_surcharge,
                "airport_tolls_surcharge": airport_surcharge,
            },
            "duration_components": {
                "free_flow_seconds": free_flow_duration_sec,
                "traffic_congestion_delay_seconds": delay_sec,
                "congestion_level": "Heavy Gridlock" if congestion_factor > 2.0 else ("Moderate Traffic" if congestion_factor > 1.3 else "Normal / Free Flow"),
            },
            "features_summary": {
                "haversine_distance_km": round(h_dist_km, 2),
                "manhattan_distance_km": round(m_dist_km, 2),
                "bearing_degrees": round(float(df_feat["bearing_degrees"].iloc[0]), 1),
                "pickup_borough": p_borough,
                "dropoff_borough": d_borough,
                "is_rush_hour": is_rush,
                "is_airport_trip": is_jfk or is_lga or is_ewr,
            }
        }

        # Generate realistic route geometry for map polyline
        route_points = generate_route_geometry(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

        return {
            "model_used": model_name,
            "predicted_duration_seconds": duration_sec,
            "predicted_duration_minutes": round(duration_sec / 60.0, 1),
            "formatted_duration": formatted_duration,
            "estimated_fare_usd": fare_usd,
            "distance_km": round(h_dist_km, 2),
            "distance_miles": dist_miles,
            "manhattan_distance_km": round(m_dist_km, 2),
            "congestion_factor": congestion_factor,
            "pickup_borough": p_borough,
            "dropoff_borough": d_borough,
            "explanation": explanation,
            "route_geometry": route_points,
        }


# Global singleton predictor
_predictor = None

def get_predictor() -> TaxiPredictor:
    global _predictor
    if _predictor is None:
        _predictor = TaxiPredictor()
    return _predictor
