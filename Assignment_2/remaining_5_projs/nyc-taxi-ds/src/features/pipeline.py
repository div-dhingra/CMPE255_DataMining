"""
Feature Engineering Pipeline.
Transforms raw trip records into rich geospatial and temporal features for ML models.
Handles both batch DataFrame transformation and real-time single-record inference.
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from src.config import FEATURE_COLS, MODELS_DIR
from src.features.geospatial import (
    haversine_distance,
    manhattan_distance,
    bearing,
    is_near_airport,
    classify_borough,
)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract geospatial, temporal, and interaction features from raw trip data.
    Works for both training datasets and single-row inference DataFrames.
    """
    df = df.copy()

    # Parse pickup_datetime
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    p_lat = df["pickup_latitude"].values
    p_lon = df["pickup_longitude"].values
    d_lat = df["dropoff_latitude"].values
    d_lon = df["dropoff_longitude"].values

    # Geospatial Features
    df["haversine_distance_km"] = haversine_distance(p_lat, p_lon, d_lat, d_lon)
    df["manhattan_distance_km"] = manhattan_distance(p_lat, p_lon, d_lat, d_lon)
    df["bearing_degrees"] = bearing(p_lat, p_lon, d_lat, d_lon)

    # Coordinate differences
    df["lat_diff"] = np.abs(d_lat - p_lat)
    df["lon_diff"] = np.abs(d_lon - p_lon)

    # Directness Ratio (grid tortuosity): Manhattan / Haversine
    df["directness_ratio"] = df["manhattan_distance_km"] / (df["haversine_distance_km"] + 1e-5)
    df["directness_ratio"] = np.clip(df["directness_ratio"], 1.0, 3.0)

    # Airport indicators
    p_jfk = is_near_airport(p_lat, p_lon, "JFK")
    d_jfk = is_near_airport(d_lat, d_lon, "JFK")
    df["is_jfk_trip"] = (p_jfk | d_jfk).astype(int)

    p_lga = is_near_airport(p_lat, p_lon, "LGA")
    d_lga = is_near_airport(d_lat, d_lon, "LGA")
    df["is_lga_trip"] = (p_lga | d_lga).astype(int)

    p_ewr = is_near_airport(p_lat, p_lon, "EWR")
    d_ewr = is_near_airport(d_lat, d_lon, "EWR")
    df["is_ewr_trip"] = (p_ewr | d_ewr).astype(int)

    # Temporal features
    dt = df["pickup_datetime"]
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month

    # Rush hour: weekday 7-10 AM or 4-7 PM
    is_weekday = df["pickup_dayofweek"] < 5
    is_morning_rush = (df["pickup_hour"] >= 7) & (df["pickup_hour"] <= 9)
    is_evening_rush = (df["pickup_hour"] >= 16) & (df["pickup_hour"] <= 19)
    df["is_rush_hour"] = (is_weekday & (is_morning_rush | is_evening_rush)).astype(int)

    # Overnight: 11 PM to 5 AM
    df["is_overnight"] = ((df["pickup_hour"] >= 23) | (df["pickup_hour"] <= 5)).astype(int)
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)

    # Ensure passenger count is present
    if "passenger_count" not in df.columns:
        df["passenger_count"] = 1

    return df


class TaxiFeaturePipeline:
    """
    Scikit-learn compatible preprocessor for scaling and formatting features.
    """

    def __init__(self, feature_cols: List[str] = FEATURE_COLS):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        X_feat = extract_features(X)[self.feature_cols]
        self.scaler.fit(X_feat)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming data.")
        X_feat = extract_features(X)[self.feature_cols]
        return self.scaler.transform(X_feat)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def transform_single(self, trip_dict: Dict[str, Any]) -> np.ndarray:
        """Helper to transform a single trip dictionary for live API inference."""
        df_single = pd.DataFrame([trip_dict])
        return self.transform(df_single)

    def save(self, filepath: Optional[str] = None):
        """Save pipeline to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = filepath or (MODELS_DIR / "preprocessor.joblib")
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "TaxiFeaturePipeline":
        """Load pipeline from disk."""
        path = filepath or (MODELS_DIR / "preprocessor.joblib")
        return joblib.load(path)
