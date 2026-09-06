"""
Data loading and filtering module.
Loads Kaggle NYC Taxi dataset or automatically generates synthetic fallback.
Applies geographic bounding box and outlier sanitization.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from src.config import RAW_DATA_DIR, NYC_BOUNDS
from src.data.generator import generate_synthetic_taxi_data
from src.features.geospatial import is_in_nyc_bounds, haversine_distance

logger = logging.getLogger(__name__)


def get_raw_data_path() -> Path:
    """Return the expected path to the raw train.csv."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR / "train.csv"


def load_dataset(
    file_path: Optional[Path] = None,
    n_synthetic_samples: int = 25000,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Load raw NYC Taxi dataset. If file does not exist, automatically synthesize
    a realistic dataset and cache it to disk.
    """
    path = file_path or get_raw_data_path()

    if not force_synthetic and path.exists():
        logger.info(f"Loading existing taxi dataset from {path}")
        df = pd.read_csv(path)
        # If dataset lacks fare_amount (e.g. pure Kaggle trip_duration competition), synthesize plausible fares
        if "fare_amount" not in df.columns:
            logger.info("Synthesizing fare_amount for Kaggle trip duration dataset...")
            dist = haversine_distance(
                df["pickup_latitude"], df["pickup_longitude"],
                df["dropoff_latitude"], df["dropoff_longitude"]
            )
            df["fare_amount"] = np.round(np.clip(3.0 + dist * 2.15 + (df["trip_duration"] / 60.0) * 0.35, 3.50, 150.0), 2)
    else:
        logger.info(f"Generating {n_synthetic_samples} synthetic NYC taxi trips...")
        df = generate_synthetic_taxi_data(n_samples=n_synthetic_samples)
        df.to_csv(path, index=False)
        logger.info(f"Saved generated dataset to {path}")

    return df


def clean_and_filter_data(
    df: pd.DataFrame,
    bounds: dict = NYC_BOUNDS,
    min_duration: int = 60,       # at least 1 minute
    max_duration: int = 7200,     # at most 2 hours
    min_distance_km: float = 0.05,
    max_distance_km: float = 100.0,
) -> pd.DataFrame:
    """
    Apply CRISP-DM Data Understanding / Cleaning criteria:
    - NYC Geographic bounding box filtering
    - Drop null values and corrupted coordinates (0, 0)
    - Filter unrealistic trip durations (< 1 min or > 2 hours)
    - Filter unrealistic distances (< 50m or > 100 km)
    - Filter passenger counts (1 to 6)
    """
    initial_len = len(df)

    # Convert datetimes if strings
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # Coordinate bounding box filtering
    p_in_nyc = is_in_nyc_bounds(df["pickup_latitude"], df["pickup_longitude"], bounds)
    d_in_nyc = is_in_nyc_bounds(df["dropoff_latitude"], df["dropoff_longitude"], bounds)
    valid_coords = p_in_nyc & d_in_nyc

    # Distance calculation
    dist = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    valid_distance = (dist >= min_distance_km) & (dist <= max_distance_km)

    # Valid duration
    valid_duration = (df["trip_duration"] >= min_duration) & (df["trip_duration"] <= max_duration)

    # Valid passenger count
    valid_passengers = (df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)

    # Combined filter
    mask = valid_coords & valid_distance & valid_duration & valid_passengers
    df_clean = df[mask].copy().reset_index(drop=True)

    logger.info(
        f"Data Cleaning: {initial_len} initial records -> {len(df_clean)} cleaned records "
        f"({((initial_len - len(df_clean)) / initial_len) * 100:.1f}% removed)"
    )

    return df_clean
