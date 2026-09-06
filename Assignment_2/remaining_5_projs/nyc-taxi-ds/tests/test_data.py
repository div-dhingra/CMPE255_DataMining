"""
Unit tests for data generation, schema validation, cleaning, and EDA.
"""
import pytest
import pandas as pd
from src.data.generator import generate_synthetic_taxi_data
from src.data.loader import clean_and_filter_data
from src.data.eda import compute_summary_statistics, compute_temporal_patterns, compute_hotspots


def test_synthetic_data_generator_schema():
    df = generate_synthetic_taxi_data(n_samples=100, random_seed=123)
    expected_cols = [
        "id", "vendor_id", "pickup_datetime", "dropoff_datetime",
        "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "store_and_fwd_flag",
        "trip_duration", "fare_amount",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    assert len(df) == 100
    assert (df["trip_duration"] >= 60).all()
    assert (df["fare_amount"] >= 3.50).all()
    assert (df["passenger_count"] >= 1).all()


def test_data_cleaning():
    df = generate_synthetic_taxi_data(n_samples=50, random_seed=42)
    # Inject bad row
    bad_row = df.iloc[0].copy()
    bad_row["pickup_latitude"] = 0.0  # Off coast of Africa
    bad_df = pd.concat([df, pd.DataFrame([bad_row])], ignore_index=True)

    cleaned = clean_and_filter_data(bad_df)
    assert len(cleaned) == len(df)  # The invalid coordinate was dropped


def test_eda_computations():
    df = generate_synthetic_taxi_data(n_samples=200, random_seed=42)
    stats = compute_summary_statistics(df)
    assert stats["total_trips"] == 200
    assert "trip_duration" in stats
    assert "fare_amount" in stats
    assert stats["distance_km"]["mean"] > 0

    temporal = compute_temporal_patterns(df)
    assert len(temporal["hourly_trip_counts"]) == 24
    assert len(temporal["day_of_week_counts"]) == 7

    hotspots = compute_hotspots(df, n_clusters=4)
    assert len(hotspots) == 4
    assert sum(h["trip_count"] for h in hotspots) == 200
