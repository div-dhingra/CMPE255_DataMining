"""
Exploratory Data Analysis (EDA) Module.
Computes summary statistics, hourly/weekly demand distributions, speed patterns,
and geospatial pickup hotspot clustering for CRISP-DM Phase 2.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.cluster import MiniBatchKMeans
from src.features.geospatial import (
    haversine_distance,
    manhattan_distance,
    classify_borough,
)


def compute_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive descriptive statistics across numerical and categorical features."""
    dist_km = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    speed_kmh = dist_km / (df["trip_duration"] / 3600.0)

    stats = {
        "total_trips": int(len(df)),
        "trip_duration": {
            "mean_seconds": round(float(df["trip_duration"].mean()), 1),
            "median_seconds": round(float(df["trip_duration"].median()), 1),
            "std_seconds": round(float(df["trip_duration"].std()), 1),
            "min_seconds": int(df["trip_duration"].min()),
            "max_seconds": int(df["trip_duration"].max()),
            "mean_minutes": round(float(df["trip_duration"].mean() / 60.0), 2),
        },
        "fare_amount": {
            "mean_usd": round(float(df["fare_amount"].mean()), 2) if "fare_amount" in df.columns else None,
            "median_usd": round(float(df["fare_amount"].median()), 2) if "fare_amount" in df.columns else None,
            "min_usd": round(float(df["fare_amount"].min()), 2) if "fare_amount" in df.columns else None,
            "max_usd": round(float(df["fare_amount"].max()), 2) if "fare_amount" in df.columns else None,
        },
        "distance_km": {
            "mean": round(float(dist_km.mean()), 2),
            "median": round(float(dist_km.median()), 2),
            "std": round(float(dist_km.std()), 2),
            "min": round(float(dist_km.min()), 2),
            "max": round(float(dist_km.max()), 2),
            "mean_miles": round(float(dist_km.mean() * 0.621371), 2),
        },
        "speed_kmh": {
            "mean": round(float(speed_kmh.mean()), 2),
            "median": round(float(speed_kmh.median()), 2),
            "p25": round(float(np.percentile(speed_kmh, 25)), 2),
            "p75": round(float(np.percentile(speed_kmh, 75)), 2),
        },
        "passenger_distribution": {
            int(k): int(v) for k, v in df["passenger_count"].value_counts().sort_index().items()
        },
    }
    return stats


def compute_temporal_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze hourly and weekly demand and congestion patterns."""
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        dt_series = pd.to_datetime(df["pickup_datetime"])
    else:
        dt_series = df["pickup_datetime"]

    hours = dt_series.dt.hour
    dayofweeks = dt_series.dt.dayofweek
    dist_km = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    speed_kmh = dist_km / (df["trip_duration"] / 3600.0)

    # Hourly distribution
    hourly_counts = hours.value_counts().reindex(range(24), fill_value=0).to_dict()
    hourly_duration_avg = df.groupby(hours)["trip_duration"].mean().reindex(range(24), fill_value=0) / 60.0
    hourly_speed_avg = pd.Series(speed_kmh).groupby(hours).mean().reindex(range(24), fill_value=0)

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_counts = dayofweeks.value_counts().reindex(range(7), fill_value=0)
    dow_dict = {day_names[i]: int(dow_counts[i]) for i in range(7)}

    return {
        "hourly_trip_counts": {int(k): int(v) for k, v in hourly_counts.items()},
        "hourly_avg_duration_min": {int(k): round(float(v), 1) for k, v in hourly_duration_avg.items()},
        "hourly_avg_speed_kmh": {int(k): round(float(v), 1) for k, v in hourly_speed_avg.items()},
        "day_of_week_counts": dow_dict,
        "peak_hours": [8, 9, 17, 18, 19],
    }


def compute_hotspots(df: pd.DataFrame, n_clusters: int = 8, random_state: int = 42) -> List[Dict[str, Any]]:
    """Discover high-density pickup zones using spatial clustering."""
    coords = df[["pickup_latitude", "pickup_longitude"]].values
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=1024)
    labels = kmeans.fit_predict(coords)

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    hotspots = []
    for cluster_id, center in enumerate(kmeans.cluster_centers_):
        c_lat, c_lon = center[0], center[1]
        cnt = int(counts[cluster_id])
        pct = round((cnt / total) * 100.0, 1)
        borough = classify_borough(c_lat, c_lon)

        hotspots.append({
            "cluster_id": cluster_id,
            "latitude": round(float(c_lat), 5),
            "longitude": round(float(c_lon), 5),
            "borough": borough,
            "trip_count": cnt,
            "percentage": pct,
        })

    # Sort descending by popularity
    hotspots.sort(key=lambda x: x["trip_count"], reverse=True)
    return hotspots


def generate_eda_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Assemble complete Phase 2 Data Understanding output."""
    return {
        "summary": compute_summary_statistics(df),
        "temporal": compute_temporal_patterns(df),
        "hotspots": compute_hotspots(df, n_clusters=6),
    }
