"""
Synthetic NYC Taxi Trip Generator.
Generates realistic, physically plausible NYC taxi trips matching Kaggle schema
and real NYC yellow cab traffic and fare dynamics.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from src.config import NYC_BOUNDS, AIRPORTS, LANDMARKS
from src.features.geospatial import haversine_distance, manhattan_distance


def generate_synthetic_taxi_data(
    n_samples: int = 15000,
    random_seed: int = 42,
    start_date: str = "2016-01-01",
    end_date: str = "2016-06-30",
) -> pd.DataFrame:
    """
    Generate synthetic NYC Taxi trips following the Kaggle dataset schema:
    id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count,
    pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
    store_and_fwd_flag, trip_duration (seconds), fare_amount (USD).
    """
    rng = np.random.default_rng(random_seed)

    # Common pickup/dropoff centers (Midtown, Lower Manhattan, Airports, Brooklyn)
    hotspots = [
        {"name": "Midtown Manhattan", "lat": 40.7580, "lon": -73.9855, "weight": 0.40, "spread": 0.02},
        {"name": "Lower Manhattan / FiDi", "lat": 40.7127, "lon": -74.0090, "weight": 0.20, "spread": 0.018},
        {"name": "Upper East / West Side", "lat": 40.7780, "lon": -73.9650, "weight": 0.15, "spread": 0.022},
        {"name": "JFK Airport", "lat": 40.6413, "lon": -73.7781, "weight": 0.08, "spread": 0.008},
        {"name": "LaGuardia Airport", "lat": 40.7769, "lon": -73.8740, "weight": 0.07, "spread": 0.007},
        {"name": "Brooklyn (Downtown/Williamsburg)", "lat": 40.6920, "lon": -73.9870, "weight": 0.08, "spread": 0.025},
        {"name": "Bronx South", "lat": 40.8250, "lon": -73.9250, "weight": 0.02, "spread": 0.02},
    ]

    weights = [h["weight"] for h in hotspots]

    # Sample cluster index for pickups and dropoffs
    pickup_clusters = rng.choice(len(hotspots), size=n_samples, p=weights)
    # Dropoffs have a slight bias towards another cluster or within the same cluster
    dropoff_clusters = rng.choice(len(hotspots), size=n_samples, p=weights)

    # Generate coordinates around cluster centers
    p_lat = np.array([hotspots[i]["lat"] for i in pickup_clusters]) + rng.normal(0, [hotspots[i]["spread"] for i in pickup_clusters])
    p_lon = np.array([hotspots[i]["lon"] for i in pickup_clusters]) + rng.normal(0, [hotspots[i]["spread"] for i in pickup_clusters])

    d_lat = np.array([hotspots[i]["lat"] for i in dropoff_clusters]) + rng.normal(0, [hotspots[i]["spread"] for i in dropoff_clusters])
    d_lon = np.array([hotspots[i]["lon"] for i in dropoff_clusters]) + rng.normal(0, [hotspots[i]["spread"] for i in dropoff_clusters])

    # Clip to bounding box
    p_lat = np.clip(p_lat, NYC_BOUNDS["min_lat"] + 0.01, NYC_BOUNDS["max_lat"] - 0.01)
    p_lon = np.clip(p_lon, NYC_BOUNDS["min_lon"] + 0.01, NYC_BOUNDS["max_lon"] - 0.01)
    d_lat = np.clip(d_lat, NYC_BOUNDS["min_lat"] + 0.01, NYC_BOUNDS["max_lat"] - 0.01)
    d_lon = np.clip(d_lon, NYC_BOUNDS["min_lon"] + 0.01, NYC_BOUNDS["max_lon"] - 0.01)

    # Replace dropoffs that are identical with small random shift
    min_dist_threshold = 0.002
    identical_mask = (np.abs(p_lat - d_lat) < min_dist_threshold) & (np.abs(p_lon - d_lon) < min_dist_threshold)
    d_lat[identical_mask] += rng.uniform(0.005, 0.02, size=identical_mask.sum())
    d_lon[identical_mask] += rng.uniform(0.005, 0.02, size=identical_mask.sum())

    # Random timestamps across 6 months
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    random_timestamps = rng.integers(start_ts, end_ts, size=n_samples)
    pickup_datetimes = pd.to_datetime(random_timestamps, unit="s")

    # Temporal feature extraction
    hours = pickup_datetimes.hour.values
    dayofweeks = pickup_datetimes.dayofweek.values

    # Passenger count: mostly 1 (70%), 2 (15%), 3 (4%), 4 (2%), 5 (6%), 6 (3%)
    passenger_dist = [0.70, 0.15, 0.04, 0.02, 0.06, 0.03]
    passenger_counts = rng.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=passenger_dist)

    # Vendors (1: Creative Mobile Technologies, 2: VeriFone Inc.)
    vendors = rng.choice([1, 2], size=n_samples, p=[0.46, 0.54])
    store_and_fwd = rng.choice(["N", "Y"], size=n_samples, p=[0.99, 0.01])

    # Distance calculation
    dist_km = haversine_distance(p_lat, p_lon, d_lat, d_lon)
    dist_km = np.maximum(dist_km, 0.1)  # at least 100 meters

    # Calculate Realistic Speed based on hour, day, and distance:
    # Base speeds: 15 km/h in dense Manhattan core during rush hours (8-10 AM, 4-7 PM)
    # 22 km/h off-peak daytime, 32 km/h overnight (11 PM - 5 AM)
    # Long trips (> 12 km, e.g., JFK) include highways where speed increases to ~45-60 km/h
    is_weekday = dayofweeks < 5
    is_morning_rush = is_weekday & (hours >= 7) & (hours <= 10)
    is_evening_rush = is_weekday & (hours >= 16) & (hours <= 19)
    is_rush = is_morning_rush | is_evening_rush
    is_overnight = (hours >= 23) | (hours <= 5)

    base_speed = np.full(n_samples, 20.0)  # default km/h
    base_speed[is_rush] = 13.5
    base_speed[is_overnight] = 28.0

    # Highway acceleration for trips > 10 km
    highway_factor = np.where(dist_km > 8.0, 1.4 + 0.03 * np.minimum(dist_km - 8.0, 20.0), 1.0)
    effective_speed = base_speed * highway_factor

    # Add speed variance (log-normal multiplier)
    speed_noise = rng.lognormal(mean=0.0, sigma=0.22, size=n_samples)
    effective_speed = np.clip(effective_speed * speed_noise, 6.0, 75.0)

    # Trip Duration = (distance / speed in km/h) * 3600 seconds + red light / pickup queue
    traffic_delay_sec = rng.uniform(40, 180, size=n_samples)
    trip_duration_sec = (dist_km / effective_speed) * 3600.0 + traffic_delay_sec
    trip_duration_sec = np.round(np.clip(trip_duration_sec, 60, 7200)).astype(int)

    # Dropoff datetimes
    dropoff_datetimes = [p_dt + timedelta(seconds=int(d_sec)) for p_dt, d_sec in zip(pickup_datetimes, trip_duration_sec)]

    # Calculate Fare Amount based on Official NYC Yellow Cab Rules:
    # - Base entry fare: $3.00
    # - Mileage charge: $0.70 per 1/5 mile = $2.17 per km ($3.50 per mile)
    # - Delay / slow traffic charge: $0.50 per 60 sec below 12mph
    # - Peak hour surcharge: $1.00 (4-8 PM weekdays)
    # - Overnight surcharge: $0.50 (8 PM - 6 AM)
    # - MTA surcharge: $0.50
    # - Improvement surcharge: $1.00
    # - JFK Flat Rate / Toll factor for airport routes: ~$15 - $25 extra tolls
    is_jfk_p = haversine_distance(p_lat, p_lon, AIRPORTS["JFK"]["lat"], AIRPORTS["JFK"]["lon"]) <= AIRPORTS["JFK"]["radius_km"]
    is_jfk_d = haversine_distance(d_lat, d_lon, AIRPORTS["JFK"]["lat"], AIRPORTS["JFK"]["lon"]) <= AIRPORTS["JFK"]["radius_km"]
    is_jfk = is_jfk_p | is_jfk_d

    fare = (
        3.00
        + dist_km * 2.15
        + (trip_duration_sec / 60.0) * 0.35
        + np.where(is_rush, 1.00, 0.0)
        + np.where((hours >= 20) | (hours <= 6), 0.50, 0.0)
        + 1.50  # MTA + improvement surcharges
        + np.where(is_jfk, rng.uniform(15.0, 25.0, size=n_samples), 0.0)
    )
    # Random tip & small meter noise
    fare += rng.normal(0, 0.75, size=n_samples)
    fare = np.round(np.clip(fare, 3.50, 150.0), 2)

    # Unique trip IDs
    trip_ids = [f"id{1000000 + i}" for i in range(n_samples)]

    df = pd.DataFrame({
        "id": trip_ids,
        "vendor_id": vendors,
        "pickup_datetime": pickup_datetimes.strftime("%Y-%m-%d %H:%M:%S"),
        "dropoff_datetime": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in dropoff_datetimes],
        "passenger_count": passenger_counts,
        "pickup_longitude": np.round(p_lon, 6),
        "pickup_latitude": np.round(p_lat, 6),
        "dropoff_longitude": np.round(d_lon, 6),
        "dropoff_latitude": np.round(d_lat, 6),
        "store_and_fwd_flag": store_and_fwd,
        "trip_duration": trip_duration_sec,
        "fare_amount": fare,
    })

    return df
