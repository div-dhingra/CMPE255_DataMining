"""
Configuration settings for NYC Taxi Data Science CRISP-DM project.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
STATIC_DIR = BASE_DIR / "static"

# NYC Geographic Bounding Box
NYC_BOUNDS = {
    "min_lat": 40.50,
    "max_lat": 40.92,
    "min_lon": -74.25,
    "max_lon": -73.70,
}

# Airport Coordinates & Radii (km)
AIRPORTS = {
    "JFK": {"lat": 40.6413, "lon": -73.7781, "radius_km": 3.2, "name": "John F. Kennedy Intl Airport"},
    "LGA": {"lat": 40.7769, "lon": -73.8740, "radius_km": 2.5, "name": "LaGuardia Airport"},
    "EWR": {"lat": 40.6895, "lon": -74.1745, "radius_km": 3.0, "name": "Newark Liberty Intl Airport"},
}

# NYC Borough Centroids for coarse classification
BOROUGHS = {
    "Manhattan": {"lat": 40.7831, "lon": -73.9712},
    "Brooklyn": {"lat": 40.6782, "lon": -73.9442},
    "Queens": {"lat": 40.7282, "lon": -73.7949},
    "Bronx": {"lat": 40.8448, "lon": -73.8648},
    "Staten Island": {"lat": 40.5795, "lon": -74.1502},
}

# NYC Iconic Landmark Presets for Map Frontend
LANDMARKS = [
    {"id": "times_square", "name": "Times Square", "lat": 40.7580, "lon": -73.9855, "borough": "Manhattan"},
    {"id": "grand_central", "name": "Grand Central Terminal", "lat": 40.7527, "lon": -73.9772, "borough": "Manhattan"},
    {"id": "jfk_airport", "name": "JFK International Airport", "lat": 40.6413, "lon": -73.7781, "borough": "Queens"},
    {"id": "lga_airport", "name": "LaGuardia Airport", "lat": 40.7769, "lon": -73.8740, "borough": "Queens"},
    {"id": "brooklyn_bridge", "name": "Brooklyn Bridge", "lat": 40.7061, "lon": -73.9969, "borough": "Brooklyn"},
    {"id": "wall_street", "name": "Wall Street / FiDi", "lat": 40.7069, "lon": -74.0090, "borough": "Manhattan"},
    {"id": "central_park", "name": "Central Park South", "lat": 40.7661, "lon": -73.9772, "borough": "Manhattan"},
    {"id": "empire_state", "name": "Empire State Building", "lat": 40.7484, "lon": -73.9857, "borough": "Manhattan"},
    {"id": "world_trade_center", "name": "One World Trade Center", "lat": 40.7127, "lon": -74.0134, "borough": "Manhattan"},
    {"id": "yankee_stadium", "name": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262, "borough": "Bronx"},
]

# Preset Iconic Routes
PRESET_ROUTES = [
    {
        "id": "jfk_to_midtown",
        "name": "JFK Airport → Times Square",
        "description": "Iconic airport transfer to Midtown Manhattan (Highway + City streets)",
        "pickup": {"name": "JFK International Airport", "lat": 40.6413, "lon": -73.7781},
        "dropoff": {"name": "Times Square", "lat": 40.7580, "lon": -73.9855},
        "passenger_count": 2,
    },
    {
        "id": "wall_st_to_grand_central",
        "name": "Wall Street → Grand Central",
        "description": "Manhattan Commuter Corridor during rush hour",
        "pickup": {"name": "Wall Street / FiDi", "lat": 40.7069, "lon": -74.0090},
        "dropoff": {"name": "Grand Central Terminal", "lat": 40.7527, "lon": -73.9772},
        "passenger_count": 1,
    },
    {
        "id": "times_square_to_brooklyn",
        "name": "Times Square → Brooklyn Bridge",
        "description": "Midtown to Brooklyn crossing East River",
        "pickup": {"name": "Times Square", "lat": 40.7580, "lon": -73.9855},
        "dropoff": {"name": "Brooklyn Bridge", "lat": 40.7061, "lon": -73.9969},
        "passenger_count": 2,
    },
    {
        "id": "lga_to_central_park",
        "name": "LaGuardia Airport → Central Park",
        "description": "Queens to Upper East / Central Park",
        "pickup": {"name": "LaGuardia Airport", "lat": 40.7769, "lon": -73.8740},
        "dropoff": {"name": "Central Park South", "lat": 40.7661, "lon": -73.9772},
        "passenger_count": 3,
    },
    {
        "id": "midtown_to_yankee_stadium",
        "name": "Empire State Building → Yankee Stadium",
        "description": "Midtown Manhattan to The Bronx game day trip",
        "pickup": {"name": "Empire State Building", "lat": 40.7484, "lon": -73.9857},
        "dropoff": {"name": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262},
        "passenger_count": 4,
    },
]

# Feature columns used for modeling
FEATURE_COLS = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "haversine_distance_km",
    "manhattan_distance_km",
    "bearing_degrees",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "is_overnight",
    "is_jfk_trip",
    "is_lga_trip",
    "is_ewr_trip",
    "directness_ratio",
    "lat_diff",
    "lon_diff",
]

TARGET_COLS = {
    "duration": "trip_duration",  # in seconds
    "fare": "fare_amount",        # in USD
}
