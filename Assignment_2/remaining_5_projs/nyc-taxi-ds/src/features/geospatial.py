"""
Geospatial Feature Engineering Module for NYC Taxi Trips.
Includes Haversine distance, Manhattan distance, bearing, airport detection,
borough classification, and realistic routing waypoints.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union
from src.config import NYC_BOUNDS, AIRPORTS, BOROUGHS


def haversine_distance(
    lat1: Union[float, np.ndarray, pd.Series],
    lon1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lon2: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate the great circle distance between two points on the earth in kilometers.
    Vectorized for numpy arrays and pandas Series.
    """
    R = 6371.0  # Earth radius in kilometers

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    # Clip for numerical stability
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def manhattan_distance(
    lat1: Union[float, np.ndarray, pd.Series],
    lon1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lon2: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate Manhattan (L1) street grid distance in kilometers.
    Decomposes into latitude (north-south) and longitude (east-west) distance.
    """
    # 1 degree latitude ~ 111 km
    lat_dist = np.abs(lat2 - lat1) * 111.0

    # 1 degree longitude ~ 111 * cos(latitude) km
    avg_lat_rad = np.radians((lat1 + lat2) / 2.0)
    lon_dist = np.abs(lon2 - lon1) * 111.0 * np.cos(avg_lat_rad)

    return lat_dist + lon_dist


def bearing(
    lat1: Union[float, np.ndarray, pd.Series],
    lon1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lon2: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate compass bearing from point 1 to point 2 in degrees (0 to 360).
    0 = North, 90 = East, 180 = South, 270 = West.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)

    compass = np.degrees(np.arctan2(y, x))
    return (compass + 360.0) % 360.0


def is_in_nyc_bounds(
    lat: Union[float, np.ndarray, pd.Series],
    lon: Union[float, np.ndarray, pd.Series],
    bounds: Dict[str, float] = NYC_BOUNDS,
) -> Union[bool, np.ndarray, pd.Series]:
    """
    Check if coordinates fall within NYC geographic bounding box.
    """
    return (
        (lat >= bounds["min_lat"])
        & (lat <= bounds["max_lat"])
        & (lon >= bounds["min_lon"])
        & (lon <= bounds["max_lon"])
    )


def is_near_airport(
    lat: Union[float, np.ndarray, pd.Series],
    lon: Union[float, np.ndarray, pd.Series],
    airport_code: str,
) -> Union[bool, np.ndarray, pd.Series]:
    """
    Check whether coordinate is within the airport radius.
    """
    cfg = AIRPORTS.get(airport_code)
    if not cfg:
        raise ValueError(f"Unknown airport code: {airport_code}")

    dist = haversine_distance(lat, lon, cfg["lat"], cfg["lon"])
    return dist <= cfg["radius_km"]


def classify_borough(
    lat: float,
    lon: float,
) -> str:
    """
    Classify a single coordinate into a NYC borough based on proximity and boundaries.
    """
    # Specialized polygon/box approximations for NYC boroughs
    # Manhattan: long narrow island oriented SW-NE
    if 40.700 <= lat <= 40.880 and -74.025 <= lon <= -73.910:
        # Check boundary with Queens / Brooklyn
        if lon < -73.935 or lat > 40.790:
            return "Manhattan"

    # Staten Island
    if lat <= 40.650 and lon <= -74.050:
        return "Staten Island"

    # Bronx
    if lat >= 40.795 and lon >= -73.930:
        return "Bronx"

    # Brooklyn vs Queens
    if lat <= 40.739 and -74.045 <= lon <= -73.855:
        return "Brooklyn"

    if -73.960 <= lon <= -73.700 and 40.540 <= lat <= 40.800:
        return "Queens"

    # Fallback nearest borough centroid
    best_b = "Manhattan"
    min_d = float("inf")
    for b_name, b_coords in BOROUGHS.items():
        d = (lat - b_coords["lat"]) ** 2 + (lon - b_coords["lon"]) ** 2
        if d < min_d:
            min_d = d
            best_b = b_name
    return best_b


def generate_route_geometry(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    num_steps: int = 15,
) -> List[Tuple[float, float]]:
    """
    Generate realistic street-grid route waypoints between pickup and dropoff.
    Simulates Manhattan grid avenue/street travel and bridge/highway links.
    """
    waypoints = [(lat1, lon1)]

    # If points are very close, direct line
    dist_km = float(haversine_distance(lat1, lon1, lat2, lon2))
    if dist_km < 0.1:
        waypoints.append((lat2, lon2))
        return waypoints

    # Simulate street-grid turns (L-shaped or 2-turn staircase)
    # Intermediate step: avenue travel first, then cross street
    mid_lat = lat1 + (lat2 - lat1) * 0.45
    mid_lon = lon1 + (lon2 - lon1) * 0.15

    mid2_lat = lat1 + (lat2 - lat1) * 0.85
    mid2_lon = lon1 + (lon2 - lon1) * 0.65

    pts = [(lat1, lon1), (mid_lat, mid_lon), (mid2_lat, mid2_lon), (lat2, lon2)]

    # Interpolate smoothly along segments
    interpolated = []
    for i in range(len(pts) - 1):
        p_start = pts[i]
        p_end = pts[i + 1]
        steps_seg = max(2, num_steps // (len(pts) - 1))
        for step in range(steps_seg):
            ratio = step / steps_seg
            cur_lat = p_start[0] + ratio * (p_end[0] - p_start[0])
            cur_lon = p_start[1] + ratio * (p_end[1] - p_start[1])
            interpolated.append((round(cur_lat, 6), round(cur_lon, 6)))

    interpolated.append((round(lat2, 6), round(lon2, 6)))
    return interpolated
