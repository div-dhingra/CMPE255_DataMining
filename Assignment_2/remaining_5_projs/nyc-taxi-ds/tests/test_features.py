"""
Unit tests for geospatial features and calculations.
"""
import pytest
import numpy as np
from src.features.geospatial import (
    haversine_distance,
    manhattan_distance,
    bearing,
    is_in_nyc_bounds,
    is_near_airport,
    classify_borough,
    generate_route_geometry,
)


def test_haversine_distance():
    # Times Square (40.7580, -73.9855) to Grand Central (40.7527, -73.9772) ~0.9 km
    dist = haversine_distance(40.7580, -73.9855, 40.7527, -73.9772)
    assert 0.7 < dist < 1.2

    # Zero distance
    assert haversine_distance(40.7580, -73.9855, 40.7580, -73.9855) == 0.0

    # Vectorized test
    lats1 = np.array([40.7580, 40.7061])
    lons1 = np.array([-73.9855, -73.9969])
    lats2 = np.array([40.7527, 40.7580])
    lons2 = np.array([-73.9772, -73.9855])
    dists = haversine_distance(lats1, lons1, lats2, lons2)
    assert len(dists) == 2
    assert dists[0] > 0
    assert dists[1] > 0


def test_manhattan_distance():
    # Manhattan distance should always be >= Haversine distance
    h_dist = haversine_distance(40.7580, -73.9855, 40.7061, -73.9969)
    m_dist = manhattan_distance(40.7580, -73.9855, 40.7061, -73.9969)
    assert m_dist >= h_dist


def test_bearing():
    # Times Square to North
    b_north = bearing(40.7580, -73.9855, 40.8580, -73.9855)
    assert 350.0 <= b_north or b_north <= 10.0

    # Times Square to East
    b_east = bearing(40.7580, -73.9855, 40.7580, -73.8855)
    assert 80.0 <= b_east <= 100.0


def test_nyc_bounding_box():
    # Times Square is in NYC
    assert is_in_nyc_bounds(40.7580, -73.9855)

    # London is NOT in NYC
    assert not is_in_nyc_bounds(51.5074, -0.1278)


def test_airport_detection():
    # JFK airport center
    assert is_near_airport(40.6413, -73.7781, "JFK")
    # Times Square is not JFK
    assert not is_near_airport(40.7580, -73.9855, "JFK")


def test_borough_classification():
    # Times Square is Manhattan
    assert classify_borough(40.7580, -73.9855) == "Manhattan"
    # JFK is Queens
    assert classify_borough(40.6413, -73.7781) == "Queens"


def test_route_waypoints():
    pts = generate_route_geometry(40.7580, -73.9855, 40.7061, -73.9969, num_steps=10)
    assert len(pts) >= 5
    assert pts[0] == (40.7580, -73.9855)
    assert pts[-1] == (40.7061, -73.9969)
