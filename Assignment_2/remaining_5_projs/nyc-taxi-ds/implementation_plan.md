# NYC Taxi DS - Implementation Plan (Retroactive)

## 1. Goal Description
This document outlines the retroactive implementation plan for the **NYC Taxi Challenge** project. The project follows the **CRISP-DM 6-Phase Lifecycle** to build an end-to-end Machine Learning pipeline and an interactive FastAPI + Leaflet web application for predicting NYC taxi trip durations and fares.

## 2. Technical Stack
- **Backend / API**: FastAPI (Python 3.11), Uvicorn
- **Machine Learning**: Scikit-learn (Linear Regression, Ridge, Random Forest, HistGradientBoosting)
- **Data Manipulation**: Pandas, NumPy
- **Geospatial Processing**: Custom Haversine/Manhattan distance and bearing calculations
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Leaflet.js
- **Testing**: Pytest

## 3. Architecture & Key Components

### 3.1 Data Pipeline & Feature Engineering (`src/data/`, `src/features/`)
- **`src/data/generator.py` & `loader.py`**: Generates and loads synthetic NYC taxi data matching the Kaggle schema, applying data cleaning and bounding box filters.
- **`src/features/geospatial.py`**: Computes geospatial features like Haversine distance, Manhattan distance, bearing, and airport geofencing (JFK, LGA, EWR).
- **`src/features/pipeline.py`**: A `TaxiFeaturePipeline` that orchestrates data transformation, standard scaling, and feature extraction.

### 3.2 Machine Learning Models (`src/models/`)
- **`src/models/train.py` & `evaluate.py`**: Implements training and 3-fold cross-validation for multiple model architectures (Linear, Ridge, RF, Gradient Boosting). Evaluates using RMSE, MAE, R², and RMSLE.
- **`src/models/predictor.py`**: Contains the `TaxiPredictor` class. This is the inference engine that loads `joblib` artifacts, processes raw coordinate inputs, enforces realistic physical constraints (e.g., minimum duration and fare), and generates detailed prediction explanations (congestion factors, fare breakdowns).

### 3.3 REST API (`src/api/`)
- **`src/api/app.py`**: Initializes the FastAPI application, mounts static frontend files, and handles application lifecycle events (loading models on startup).
- **`src/api/routes.py`**: Exposes core endpoints:
  - `POST /api/predict`: Handles inference requests.
  - `GET /api/models`: Returns model comparison metrics.
  - `GET /api/stats`: Returns dataset statistics (EDA).
  - `GET /api/routes` & `GET /api/landmarks`: Returns preset locations.

### 3.4 Interactive Frontend (`static/`)
- **`static/index.html` & `style.css`**: The dark-themed, tabbed UI layout containing the Leaflet map, dashboard metrics, and model evaluation tables.
- **`static/app.js`**: Manages application state, handles Leaflet map initialization, dragging events for pickup/dropoff markers, and dynamically renders the route polyline and prediction breakdowns via API calls.

## 4. Testing & Verification
The project includes a comprehensive Pytest suite (`tests/`) containing 21 passing tests:
- **`test_api.py`**: Validates FastAPI endpoints (`/api/predict`, `/api/health`, `/api/models`).
- **`test_data.py`**: Ensures correct data loading and schema validation.
- **`test_features.py`**: Tests geospatial calculations (e.g., Haversine accuracy, borough classification).
- **`test_models.py`**: Tests the `TaxiPredictor` inference engine and model evaluation functions.

All tests can be executed via the provided `run_tests.sh` script.
