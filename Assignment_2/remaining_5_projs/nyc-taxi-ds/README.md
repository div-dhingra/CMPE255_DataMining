# NYC Taxi DS: End-to-End Machine Learning & Trip Estimator Platform

[![CRISP-DM](https://img.shields.io/badge/Framework-CRISP--DM-orange.svg)](docs/CRISP_DM.md)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Leaflet](https://img.shields.io/badge/Frontend-Leaflet%20Maps-199900.svg)](https://leafletjs.com/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(21%2F21)-brightgreen.svg)](run_tests.sh)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)

An end-to-end Data Science project for the **Kaggle NYC Taxi Challenge** following the **CRISP-DM 6-Phase Lifecycle**, featuring automated data processing, geospatial feature engineering, multi-model evaluation, high-throughput FastAPI REST API, and an interactive Leaflet map frontend.

---

## Highlights

- **CRISP-DM 6-Phase Lifecycle:** Full execution across Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.
- **Geospatial Feature Engineering:** Haversine great-circle distance, Manhattan orthogonal grid distance, compass bearing, borough boundary classification, and airport radial geofencing (JFK, LGA, EWR).
- **Multi-Model Benchmark:** Comparative evaluation of **Linear Regression**, **Ridge ($L_2$)**, **Random Forest**, and **Gradient Boosting (HistGradientBoosting)** with 3-fold cross-validation.
- **Dual Target Prediction:** Predicts both **trip duration** (seconds / minutes) and **fare amount** (USD, incorporating TLC base meter, distance, idle wait time, rush-hour, overnight, and airport surcharges).
- **Interactive Map Frontend:** Leaflet map with draggable pickup/dropoff markers, preset iconic routes, real-time polyline trajectory, congestion estimation, and model explanation breakdown.
- **Self-Contained Execution:** Built-in calibrated synthetic NYC taxi dataset generator matching the official Kaggle schema.

---

## Project Structure

```
nyc-taxi-ds/
├── docs/
│   └── CRISP_DM.md                 # In-depth 6-Phase CRISP-DM documentation
├── data/
│   ├── raw/train.csv               # Raw / synthesized dataset (20,000+ records)
│   └── processed/                  # Processed datasets
├── models/                         # Serialized ML artifacts & evaluation metrics
│   ├── duration_models.joblib      # Fitted duration models
│   ├── fare_models.joblib          # Fitted fare models
│   ├── preprocessor.joblib         # Fitted TaxiFeaturePipeline
│   ├── model_comparison.json       # Benchmark metrics (RMSE, MAE, R²)
│   ├── feature_importance.json     # Feature rankings
│   └── residual_data.json          # Residual error distributions
├── src/
│   ├── config.py                   # NYC bounds, landmarks, airport coordinates
│   ├── data/
│   │   ├── generator.py            # Realistic synthetic NYC taxi generator
│   │   ├── loader.py               # Data loading, cleaning & bounding box filtering
│   │   └── eda.py                  # Summary stats, temporal patterns, hotspot clustering
│   ├── features/
│   │   ├── geospatial.py           # Haversine, Manhattan, bearing, airport detection
│   │   └── pipeline.py             # Feature pipeline & scaling
│   ├── models/
│   │   ├── train.py                # Multi-model training with K-Fold cross-validation
│   │   ├── evaluate.py             # RMSE, RMSLE, MAE, R², residual distributions
│   │   └── predictor.py            # Prediction service & feature contribution breakdown
│   └── api/
│       ├── app.py                  # FastAPI server & static file mount
│       ├── routes.py               # /api/predict, /api/stats, /api/models, /api/routes
│       └── schemas.py              # Pydantic request & response models
├── static/                         # Interactive Frontend
│   ├── index.html                  # Leaflet map UI & analytics dashboards
│   ├── style.css                   # Modern dark UI stylesheet
│   └── app.js                      # Map logic, draggable pins & API integration
├── tests/                          # Comprehensive Pytest Suite (21 tests)
│   ├── test_api.py                 # FastAPI client endpoint tests
│   ├── test_data.py                # Schema & data cleaning tests
│   ├── test_features.py            # Geospatial calculations tests
│   └── test_models.py              # Model inference & evaluation tests
├── scripts/
│   └── run_pipeline.py             # CLI runner for all 6 CRISP-DM phases
├── run_tests.sh                    # Automated test runner script
├── run_server.sh                   # Server startup script (FastAPI + uvicorn)
└── README.md                       # Project overview & documentation
```

---

## Quick Start

### 1. Run All Tests
Execute the automated test suite covering geospatial mathematics, feature pipelines, model predictions, and API endpoints:
```bash
./run_tests.sh
```

### 2. Run the Full CRISP-DM Pipeline
To retrain models and regenerate evaluation metrics from scratch:
```bash
/Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/python scripts/run_pipeline.py
```

### 3. Launch the Server & Interactive Map Frontend
```bash
./run_server.sh
```
Open your browser and navigate to:
- **Interactive Map UI:** [http://localhost:8000/](http://localhost:8000/)
- **Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Model Benchmark Results

Evaluation on 20,000 trips using 3-fold cross-validation and a 20% holdout test set:

### Trip Duration Prediction
| Model Architecture | CV RMSE (s) | Test RMSE (s) | Test RMSLE | Test MAE (s) | Test $R^2$ | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 396.12 | 395.33 | 0.3399 | 284.75 | 0.7462 | 0.05s |
| **Ridge Regression** | 396.10 | 395.32 | 0.3417 | 284.54 | 0.7462 | 0.03s |
| **Random Forest** | 332.40 | 329.52 | 0.2135 | 226.52 | 0.8237 | 1.36s |
| **Gradient Boosting** ⭐ | **325.80** | **322.90** | **0.2094** | **220.20** | **0.8307** | **2.77s** |

### Fare Amount Prediction
| Model Architecture | CV RMSE ($) | Test RMSE ($) | Test MAE ($) | Test $R^2$ | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | $2.75 | $2.73 | $1.95 | 0.9869 | 0.04s |
| **Ridge Regression** | $2.75 | $2.74 | $1.96 | 0.9868 | 0.02s |
| **Random Forest** | $2.58 | $2.55 | $1.73 | 0.9886 | 1.58s |
| **Gradient Boosting** ⭐ | **$2.45** | **$2.42** | **$1.64** | **0.9897** | **3.31s** |

---

## REST API Reference

### Predict Trip (`POST /api/predict`)
Estimates trip duration, fare, distance, and congestion factor for given coordinates.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_latitude": 40.7580,
    "pickup_longitude": -73.9855,
    "dropoff_latitude": 40.7061,
    "dropoff_longitude": -73.9969,
    "passenger_count": 2,
    "model_name": "gradient_boosting"
  }'
```

**Response:**
```json
{
  "model_used": "gradient_boosting",
  "predicted_duration_seconds": 1671,
  "predicted_duration_minutes": 27.9,
  "formatted_duration": "27m 51s",
  "estimated_fare_usd": 27.50,
  "distance_km": 5.85,
  "distance_miles": 3.64,
  "manhattan_distance_km": 6.78,
  "congestion_factor": 2.54,
  "pickup_borough": "Manhattan",
  "dropoff_borough": "Brooklyn",
  "explanation": {
    "fare_components": {
      "base_meter_fare": 3.0,
      "distance_cost": 14.58,
      "traffic_delay_charge": 9.92,
      "rush_or_overnight_surcharge": 0.0,
      "airport_tolls_surcharge": 0.0
    },
    "duration_components": {
      "free_flow_seconds": 762,
      "traffic_congestion_delay_seconds": 909,
      "congestion_level": "Heavy Gridlock"
    },
    "features_summary": {
      "haversine_distance_km": 5.85,
      "manhattan_distance_km": 6.78,
      "bearing_degrees": 172.5,
      "pickup_borough": "Manhattan",
      "dropoff_borough": "Brooklyn",
      "is_rush_hour": false,
      "is_airport_trip": false
    }
  },
  "route_geometry": [
    [40.7580, -73.9855],
    [40.7346, -73.9838],
    [40.7139, -73.9929],
    [40.7061, -73.9969]
  ]
}
```

### Other Endpoints
- `GET /api/stats`: Comprehensive dataset summary, hourly/weekly distributions, and spatial hotspots.
- `GET /api/models`: Model comparison metrics across all 4 architectures and feature importances.
- `GET /api/routes`: NYC iconic landmark preset routes.
- `GET /api/landmarks`: Coordinates of iconic NYC points of interest.
- `GET /api/health`: Service health check.

---

## Interactive Map UI Features

1. **Draggable Markers:** Reposition Green (Pickup) and Red (Dropoff) pins anywhere in NYC. The map instantly redraws the route and updates predictions.
2. **Iconic Preset Routes:** One-click shortcuts for classic NYC trips:
   - JFK Airport → Times Square
   - Wall Street → Grand Central Terminal
   - Times Square → Brooklyn Bridge
   - LaGuardia Airport → Central Park
   - Empire State Building → Yankee Stadium
3. **Model Selection:** Compare live outputs between Gradient Boosting, Random Forest, Ridge, and Linear Regression.
4. **Traffic Scenarios:** Toggle departure time between Morning Rush, Midday, Evening Rush, and Late Night.
5. **Model Explanation Breakdown:**
   - Fare composition stack (Base $3 + Distance + Slow Traffic Delay + Surcharges)
   - Duration decomposition (Free-flow cruise vs. signal congestion delay)
   - Real-time congestion multiplier ($1.0\times$ to $3.2\times$).

---

## License
MIT License. Developed for CMPE 255 Data Mining.
