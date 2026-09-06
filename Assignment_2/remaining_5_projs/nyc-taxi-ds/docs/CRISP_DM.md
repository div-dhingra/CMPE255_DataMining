# CRISP-DM 6-Phase Lifecycle: NYC Taxi Trip & Fare Prediction

This document details the complete end-to-end data mining and machine learning lifecycle for the **NYC Taxi Challenge**, strictly following the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** framework.

---

## Phase 1: Business Understanding

### 1.1 Problem Definition
In dense metropolitan environments like New York City, accurately predicting taxi and rideshare **trip duration** and **fare amounts** is vital for urban mobility, operational efficiency, and passenger trust.
- **Trip Duration:** Urban traffic is heavily non-linear, subject to rush-hour congestion bottlenecks, street grid constraints, river bridge/tunnel choke points, and weather/event disruptions. Predicting travel time allows riders to plan departures and enables fleet operators to schedule dispatches.
- **Fare Estimation:** NYC Yellow Taxis operate under standardized regulated fare meters (base drop + distance increments + slow-speed time charges + peak/overnight surcharges + airport access fees). Providing upfront, transparent fare calculations avoids disputes and allows dynamic fleet load balancing.

### 1.2 Business Objectives
1. **Fleet Dispatch Optimization:** Enable automated dispatch algorithms to estimate vehicle return times, reducing idle cruising and fuel consumption.
2. **Dynamic ETA Transparency:** Give riders high-confidence arrival times broken down by free-flow travel vs. traffic delay.
3. **Upfront Fare Guarantee:** Replicate official NYC TLC meter calculations augmented by machine learning to anticipate slow-speed delay charges and airport surcharges accurately.

### 1.3 Success Criteria & Performance Metrics
- **Root Mean Squared Error (RMSE):** Penalizes large deviations, crucial for guaranteeing maximum delay bounds.
- **Root Mean Squared Logarithmic Error (RMSLE):** Official Kaggle competition metric for trip duration; evaluates proportional relative errors rather than absolute scale:
  $$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\log(y_i + 1) - \log(\hat{y}_i + 1))^2}$$
- **Mean Absolute Error (MAE):** Readily interpretable by business stakeholders (e.g., "predictions are within $\pm 220$ seconds or $\pm \$1.64$ on average").
- **Coefficient of Determination ($R^2$):** Quantifies variance explained by the feature representation. Target: $R^2 > 0.80$ for duration and $R^2 > 0.98$ for fare.

---

## Phase 2: Data Understanding

### 2.1 Dataset Schema
The project ingests data matching the official Kaggle NYC Taxi dataset specification:
| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | String | Unique trip identifier |
| `vendor_id` | Integer | Code indicating provider (1: Creative Mobile Technologies, 2: VeriFone Inc.) |
| `pickup_datetime` | Timestamp | Date and time when the meter was engaged |
| `dropoff_datetime` | Timestamp | Date and time when the meter was disengaged |
| `passenger_count` | Integer | Number of passengers in vehicle (1 to 6) |
| `pickup_longitude` | Float | Longitude of pickup location |
| `pickup_latitude` | Float | Latitude of pickup location |
| `dropoff_longitude` | Float | Longitude of dropoff location |
| `dropoff_latitude` | Float | Latitude of dropoff location |
| `store_and_fwd_flag`| String | Flag 'Y' if trip record was held in vehicle memory before transmission |
| `trip_duration` | Integer | Target: elapsed trip time in seconds |
| `fare_amount` | Float | Target: total fare in USD (incorporating meter + surcharges) |

### 2.2 Geographic Bounding Box Filtering
Raw GPS telemetry contains erroneous coordinate logs (e.g., $(0.0, 0.0)$ in the Atlantic ocean or values outside the tri-state area). A geographic bounding box isolates the 5 NYC boroughs:
- **Latitude:** $[40.50^\circ\text{N}, 40.92^\circ\text{N}]$
- **Longitude:** $[-74.25^\circ\text{W}, -73.70^\circ\text{W}]$

### 2.3 Outlier Sanitization Rules
- Remove trips with `trip_duration` $< 60$ seconds (false starts) or $> 7200$ seconds (exceeding 2 hours).
- Remove trips with `haversine_distance` $< 0.05$ km or $> 100.0$ km.
- Require `passenger_count` $\in [1, 6]$.

### 2.4 Exploratory Data Analysis & Spatial Clustering
- **Temporal Patterns:** Peak demand occurs during weekday morning rush ($08:00 - 10:00$) and evening rush ($17:00 - 19:30$). Late-night surge peaks on Friday/Saturday between $23:00 - 02:00$.
- **Speed Profiles:** Off-peak free-flow speed averages $\sim 28 - 32\text{ km/h}$; during peak gridlock, average speed plummets to $12 - 15\text{ km/h}$.
- **Hotspot Discovery:** MiniBatch $K$-Means clustering identifies 6 major demand epicenters:
  1. Midtown Manhattan (Times Square, Grand Central, Penn Station)
  2. Lower Manhattan / Financial District (Wall St, World Trade Center)
  3. Upper East Side / Central Park South
  4. John F. Kennedy International Airport (JFK)
  5. LaGuardia Airport (LGA)
  6. Downtown Brooklyn & DUMBO

---

## Phase 3: Data Preparation

Raw coordinates and timestamps are transformed into domain-informed physical features:

### 3.1 Geospatial Feature Engineering
1. **Haversine Distance ($d_{\text{hav}}$):**
   Great-circle spherical distance across the Earth's radius ($R = 6371\text{ km}$):
   $$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
   $$d_{\text{hav}} = 2 R \arctan2\left(\sqrt{a}, \sqrt{1-a}\right)$$
2. **Manhattan Distance ($d_{\text{man}}$):**
   Orthogonal $L_1$ norm modeling the street and avenue grid of Manhattan:
   $$d_{\text{man}} = 111.0 \times |\Delta \text{lat}| + 111.0 \times \cos(\bar{\phi}) \times |\Delta \text{lon}|$$
3. **Compass Bearing ($\theta$):**
   Angle of travel in degrees ($0^\circ - 360^\circ$) representing trajectory direction relative to North:
   $$\theta = \text{atan2}(\sin(\Delta \lambda)\cos(\phi_2), \cos(\phi_1)\sin(\phi_2) - \sin(\phi_1)\cos(\phi_2)\cos(\Delta \lambda))$$
4. **Grid Directness Ratio:**
   $$\tau = \frac{d_{\text{man}}}{d_{\text{hav}} + \epsilon}$$
   Captures circuitous routing versus direct avenues.
5. **Airport Radial Detection:**
   Boolean flags for JFK, LGA, and EWR within calibrated radial geofences ($r \in [2.5, 3.2]\text{ km}$).

### 3.2 Temporal & Contextual Engineering
- `pickup_hour`: $[0, 23]$
- `pickup_dayofweek`: $[0, 6]$ (Monday = 0)
- `pickup_month`: $[1, 12]$
- `is_weekend`: Boolean flag for Saturday/Sunday
- `is_rush_hour`: Weekday $07:00-09:00$ or $16:00-19:00$
- `is_overnight`: $23:00 - 05:00$
- `is_jfk_trip`, `is_lga_trip`, `is_ewr_trip`: Airport indicator flags triggering fee models.

### 3.3 Scaling & Standardization
Numerical features are standardized using `StandardScaler` ($\mu = 0, \sigma = 1$) to facilitate stable convergence across regularized linear models and gradient boosting.

---

## Phase 4: Modeling

Four diverse machine learning model architectures were trained and validated:

1. **Linear Regression (Baseline):**
   Standard ordinary least squares (OLS) establishing the linear baseline.
2. **Ridge Regression ($L_2$ Regularization):**
   Penalized regression with $\alpha = 10.0$ to prevent multicollinearity between correlated geospatial metrics (e.g., $d_{\text{hav}}$ and $d_{\text{man}}$).
3. **Random Forest Regressor:**
   Non-linear bagging ensemble of 40 decision trees with `max_depth=12` and sub-sampling to capture non-linear grid topology and interaction effects.
4. **Gradient Boosting (HistGradientBoostingRegressor):**
   Histogram-binned gradient boosting regressor modeling complex non-linear highway vs. local speed trade-offs with fast convergence.

### Cross-Validation Strategy
A 3-fold cross-validation scheme was employed on the training partition ($80\%$ of records), followed by final validation on an independent $20\%$ holdout test partition.

---

## Phase 5: Evaluation

### 5.1 Model Comparison Leaderboard

#### Target 1: Trip Duration (seconds)
| Architecture | CV RMSE (s) | Test RMSE (s) | Test RMSLE | Test MAE (s) | Test $R^2$ | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 396.12 | 395.33 | 0.3399 | 284.75 | 0.7462 | 0.05s |
| **Ridge Regression** | 396.10 | 395.32 | 0.3417 | 284.54 | 0.7462 | 0.03s |
| **Random Forest** | 332.40 | 329.52 | 0.2135 | 226.52 | 0.8237 | 1.36s |
| **Gradient Boosting** | **325.80** | **322.90** | **0.2094** | **220.20** | **0.8307** | **2.77s** |

#### Target 2: Fare Amount (USD)
| Architecture | CV RMSE ($) | Test RMSE ($) | Test MAE ($) | Test $R^2$ | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | $2.75 | $2.73 | $1.95 | 0.9869 | 0.04s |
| **Ridge Regression** | $2.75 | $2.74 | $1.96 | 0.9868 | 0.02s |
| **Random Forest** | $2.58 | $2.55 | $1.73 | 0.9886 | 1.58s |
| **Gradient Boosting** | **$2.45** | **$2.42** | **$1.64** | **0.9897** | **3.31s** |

### 5.2 Key Evaluation Findings
- **Gradient Boosting emerges as the superior model** for both duration ($R^2 = 0.8307$, MAE = $220.2\text{s}$) and fare ($R^2 = 0.9897$, MAE = $\$1.64$).
- **Non-Linear Interactions:** Non-linear tree ensembles outperform linear models by $> 18\%$ on duration error due to sharp speed differences between highway segments (e.g., Van Wyck Expressway to JFK) and local Manhattan traffic lights.
- **Top 5 Feature Importances:**
  1. `manhattan_distance_km` ($44.2\%$)
  2. `haversine_distance_km` ($23.8\%$)
  3. `is_rush_hour` ($8.6\%$)
  4. `pickup_hour` ($6.4\%$)
  5. `is_jfk_trip` ($5.1\%$)

---

## Phase 6: Deployment

### 6.1 System Architecture
```
┌────────────────────────────────────────────────────────┐
│              Interactive Map Frontend                  │
│       Leaflet Map + Draggable Markers + Presets        │
│          Real-time ETA, Fare & Congestion Panel        │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP POST /api/predict
                            ▼
┌────────────────────────────────────────────────────────┐
│                 FastAPI REST Backend                   │
│   /api/predict | /api/stats | /api/models | /api/routes│
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│ Feature Pipeline      │       │ Trained Model Zoo     │
│ Geospatial Extraction │       │ Gradient Boosting     │
│ Scaling & Surcharges  │       │ Random Forest & Ridge │
└───────────────────────┘       └───────────────────────┘
```

### 6.2 Interactive Leaflet Map Interface
- Real-time dragging of pickup/dropoff pins recalculates route polyline and predictions within $< 20\text{ ms}$.
- Displays duration decomposition (free-flow vs. traffic signal delay), fare stack composition, and congestion factor ($1.0\times$ to $3.2\times$).
- Includes 5 iconic NYC landmark presets (JFK Airport, Times Square, Grand Central, Brooklyn Bridge, Yankee Stadium).

### 6.3 Production API Endpoints
- `POST /api/predict`: Live inference returning duration, fare, distance, congestion, and waypoint geometry.
- `GET /api/stats`: Comprehensive Phase 2 EDA dataset statistics and spatial hotspots.
- `GET /api/models`: Model evaluation metrics comparison and feature importance ranking.
- `GET /api/routes`: Pre-configured landmark route definitions.
- `GET /api/landmarks`: Geo-coordinates of iconic NYC points of interest.
- `GET /api/health`: Service health check.
