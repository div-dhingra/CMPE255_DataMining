# NYC Taxi DS - Video Demo Script

**Target Audience:** Professor / Grader
**Goal:** Demonstrate the end-to-end NYC Taxi ML platform, connecting the underlying code directly to the interactive UI.

---

## 1. Introduction & Overview (0:00 - 0:15)
**[Screen Recording: Show the main web interface (Leaflet map and dashboards)]**
**Voiceover:** "Hello, this is a demonstration of the NYC Taxi Data Science Platform. We built a full CRISP-DM machine learning pipeline that predicts trip durations and fares, served via a FastAPI backend and a Leaflet interactive frontend."

---

## 2. Code-to-UI Connection 1: The Inference API (0:15 - 0:40)
**[Screen Recording: Open `src/api/routes.py`, highlight the `@router.post("/predict")` endpoint]**
**Voiceover:** "Let's look at how predictions are requested. In our backend, specifically `src/api/routes.py`, we define the `/predict` POST endpoint. This takes the pickup and dropoff coordinates, passenger count, and chosen model, and passes it to our ML predictor."

**[Screen Recording: Switch back to UI, click the 'Calculate Prediction' button, show the big Duration and Fare numbers updating.]**
**Voiceover:** "In the UI, when we click 'Calculate Prediction' or move a marker, the frontend sends a payload to this exact endpoint. The returned results instantly update the primary duration and fare metrics shown in the dashboard header."

---

## 3. Code-to-UI Connection 2: Model Explainability & Constraints (0:40 - 1:10)
**[Screen Recording: Open `src/models/predictor.py`, highlight the `predict_trip` method, specifically the `explanation` dictionary logic (e.g., congestion factor, fare components).]**
**Voiceover:** "Behind that endpoint is the inference engine in `src/models/predictor.py`. Within the `predict_trip` method, we don't just output raw model predictions. We calculate a physical congestion factor, enforce minimum base fares, and decompose the fare into base, distance, and traffic delay components based on geospatial features."

**[Screen Recording: Switch to UI, highlight the 'Congestion Level' text and the 'Fare Breakdown' horizontal stacked bar chart.]**
**Voiceover:** "On the dashboard, this logic translates directly into the Explainability section. You can see the 'Congestion Level' indicator adapting dynamically, and the 'Fare Breakdown' stack visually separating the base fare, distance cost, and traffic delay based on the dictionary returned by that Python file."

---

## 4. Code-to-UI Connection 3: Interactive Leaflet Map (1:10 - 1:40)
**[Screen Recording: Open `static/app.js`, highlight the `marker.on("dragend", ...)` event listeners and the `drawRoutePolyline` function.]**
**Voiceover:** "Finally, let's look at the map interactivity. In `static/app.js`, we have drag event listeners attached to our Leaflet markers. Whenever a pin is dropped, it updates the application state and automatically calls `triggerPrediction()`. We also have a `drawRoutePolyline` function that renders the synthetic route geometry returned by the API."

**[Screen Recording: Switch to UI, drag the green pickup pin across Manhattan. Show the glowing orange route polyline redrawing and the metrics updating seamlessly.]**
**Voiceover:** "As I drag the pickup marker on the map, you can see those event listeners firing in real-time. The orange polyline instantly redraws to connect the new points, and the prediction metrics update automatically without needing to refresh the page."

---

## 5. Conclusion (1:40 - 1:50)
**[Screen Recording: Show the Models tab with the comparison table, then return to the main map.]**
**Voiceover:** "The platform also includes full EDA statistics and cross-validation metrics for four different ML models. This concludes the demo of the NYC Taxi Trip Estimator. Thank you!"
