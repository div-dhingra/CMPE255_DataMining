/**
 * NYC Taxi DS Platform - Interactive Leaflet & ML Inference Engine
 */

// Application State
const state = {
  pickup: { lat: 40.7580, lon: -73.9855, name: "Times Square" },
  dropoff: { lat: 40.7061, lon: -73.9969, name: "Brooklyn Bridge" },
  passengerCount: 1,
  modelName: "gradient_boosting",
  pickupDatetime: null,
  landmarks: [],
  presetRoutes: [],
};

// Map & Layer References
let map = null;
let pickupMarker = null;
let dropoffMarker = null;
let routePolyline = null;

// Custom Leaflet Pin Icons
function createCustomIcon(color, glyph) {
  return L.divIcon({
    className: "custom-map-marker",
    html: `<div style="
      background-color: ${color};
      width: 32px;
      height: 32px;
      border-radius: 50% 50% 50% 0;
      transform: rotate(-45deg);
      border: 2px solid white;
      box-shadow: 0 4px 10px rgba(0,0,0,0.5);
      display: flex;
      align-items: center;
      justify-content: center;
    ">
      <div style="
        transform: rotate(45deg);
        color: white;
        font-size: 14px;
        font-weight: bold;
      ">${glyph}</div>
    </div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32],
  });
}

// Document Ready Initialization
document.addEventListener("DOMContentLoaded", async () => {
  initDateTime();
  initTabs();
  initMap();
  await loadLandmarksAndRoutes();
  await loadModelEvaluation();
  await triggerPrediction();
  setupEventListeners();
});

// 1. Initialize Default Datetime
function initDateTime() {
  const now = new Date();
  // Format as YYYY-MM-DDTHH:mm
  const localIso = new Date(now.getTime() - now.getTimezoneOffset() * 60000)
    .toISOString()
    .slice(0, 16);
  const dateInput = document.getElementById("pickupDateTimeInput");
  if (dateInput) {
    dateInput.value = localIso;
  }
}

// 2. Tab Navigation
function initTabs() {
  const tabButtons = [
    { btn: "btnTabMap", tab: "tabMap" },
    { btn: "btnTabModels", tab: "tabModels" },
    { btn: "btnTabCrisp", tab: "tabCrisp" },
  ];

  tabButtons.forEach(({ btn, tab }) => {
    const elBtn = document.getElementById(btn);
    if (!elBtn) return;
    elBtn.addEventListener("click", () => {
      // Deactivate all
      document.querySelectorAll(".nav-tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

      // Activate clicked
      elBtn.classList.add("active");
      const targetPane = document.getElementById(tab);
      if (targetPane) targetPane.classList.add("active");

      // Invalidate map size when switching back to map
      if (tab === "tabMap" && map) {
        setTimeout(() => map.invalidateSize(), 200);
      }
    });
  });
}

// 3. Initialize Leaflet Map
function initMap() {
  const nycCenter = [40.730610, -73.935242];
  map = L.map("leafletMap", {
    zoomControl: false,
    attributionControl: false,
  }).setView(nycCenter, 12);

  // Smooth Dark/Voyager CartoDB Tile Layer
  L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
    subdomains: "abcd",
  }).addTo(map);

  L.control.zoom({ position: "bottomright" }).addTo(map);

  // Markers
  const greenIcon = createCustomIcon("#10b981", "P");
  const redIcon = createCustomIcon("#ef4444", "D");

  pickupMarker = L.marker([state.pickup.lat, state.pickup.lon], {
    draggable: true,
    icon: greenIcon,
  }).addTo(map);
  pickupMarker.bindPopup("<b>Pickup Location</b><br>Drag to reposition");

  dropoffMarker = L.marker([state.dropoff.lat, state.dropoff.lon], {
    draggable: true,
    icon: redIcon,
  }).addTo(map);
  dropoffMarker.bindPopup("<b>Dropoff Location</b><br>Drag to reposition");

  // Drag Events
  pickupMarker.on("dragend", async (e) => {
    const pos = e.target.getLatLng();
    state.pickup.lat = Number(pos.lat.toFixed(5));
    state.pickup.lon = Number(pos.lng.toFixed(5));
    updateCoordLabels();
    await triggerPrediction();
  });

  dropoffMarker.on("dragend", async (e) => {
    const pos = e.target.getLatLng();
    state.dropoff.lat = Number(pos.lat.toFixed(5));
    state.dropoff.lon = Number(pos.lng.toFixed(5));
    updateCoordLabels();
    await triggerPrediction();
  });

  // Map Click (alternate pickup/dropoff repositioning if clicked)
  let clickToggle = false;
  map.on("click", async (e) => {
    if (clickToggle) {
      dropoffMarker.setLatLng(e.latlng);
      state.dropoff.lat = Number(e.latlng.lat.toFixed(5));
      state.dropoff.lon = Number(e.latlng.lng.toFixed(5));
    } else {
      pickupMarker.setLatLng(e.latlng);
      state.pickup.lat = Number(e.latlng.lat.toFixed(5));
      state.pickup.lon = Number(e.latlng.lng.toFixed(5));
    }
    clickToggle = !clickToggle;
    updateCoordLabels();
    await triggerPrediction();
  });
}

// 4. Update Coordinate Display Labels
function updateCoordLabels() {
  const pLabel = document.getElementById("pickupLabel");
  const dLabel = document.getElementById("dropoffLabel");
  if (pLabel) pLabel.textContent = `(${state.pickup.lat}, ${state.pickup.lon})`;
  if (dLabel) dLabel.textContent = `(${state.dropoff.lat}, ${state.dropoff.lon})`;
}

// 5. Load Landmarks & Routes from API
async function loadLandmarksAndRoutes() {
  try {
    const res = await fetch("/api/landmarks");
    if (res.ok) {
      const data = await res.json();
      state.landmarks = data.landmarks || [];
      populateLandmarkDropdowns();
    }

    const rRes = await fetch("/api/routes");
    if (rRes.ok) {
      const rData = await rRes.json();
      state.presetRoutes = rData.routes || [];
      populatePresetButtons();
    }
  } catch (err) {
    console.warn("Could not fetch landmarks or routes from API:", err);
  }
}

function populateLandmarkDropdowns() {
  const pSelect = document.getElementById("pickupLandmarkSelect");
  const dSelect = document.getElementById("dropoffLandmarkSelect");
  if (!pSelect || !dSelect) return;

  state.landmarks.forEach((lm) => {
    const optP = document.createElement("option");
    optP.value = lm.id;
    optP.textContent = `${lm.name} (${lm.borough})`;
    pSelect.appendChild(optP);

    const optD = document.createElement("option");
    optD.value = lm.id;
    optD.textContent = `${lm.name} (${lm.borough})`;
    dSelect.appendChild(optD);
  });

  pSelect.addEventListener("change", async (e) => {
    const lm = state.landmarks.find((l) => l.id === e.target.value);
    if (lm) {
      state.pickup.lat = lm.lat;
      state.pickup.lon = lm.lon;
      state.pickup.name = lm.name;
      pickupMarker.setLatLng([lm.lat, lm.lon]);
      updateCoordLabels();
      await triggerPrediction();
    }
  });

  dSelect.addEventListener("change", async (e) => {
    const lm = state.landmarks.find((l) => l.id === e.target.value);
    if (lm) {
      state.dropoff.lat = lm.lat;
      state.dropoff.lon = lm.lon;
      state.dropoff.name = lm.name;
      dropoffMarker.setLatLng([lm.lat, lm.lon]);
      updateCoordLabels();
      await triggerPrediction();
    }
  });
}

function populatePresetButtons() {
  const container = document.getElementById("presetButtons");
  if (!container || !state.presetRoutes.length) return;

  container.innerHTML = "";
  state.presetRoutes.forEach((route) => {
    const btn = document.createElement("button");
    btn.className = "preset-btn";
    btn.innerHTML = `📍 ${route.name}`;
    btn.title = route.description;
    btn.addEventListener("click", async () => {
      state.pickup.lat = route.pickup.lat;
      state.pickup.lon = route.pickup.lon;
      state.dropoff.lat = route.dropoff.lat;
      state.dropoff.lon = route.dropoff.lon;
      pickupMarker.setLatLng([route.pickup.lat, route.pickup.lon]);
      dropoffMarker.setLatLng([route.dropoff.lat, route.dropoff.lon]);
      updateCoordLabels();
      await triggerPrediction();
    });
    container.appendChild(btn);
  });
}

// 6. Setup Event Listeners
function setupEventListeners() {
  // Calculate Button
  const btnCalc = document.getElementById("btnCalculate");
  if (btnCalc) {
    btnCalc.addEventListener("click", triggerPrediction);
  }

  // Model Selection
  const modelSelect = document.getElementById("modelSelect");
  if (modelSelect) {
    modelSelect.addEventListener("change", async (e) => {
      state.modelName = e.target.value;
      const badge = document.getElementById("activeModelBadge");
      if (badge) badge.textContent = modelSelect.options[modelSelect.selectedIndex].text;
      await triggerPrediction();
    });
  }

  // Passenger Count
  const paxSelect = document.getElementById("passengerCount");
  if (paxSelect) {
    paxSelect.addEventListener("change", async (e) => {
      state.passengerCount = parseInt(e.target.value, 10);
      await triggerPrediction();
    });
  }

  // Quick Time Buttons
  document.querySelectorAll(".time-quick-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      document.querySelectorAll(".time-quick-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const timeType = btn.getAttribute("data-time");
      const dateInput = document.getElementById("pickupDateTimeInput");
      const baseDate = new Date();

      if (timeType === "morning_rush") {
        baseDate.setHours(8, 30, 0, 0);
      } else if (timeType === "midday") {
        baseDate.setHours(13, 30, 0, 0);
      } else if (timeType === "evening_rush") {
        baseDate.setHours(17, 45, 0, 0);
      } else if (timeType === "late_night") {
        baseDate.setHours(2, 0, 0, 0);
      }

      const iso = new Date(baseDate.getTime() - baseDate.getTimezoneOffset() * 60000)
        .toISOString()
        .slice(0, 16);
      if (dateInput) dateInput.value = iso;

      await triggerPrediction();
    });
  });

  // Custom Datetime change
  const dateInput = document.getElementById("pickupDateTimeInput");
  if (dateInput) {
    dateInput.addEventListener("change", async () => {
      await triggerPrediction();
    });
  }
}

// 7. Execute Real-Time Prediction API Call
async function triggerPrediction() {
  const dateInput = document.getElementById("pickupDateTimeInput");
  const dtVal = dateInput && dateInput.value ? dateInput.value.replace("T", " ") + ":00" : null;

  const payload = {
    pickup_latitude: state.pickup.lat,
    pickup_longitude: state.pickup.lon,
    dropoff_latitude: state.dropoff.lat,
    dropoff_longitude: state.dropoff.lon,
    passenger_count: state.passengerCount,
    pickup_datetime: dtVal,
    model_name: state.modelName,
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      console.error("Predict error", await res.text());
      return;
    }

    const data = await res.json();
    updateResultUI(data);
    drawRoutePolyline(data.route_geometry);
  } catch (err) {
    console.error("Failed to predict trip:", err);
  }
}

// 8. Update UI with Prediction Results
function updateResultUI(data) {
  // Big Metrics
  document.getElementById("predDuration").textContent = data.formatted_duration;
  document.getElementById("predDurationMinutes").textContent = `~${data.predicted_duration_minutes} minutes`;
  document.getElementById("predFare").textContent = `$${data.estimated_fare_usd.toFixed(2)}`;
  document.getElementById("predDistance").textContent = `${data.distance_miles} mi`;
  document.getElementById("predDistanceKm").textContent = `${data.distance_km} km (Manhattan: ${data.manhattan_distance_km} km)`;

  // Congestion
  const congFactor = data.congestion_factor;
  document.getElementById("predCongestion").textContent = `${congFactor.toFixed(2)}x`;
  const congLevelEl = document.getElementById("predCongestionLevel");
  if (congLevelEl) {
    congLevelEl.textContent = data.explanation.duration_components.congestion_level;
    if (congFactor > 2.0) {
      congLevelEl.style.color = "#ef4444";
    } else if (congFactor > 1.3) {
      congLevelEl.style.color = "#f59e0b";
    } else {
      congLevelEl.style.color = "#10b981";
    }
  }

  // Fare Stack Percentages
  const fareParts = data.explanation.fare_components;
  const totalF = Math.max(data.estimated_fare_usd, 1.0);
  const basePct = Math.max(5, (fareParts.base_meter_fare / totalF) * 100);
  const distPct = Math.max(10, (fareParts.distance_cost / totalF) * 100);
  const delayPct = Math.max(5, (fareParts.traffic_delay_charge / totalF) * 100);
  const surgePct = Math.max(0, 100 - (basePct + distPct + delayPct));

  const stack = document.getElementById("fareStack");
  if (stack) {
    stack.innerHTML = `
      <div class="bar-seg seg-base" style="width: ${basePct}%;" title="Base: $${fareParts.base_meter_fare}">Base</div>
      <div class="bar-seg seg-dist" style="width: ${distPct}%;" title="Distance: $${fareParts.distance_cost}">Dist</div>
      <div class="bar-seg seg-traffic" style="width: ${delayPct}%;" title="Delay: $${fareParts.traffic_delay_charge}">Delay</div>
      <div class="bar-seg seg-surcharge" style="width: ${surgePct}%;" title="Surge/Airport: $${fareParts.airport_tolls_surcharge + fareParts.rush_or_overnight_surcharge}">Surge</div>
    `;
  }

  // Duration Breakdown
  const durSec = data.explanation.duration_components;
  const freeMin = Math.floor(durSec.free_flow_seconds / 60);
  const freeSec = durSec.free_flow_seconds % 60;
  const delayMin = Math.floor(durSec.traffic_congestion_delay_seconds / 60);
  const delaySecRem = durSec.traffic_congestion_delay_seconds % 60;

  document.getElementById("durFreeFlow").textContent = `${freeMin}m ${freeSec}s`;
  document.getElementById("durDelay").textContent = `+${delayMin}m ${delaySecRem}s`;

  // Geographic context
  const feat = data.explanation.features_summary;
  document.getElementById("pickupBorough").textContent = feat.pickup_borough;
  document.getElementById("dropoffBorough").textContent = feat.dropoff_borough;
  document.getElementById("bearingVal").textContent = `${feat.bearing_degrees}°`;
  document.getElementById("airportFlag").textContent = feat.is_airport_trip ? "Airport Zone (+Surcharges)" : "Standard NYC Zone";
}

// 9. Draw Neon Glowing Polyline on Leaflet
function drawRoutePolyline(points) {
  if (!points || !points.length || !map) return;

  if (routePolyline) {
    map.removeLayer(routePolyline);
  }

  // Draw smooth polyline connecting route steps
  routePolyline = L.polyline(points, {
    color: "#f59e0b",
    weight: 5,
    opacity: 0.9,
    lineJoin: "round",
    dashArray: "8, 12",
  }).addTo(map);

  // Fit bounds to show entire route with padding
  const bounds = L.latLngBounds(points);
  map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
}

// 10. Load Model Evaluation Tables for Tab 2
async function loadModelEvaluation() {
  try {
    const res = await fetch("/api/models");
    if (!res.ok) return;

    const data = await res.json();
    const comparison = data.comparison;
    if (!comparison) return;

    // Populate Duration Table
    const durTbody = document.querySelector("#durationModelTable tbody");
    if (durTbody && comparison.duration_models) {
      durTbody.innerHTML = "";
      Object.entries(comparison.duration_models).forEach(([name, m]) => {
        const isBest = name === "gradient_boosting";
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>${formatModelName(name)}</strong> ${isBest ? '<span class="badge">Best</span>' : ''}</td>
          <td>${m.cv_rmse_mean.toFixed(2)}</td>
          <td class="${isBest ? 'best-metric' : ''}">${m.test_rmse.toFixed(2)}</td>
          <td>${m.test_rmsle.toFixed(4)}</td>
          <td>${m.test_mae.toFixed(2)}</td>
          <td class="${isBest ? 'best-metric' : ''}">${m.test_r2.toFixed(4)}</td>
          <td>${m.train_time_sec}s</td>
        `;
        durTbody.appendChild(row);
      });
    }

    // Populate Fare Table
    const fareTbody = document.querySelector("#fareModelTable tbody");
    if (fareTbody && comparison.fare_models) {
      fareTbody.innerHTML = "";
      Object.entries(comparison.fare_models).forEach(([name, m]) => {
        const isBest = name === "gradient_boosting";
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>${formatModelName(name)}</strong> ${isBest ? '<span class="badge">Best</span>' : ''}</td>
          <td>$${m.cv_rmse_mean.toFixed(2)}</td>
          <td class="${isBest ? 'best-metric' : ''}">$${m.test_rmse.toFixed(2)}</td>
          <td>$${m.test_mae.toFixed(2)}</td>
          <td class="${isBest ? 'best-metric' : ''}">${m.test_r2.toFixed(4)}</td>
          <td>${m.train_time_sec}s</td>
        `;
        fareTbody.appendChild(row);
      });
    }

    // Feature Importances
    const fiContainer = document.getElementById("featureImportanceContainer");
    if (fiContainer && data.feature_importance && data.feature_importance.duration) {
      fiContainer.innerHTML = "";
      const dFeatures = data.feature_importance.duration["gradient_boosting"] || [];
      dFeatures.slice(0, 8).forEach((item) => {
        const row = document.createElement("div");
        row.className = "fi-row";
        row.innerHTML = `
          <div class="fi-name">${item.feature}</div>
          <div class="fi-bar-track">
            <div class="fi-bar-fill" style="width: ${Math.min(100, item.importance * 1.8)}%;"></div>
          </div>
          <div class="fi-value">${item.importance.toFixed(1)}%</div>
        `;
        fiContainer.appendChild(row);
      });
    }
  } catch (err) {
    console.warn("Could not load model evaluations:", err);
  }
}

function formatModelName(str) {
  return str
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
