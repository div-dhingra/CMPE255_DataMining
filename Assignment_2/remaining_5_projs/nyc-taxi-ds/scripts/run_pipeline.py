"""
End-to-End CRISP-DM Pipeline Runner.
Executes all 6 CRISP-DM phases:
- Phase 1: Business Understanding
- Phase 2: Data Understanding & EDA
- Phase 3: Data Preparation & Geospatial Features
- Phase 4: Modeling (Linear, Ridge, Random Forest, Gradient Boosting)
- Phase 5: Evaluation & Metrics Export
- Phase 6: Deployment Verification
"""
import sys
import json
import logging
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import run_training_pipeline
from src.data.loader import load_dataset, clean_and_filter_data
from src.data.eda import generate_eda_report
from src.models.predictor import get_predictor
from src.config import MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CRISP-DM-Pipeline")


def main():
    print("=" * 70)
    print("      NYC TAXI TRIP DURATION & FARE ESTIMATION (CRISP-DM)")
    print("=" * 70)

    # -------------------------------------------------------------
    # Phase 1: Business Understanding
    # -------------------------------------------------------------
    logger.info("[PHASE 1: Business Understanding]")
    logger.info("  * Objectives: Optimize fleet dispatching, dynamic pricing, & rider ETA.")
    logger.info("  * Target 1: Trip Duration (seconds / minutes).")
    logger.info("  * Target 2: Fare Amount (USD).")
    logger.info("  * Success Metrics: RMSE, RMSLE, MAE, R² score.")

    # -------------------------------------------------------------
    # Phase 2: Data Understanding
    # -------------------------------------------------------------
    logger.info("\n[PHASE 2: Data Understanding]")
    raw_df = load_dataset(n_synthetic_samples=20000)
    clean_df = clean_and_filter_data(raw_df)
    eda_report = generate_eda_report(clean_df)

    summary = eda_report["summary"]
    logger.info(f"  * Total Clean Trips: {summary['total_trips']:,}")
    logger.info(f"  * Mean Duration: {summary['trip_duration']['mean_minutes']} min (Median: {round(summary['trip_duration']['median_seconds']/60, 1)} min)")
    if summary["fare_amount"]["mean_usd"]:
        logger.info(f"  * Mean Fare: ${summary['fare_amount']['mean_usd']} (Median: ${summary['fare_amount']['median_usd']})")
    logger.info(f"  * Mean Distance: {summary['distance_km']['mean']} km ({summary['distance_km']['mean_miles']} miles)")
    logger.info(f"  * Peak Congestion Hours: {eda_report['temporal']['peak_hours']}")
    logger.info(f"  * Discovered Pickup Hotspots: {len(eda_report['hotspots'])} clusters (Midtown, FiDi, JFK, LGA)")

    # -------------------------------------------------------------
    # Phase 3: Data Preparation & Phase 4: Modeling & Phase 5: Evaluation
    # -------------------------------------------------------------
    logger.info("\n[PHASE 3, 4, 5: Feature Engineering, Model Training & Cross-Validation]")
    summary_results = run_training_pipeline(n_samples=20000, test_size=0.2, random_state=42)

    logger.info("\n[PHASE 5: Evaluation Comparison Table]")
    print("\n--- TRIP DURATION MODELS ---")
    print(f"{'Model':<22} | {'Test RMSE (s)':<14} | {'Test RMSLE':<12} | {'Test MAE (s)':<12} | {'Test R²':<10}")
    print("-" * 75)
    for m_name, m_stats in summary_results["duration_models"].items():
        print(f"{m_name:<22} | {m_stats['test_rmse']:<14.2f} | {m_stats['test_rmsle']:<12.4f} | {m_stats['test_mae']:<12.2f} | {m_stats['test_r2']:<10.4f}")

    print("\n--- FARE AMOUNT MODELS ---")
    print(f"{'Model':<22} | {'Test RMSE ($)':<14} | {'Test MAE ($)':<12} | {'Test R²':<10}")
    print("-" * 65)
    for m_name, m_stats in summary_results["fare_models"].items():
        print(f"{m_name:<22} | ${m_stats['test_rmse']:<13.2f} | ${m_stats['test_mae']:<11.2f} | {m_stats['test_r2']:<10.4f}")

    # -------------------------------------------------------------
    # Phase 6: Deployment Test
    # -------------------------------------------------------------
    logger.info("\n[PHASE 6: Deployment Verification]")
    predictor = get_predictor()
    predictor.ensure_loaded()
    test_trip = predictor.predict_trip(
        pickup_lat=40.7580, pickup_lon=-73.9855,  # Times Square
        dropoff_lat=40.7061, dropoff_lon=-73.9969,  # Brooklyn Bridge
        passenger_count=1,
    )
    logger.info(f"  * Sample Live Inference: Times Square -> Brooklyn Bridge")
    logger.info(f"    - Predicted Duration: {test_trip['formatted_duration']} ({test_trip['predicted_duration_minutes']} min)")
    logger.info(f"    - Estimated Fare: ${test_trip['estimated_fare_usd']}")
    logger.info(f"    - Distance: {test_trip['distance_miles']} miles ({test_trip['distance_km']} km)")
    logger.info(f"    - Congestion Factor: {test_trip['congestion_factor']}x")
    logger.info(f"    - Route Polyline Steps: {len(test_trip['route_geometry'])} waypoints")

    print("\n" + "=" * 70)
    print("  CRISP-DM PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
