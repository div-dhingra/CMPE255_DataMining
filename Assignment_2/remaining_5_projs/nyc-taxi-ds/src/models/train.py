"""
Model Training Pipeline with Cross-Validation for CRISP-DM Phase 4.
Trains Linear Regression, Ridge, Random Forest, and Gradient Boosting models
for both Trip Duration and Fare Amount prediction.
"""
import time
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split

from src.config import MODELS_DIR, FEATURE_COLS
from src.data.loader import load_dataset, clean_and_filter_data
from src.features.pipeline import TaxiFeaturePipeline, extract_features
from src.models.evaluate import compute_metrics, compute_residuals, extract_feature_importances

logger = logging.getLogger(__name__)


def build_model_zoo(random_state: int = 42) -> Dict[str, Any]:
    """Instantiate the 4 target candidate model architectures."""
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=40,
            max_depth=12,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=random_state,
        ),
    }


def train_and_evaluate_target(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
    n_splits: int = 3,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Perform K-Fold Cross-Validation and final holdout evaluation across all candidate models.
    Returns:
      - trained_models: dict of fitted models
      - cv_results: dict of validation metrics per model
      - evaluation_details: residuals and feature importance rankings
    """
    logger.info(f"--- Training models for target: {target_name} ---")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    trained_models = {}
    model_metrics = {}
    residuals_data = {}
    feature_importances = {}

    model_zoo = build_model_zoo(random_state=random_state)

    for model_name, model in model_zoo.items():
        logger.info(f"Evaluating {model_name}...")
        start_time = time.time()

        # Cross-validation
        cv_scores = {"rmse": [], "rmsle": [], "mae": [], "r2": []}
        for train_idx, val_idx in kf.split(X_train):
            fold_X_tr, fold_y_tr = X_train[train_idx], y_train[train_idx]
            fold_X_val, fold_y_val = X_train[val_idx], y_train[val_idx]

            fold_model = build_model_zoo(random_state=random_state)[model_name]
            fold_model.fit(fold_X_tr, fold_y_tr)
            val_preds = fold_model.predict(fold_X_val)

            fold_metrics = compute_metrics(fold_y_val, val_preds)
            for k, v in fold_metrics.items():
                cv_scores[k].append(v)

        # Train on full training set
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        test_metrics = compute_metrics(y_test, test_preds)
        train_duration = round(time.time() - start_time, 2)

        trained_models[model_name] = model

        model_metrics[model_name] = {
            "model_name": model_name,
            "train_time_sec": train_duration,
            "cv_rmse_mean": round(float(np.mean(cv_scores["rmse"])), 3),
            "cv_rmse_std": round(float(np.std(cv_scores["rmse"])), 3),
            "cv_mae_mean": round(float(np.mean(cv_scores["mae"])), 3),
            "cv_r2_mean": round(float(np.mean(cv_scores["r2"])), 4),
            "test_rmse": test_metrics["rmse"],
            "test_rmsle": test_metrics["rmsle"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
        }

        residuals_data[model_name] = compute_residuals(y_test, test_preds, max_samples=300)
        feature_importances[model_name] = extract_feature_importances(model, FEATURE_COLS)

        logger.info(
            f"  {model_name} -> Test RMSE: {test_metrics['rmse']}, Test MAE: {test_metrics['mae']}, "
            f"Test R²: {test_metrics['r2']} (Time: {train_duration}s)"
        )

    return trained_models, model_metrics, {
        "residuals": residuals_data,
        "feature_importance": feature_importances,
    }


def run_training_pipeline(
    n_samples: int = 25000,
    test_size: float = 0.2,
    random_state: int = 42,
    force_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Execute end-to-end CRISP-DM training pipeline:
    1. Load / synthesize raw data
    2. Clean and filter within NYC bounds
    3. Fit feature pipeline
    4. Train & validate duration and fare models
    5. Save models, metrics, and residual plots data to disk
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & clean data
    raw_df = load_dataset(n_synthetic_samples=n_samples, force_synthetic=force_synthetic)
    clean_df = clean_and_filter_data(raw_df)

    # 2. Extract features and fit pipeline
    pipeline = TaxiFeaturePipeline(feature_cols=FEATURE_COLS)
    pipeline.fit(clean_df)
    pipeline.save()

    X = pipeline.transform(clean_df)
    y_duration = clean_df["trip_duration"].values
    y_fare = clean_df["fare_amount"].values

    # Train / test split
    indices = np.arange(len(clean_df))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    X_train, X_test = X[train_idx], X[test_idx]
    y_dur_train, y_dur_test = y_duration[train_idx], y_duration[test_idx]
    y_fare_train, y_fare_test = y_fare[train_idx], y_fare[test_idx]

    # 3. Train duration models
    duration_models, dur_metrics, dur_eval = train_and_evaluate_target(
        X_train, y_dur_train, X_test, y_dur_test,
        target_name="trip_duration",
        random_state=random_state,
    )
    joblib.dump(duration_models, MODELS_DIR / "duration_models.joblib")

    # 4. Train fare models
    fare_models, fare_metrics, fare_eval = train_and_evaluate_target(
        X_train, y_fare_train, X_test, y_fare_test,
        target_name="fare_amount",
        random_state=random_state,
    )
    joblib.dump(fare_models, MODELS_DIR / "fare_models.joblib")

    # 5. Save model comparison and evaluation JSON artifacts
    comparison_summary = {
        "dataset_info": {
            "total_records": len(clean_df),
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "features_count": len(FEATURE_COLS),
            "features": FEATURE_COLS,
        },
        "duration_models": dur_metrics,
        "fare_models": fare_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison_summary, f, indent=2)

    with open(MODELS_DIR / "feature_importance.json", "w") as f:
        json.dump({
            "duration": dur_eval["feature_importance"],
            "fare": fare_eval["feature_importance"],
        }, f, indent=2)

    with open(MODELS_DIR / "residual_data.json", "w") as f:
        json.dump({
            "duration": dur_eval["residuals"],
            "fare": fare_eval["residuals"],
        }, f, indent=2)

    logger.info("Training pipeline completed successfully. Artifacts saved in models/.")
    return comparison_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training_pipeline()
