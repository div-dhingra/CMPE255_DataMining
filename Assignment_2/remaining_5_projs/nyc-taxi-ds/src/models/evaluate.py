"""
Model Evaluation and Metrics Module.
Calculates RMSE, RMSLE, MAE, and R², computes residual distributions,
and extracts feature importance rankings for Phase 5.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive regression evaluation metrics:
    - RMSE (Root Mean Squared Error)
    - RMSLE (Root Mean Squared Logarithmic Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Ensure non-negative for predictions
    y_pred = np.maximum(y_pred, 0.0)
    y_true = np.maximum(y_true, 0.0)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # RMSLE
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    rmsle = np.sqrt(mean_squared_error(log_true, log_pred))

    return {
        "rmse": round(float(rmse), 3),
        "rmsle": round(float(rmsle), 4),
        "mae": round(float(mae), 3),
        "r2": round(float(r2), 4),
    }


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray, max_samples: int = 500) -> Dict[str, Any]:
    """
    Compute residual distribution for error analysis and residual plotting.
    Sample points for lightweight JSON serialization.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    n_pts = min(len(y_true), max_samples)
    idx = np.random.choice(len(y_true), size=n_pts, replace=False)

    # Error bins histogram
    hist_counts, bin_edges = np.histogram(residuals, bins=15)

    return {
        "residual_mean": round(float(np.mean(residuals)), 3),
        "residual_std": round(float(np.std(residuals)), 3),
        "percentile_25": round(float(np.percentile(residuals, 25)), 3),
        "median": round(float(np.median(residuals)), 3),
        "percentile_75": round(float(np.percentile(residuals, 75)), 3),
        "histogram": {
            "counts": [int(c) for c in hist_counts],
            "bin_edges": [round(float(b), 2) for b in bin_edges],
        },
        "sample_points": [
            {
                "y_true": round(float(y_true[i]), 2),
                "y_pred": round(float(y_pred[i]), 2),
                "residual": round(float(residuals[i]), 2),
            }
            for i in idx
        ],
    }


def extract_feature_importances(
    model,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Extract normalized feature importances or coefficient magnitudes.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        # Fallback uniform
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Normalize to 100%
    total = np.sum(importances)
    if total > 0:
        norm_imp = (importances / total) * 100.0
    else:
        norm_imp = importances

    ranking = [
        {"feature": name, "importance": round(float(val), 2)}
        for name, val in zip(feature_names, norm_imp)
    ]
    ranking.sort(key=lambda x: x["importance"], reverse=True)
    return ranking
