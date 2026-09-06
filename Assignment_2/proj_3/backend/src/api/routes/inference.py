"""
Real-Time Single and Batch Customer Inference Route Handlers.
Endpoints:
- POST /api/v1/inference/predict: Real-time scoring of a single customer feature vector
- POST /api/v1/inference/batch: Bulk classification of multiple customer records
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, status
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ..schemas import (
    SingleInferenceRequest,
    SingleInferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    BatchInferenceResultItem,
    InferencePersonaMatch,
)
from ..state import state
from .cluster import _ensure_active_clustering
from crisp_dm.data_understanding import NUMERIC_FEATURE_COLUMNS

router = APIRouter(prefix="/inference", tags=["Inference & Deployment"])


def _prepare_input_df(features_list: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Constructs a valid DataFrame from raw input feature dictionaries,
    aligning with expected dataset columns and imputing missing fields with dataset medians.
    """
    df_raw = state.get_dataset()
    input_df = pd.DataFrame(features_list)

    # Ensure all numeric columns exist
    for col in NUMERIC_FEATURE_COLUMNS:
        if col not in input_df.columns:
            median_val = float(df_raw[col].median()) if col in df_raw.columns else 0.0
            input_df[col] = median_val

    # Reorder columns to match dataset
    ordered_cols = [c for c in df_raw.columns if c != "CUST_ID"]
    # If feature engineered columns exist in pipeline, pipeline handles them
    return input_df[[c for c in ordered_cols if c in input_df.columns]]


def _get_persona_match(cluster_id: int) -> InferencePersonaMatch:
    """Retrieves persona narrative and marketing recommendation for a cluster ID."""
    cached = state.get_last_result()
    personas_raw = cached.get("personas_raw", []) if cached else []

    matched = None
    for p in personas_raw:
        if p.get("cluster_id") == cluster_id:
            matched = p
            break

    if matched is not None:
        return InferencePersonaMatch(
            cluster_id=cluster_id,
            persona_name=matched.get("persona_name", f"Cluster {cluster_id}"),
            business_description=matched.get("business_description", "Standard Customer"),
            marketing_strategy=matched.get("marketing_strategy", "Standard Engagement"),
            archetype_tags=[
                f"Cluster {cluster_id}",
                matched.get("persona_name", ""),
            ],
        )

    # Fallback persona
    name = f"Cluster {cluster_id}" if cluster_id >= 0 else "Anomaly / Outlier"
    return InferencePersonaMatch(
        cluster_id=cluster_id,
        persona_name=name,
        business_description="Customer account exhibiting distinct financial behavioral patterns.",
        marketing_strategy="Standard targeted promotional campaigns and account monitoring.",
        archetype_tags=[name],
    )


@router.post(
    "/predict",
    response_model=SingleInferenceResponse,
    summary="Classify a single customer feature vector and predict persona",
)
def predict_single(request: SingleInferenceRequest) -> SingleInferenceResponse:
    """
    Accepts arbitrary customer financial behavioral attributes, executes the active
    preprocessing pipeline, predicts cluster assignment, soft membership probabilities,
    and returns matched persona insights.
    """
    _ensure_active_clustering()
    pipeline = state.get_pipeline()
    model = state.get_model()

    if pipeline is None or model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clustering pipeline is not initialized.",
        )

    input_df = _prepare_input_df([request.features])

    try:
        X_trans = pipeline.transform(input_df)
        if hasattr(pipeline, "pca_components_") and pipeline.pca_components_ is not None:
            X_trans = (X_trans - pipeline.pca_mean_) @ pipeline.pca_vt_[:pipeline.pca_components_].T
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pipeline transformation error: {str(ex)}",
        )

    labels = model.predict(X_trans)
    cluster_id = int(labels[0])

    # Soft membership probabilities
    proba_arr = model.predict_proba(X_trans)
    probabilities: Dict[str, float] = {}
    if proba_arr is not None and len(proba_arr) > 0:
        for idx, prob in enumerate(proba_arr[0]):
            probabilities[f"cluster_{idx}"] = round(float(prob), 4)

    # Centroid distances if available
    distances_dict: Optional[Dict[str, float]] = None
    if hasattr(model, "centroids_") and model.centroids_ is not None:
        dists = cdist(X_trans, model.centroids_, metric="euclidean")[0]
        distances_dict = {
            f"cluster_{idx}": round(float(d), 4) for idx, d in enumerate(dists)
        }

    persona = _get_persona_match(cluster_id)

    return SingleInferenceResponse(
        cluster_id=cluster_id,
        probabilities=probabilities,
        distances_to_centroids=distances_dict,
        persona=persona,
    )


@router.post(
    "/batch",
    response_model=BatchInferenceResponse,
    summary="Bulk classify a batch of customer records",
)
def predict_batch(request: BatchInferenceRequest) -> BatchInferenceResponse:
    """
    Classifies a batch of customer financial records, returning predictions,
    membership probabilities, and aggregate cluster distribution counts.
    """
    if len(request.customers) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch request must contain at least one customer record.",
        )

    _ensure_active_clustering()
    pipeline = state.get_pipeline()
    model = state.get_model()

    if pipeline is None or model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clustering pipeline is not initialized.",
        )

    input_df = _prepare_input_df(request.customers)

    try:
        X_trans = pipeline.transform(input_df)
        if hasattr(pipeline, "pca_components_") and pipeline.pca_components_ is not None:
            X_trans = (X_trans - pipeline.pca_mean_) @ pipeline.pca_vt_[:pipeline.pca_components_].T
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch transformation error: {str(ex)}",
        )

    labels = model.predict(X_trans)
    proba_arr = model.predict_proba(X_trans)

    predictions: List[BatchInferenceResultItem] = []
    cluster_counts: Dict[str, int] = {}

    for idx, c_id in enumerate(labels):
        c_int = int(c_id)
        c_key = f"cluster_{c_int}" if c_int >= 0 else "noise"
        cluster_counts[c_key] = cluster_counts.get(c_key, 0) + 1

        probs: Dict[str, float] = {}
        if proba_arr is not None and len(proba_arr) > idx:
            for p_idx, prob in enumerate(proba_arr[idx]):
                probs[f"cluster_{p_idx}"] = round(float(prob), 4)

        persona = _get_persona_match(c_int)
        predictions.append(
            BatchInferenceResultItem(
                index=idx,
                cluster_id=c_int,
                probabilities=probs,
                persona_name=persona.persona_name,
            )
        )

    return BatchInferenceResponse(
        total_samples=len(request.customers),
        predictions=predictions,
        cluster_counts=cluster_counts,
    )
