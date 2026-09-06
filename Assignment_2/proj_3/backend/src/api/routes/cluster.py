"""
Clustering Execution, Profiling & Projection Route Handlers.
Endpoints:
- POST /api/v1/cluster/run: Execute clustering pipeline & model training
- GET  /api/v1/cluster/projections: 2D/3D projection coordinates (PCA/UMAP/t-SNE)
- GET  /api/v1/cluster/profiles: Cluster personas, radar profiles, and feature importances
- GET  /api/v1/cluster/silhouette-samples: Per-sample silhouette ribbon arrays
- GET  /api/v1/cluster/elbow: WCSS inertia curve and Kneedle elbow detection
"""

import time
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Query, HTTPException, status
import numpy as np
import pandas as pd

from ..schemas import (
    ClusteringRequest,
    ClusteringResponse,
    ClusterPersona,
    DistinguishingFeature,
    ProjectionsResponse,
    ClusterProfilesResponse,
    SilhouetteSamplesResponse,
    SilhouetteSampleItem,
    ElbowResponse,
)
from ..state import state
from crisp_dm.data_preparation import (
    PipelineConfig,
    DataPreparationPipeline,
)
from crisp_dm.models import (
    ClusteringModelBase,
    KMeansModel,
    KMedoidsModel,
    DBSCANModel,
    HDBSCANModel,
    AgglomerativeModel,
    GaussianMixtureModel,
)
from crisp_dm.evaluation import (
    evaluate_clustering_solution,
    compute_silhouette_score,
    compute_subsampling_stability,
    compute_inertia_elbow_curve,
)
from crisp_dm.projections import (
    project_coordinates,
    compute_pca_projections,
)
from crisp_dm.profiling import (
    compute_cluster_centroids,
    compute_radar_profiles,
    compute_feature_importance,
    generate_cluster_personas,
)

router = APIRouter(prefix="/cluster", tags=["Clustering & Evaluation"])


def _instantiate_model(algorithm: str, params: Dict[str, Any], random_state: int = 42) -> ClusteringModelBase:
    """Instantiates a clustering model by algorithm name and parameter dictionary."""
    alg = algorithm.lower()
    p = dict(params)

    if alg in ("kmeans", "k-means"):
        n_clusters = int(p.get("n_clusters", 4))
        init = str(p.get("init", "k-means++"))
        max_iter = int(p.get("max_iter", 300))
        n_init = int(p.get("n_init", 10))
        return KMeansModel(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)

    elif alg in ("kmedoids", "k-medoids", "k_medoids"):
        n_clusters = int(p.get("n_clusters", 4))
        metric = str(p.get("metric", "euclidean"))
        max_iter = int(p.get("max_iter", 300))
        method = str(p.get("method", "fasterpam"))
        return KMedoidsModel(n_clusters=n_clusters, metric=metric, max_iter=max_iter, method=method, random_state=random_state)

    elif alg == "dbscan":
        eps = float(p.get("eps", 1.2))
        min_samples = int(p.get("min_samples", 5))
        metric = str(p.get("metric", "euclidean"))
        return DBSCANModel(eps=eps, min_samples=min_samples, metric=metric)

    elif alg == "hdbscan":
        min_cluster_size = int(p.get("min_cluster_size", 10))
        min_samples = int(p.get("min_samples", 5))
        metric = str(p.get("metric", "euclidean"))
        return HDBSCANModel(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)

    elif alg in ("agglomerative", "hierarchical"):
        n_clusters = int(p.get("n_clusters", 4))
        linkage = str(p.get("linkage", "ward"))
        metric = str(p.get("metric", "euclidean"))
        return AgglomerativeModel(n_clusters=n_clusters, linkage=linkage, metric=metric)

    elif alg in ("gmm", "gaussian_mixture"):
        n_clusters = int(p.get("n_clusters", 4))
        covariance_type = str(p.get("covariance_type", "full"))
        max_iter = int(p.get("max_iter", 100))
        reg_covar = float(p.get("reg_covar", 1e-6))
        return GaussianMixtureModel(n_clusters=n_clusters, covariance_type=covariance_type, max_iter=max_iter, reg_covar=reg_covar, random_state=random_state)

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown clustering algorithm: '{algorithm}'. Choose from kmeans, kmedoids, dbscan, hdbscan, agglomerative, gmm.",
        )


def _ensure_active_clustering() -> Dict[str, Any]:
    """Helper to ensure a clustering result is cached; runs default KMeans if none exists."""
    cached = state.get_last_result()
    if cached is not None:
        return cached

    # Run default KMeans clustering
    df = state.get_dataset()
    pipe_config = PipelineConfig(
        imputer_strategy="median",
        outlier_strategy="winsorize",
        scaling_strategy="yeo_johnson",
        engineer_ratios=True,
    )
    pipeline = DataPreparationPipeline(pipe_config)
    X_prep = pipeline.fit_transform(df)

    model = KMeansModel(n_clusters=4, random_state=42)
    labels = model.fit_predict(X_prep)

    eval_summary = evaluate_clustering_solution(X_prep, labels, model)
    personas_raw = generate_cluster_personas(df, labels)
    proj = project_coordinates(X_prep, method="pca", random_state=42)

    result = {
        "algorithm": "kmeans",
        "labels": labels,
        "eval_summary": eval_summary,
        "personas_raw": personas_raw,
        "X_prep": X_prep,
        "projections": proj,
        "execution_time_ms": 100.0,
    }
    state.set_clustering_state(pipeline, model, result)
    return result


@router.post(
    "/run",
    response_model=ClusteringResponse,
    summary="Execute clustering pipeline and model training",
)
def run_clustering(request: ClusteringRequest) -> ClusteringResponse:
    """
    Executes the CRISP-DM preprocessing pipeline on active data, fits the selected
    clustering algorithm, computes internal validation metrics, synthesizes business personas,
    and returns complete clustering results.
    """
    start_time = time.perf_counter()
    df = state.get_dataset()

    # Build pipeline
    p_conf = PipelineConfig(
        imputer_strategy=request.pipeline.imputer,
        outlier_strategy=request.pipeline.outlier_method,
        scaling_strategy=request.pipeline.scaler,
        engineer_ratios=request.pipeline.feature_engineering,
    )
    pipeline = DataPreparationPipeline(p_conf)

    try:
        X_prep = pipeline.fit_transform(df)
        if request.pipeline.pca_components is not None and request.pipeline.pca_components < X_prep.shape[1]:
            from crisp_dm.projections import compute_pca_projections
            pca_dict = compute_pca_projections(X_prep, n_components=request.pipeline.pca_components, random_state=request.random_state)
            pipeline.pca_components_ = request.pipeline.pca_components
            pipeline.pca_mean_ = np.mean(X_prep, axis=0)
            pipeline.pca_vt_ = np.array(pca_dict.get("components"))
            X_prep = (X_prep - pipeline.pca_mean_) @ pipeline.pca_vt_[:request.pipeline.pca_components].T
        else:
            pipeline.pca_components_ = None
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preprocessing pipeline error: {str(ex)}",
        )

    # Build and fit model
    model = _instantiate_model(request.algorithm, request.params, random_state=request.random_state)
    try:
        labels = model.fit_predict(X_prep)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model fitting error ({request.algorithm}): {str(ex)}",
        )

    # Evaluation
    eval_summary = evaluate_clustering_solution(X_prep, labels, model)

    # Personas
    personas_raw = generate_cluster_personas(df, labels)
    personas: List[ClusterPersona] = []
    for p in personas_raw:
        dist_feats = [
            DistinguishingFeature(
                feature=d["feature"],
                cluster_mean=d["cluster_mean"],
                global_mean=d["global_mean"],
                z_score=d["z_score"],
            )
            for d in p.get("top_distinguishing_features", [])
        ]
        personas.append(
            ClusterPersona(
                cluster_id=p["cluster_id"],
                persona_name=p["persona_name"],
                sample_count=p["sample_count"],
                percentage=p["percentage"],
                business_description=p["business_description"],
                marketing_strategy=p["marketing_strategy"],
                top_distinguishing_features=dist_feats,
                radar_metrics=p.get("radar_metrics", {}),
            )
        )

    # Compute quick 2D/3D projections
    proj = project_coordinates(X_prep, method="pca", random_state=request.random_state)
    coords_2d = proj.get("coords_2d")
    coords_3d = proj.get("coords_3d")

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Cache state
    result_dict = {
        "algorithm": request.algorithm,
        "labels": labels,
        "eval_summary": eval_summary,
        "personas_raw": personas_raw,
        "personas": personas,
        "X_prep": X_prep,
        "projections": proj,
        "execution_time_ms": elapsed_ms,
    }
    state.set_clustering_state(pipeline, model, result_dict)

    return ClusteringResponse(
        algorithm=request.algorithm,
        n_clusters=eval_summary["n_clusters"],
        labels=labels.tolist(),
        silhouette_score=eval_summary["silhouette_score"],
        davies_bouldin_index=eval_summary["davies_bouldin_index"],
        calinski_harabasz_score=eval_summary["calinski_harabasz_score"],
        stability_ari=eval_summary.get("stability_ari"),
        inertia=eval_summary.get("inertia"),
        bic=eval_summary.get("bic"),
        aic=eval_summary.get("aic"),
        noise_points_count=eval_summary["noise_points_count"],
        noise_ratio=eval_summary["noise_ratio"],
        cluster_sizes={str(k): v for k, v in eval_summary["cluster_sizes"].items()},
        personas=personas,
        sample_projections_2d=coords_2d.tolist() if coords_2d is not None else None,
        sample_projections_3d=coords_3d.tolist() if coords_3d is not None else None,
        execution_time_ms=round(elapsed_ms, 2),
    )


@router.get(
    "/projections",
    response_model=ProjectionsResponse,
    summary="Get 2D/3D projection coordinates (PCA/UMAP/t-SNE)",
)
def get_projections(
    method: str = Query(default="pca", description="Projection method: pca, umap, tsne"),
    random_state: int = Query(default=42, description="Random seed"),
) -> ProjectionsResponse:
    """
    Computes or retrieves 2D and 3D coordinate embeddings (PCA, UMAP, t-SNE)
    for the active clustering dataset.
    """
    cached = _ensure_active_clustering()
    X_prep = cached["X_prep"]
    labels = cached["labels"]

    try:
        proj = project_coordinates(X_prep, method=method, random_state=random_state)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Projection error with method '{method}': {str(ex)}",
        )

    coords_2d = proj.get("coords_2d")
    coords_3d = proj.get("coords_3d")
    evr = proj.get("explained_variance_ratio")

    return ProjectionsResponse(
        method=method.lower(),
        explained_variance_ratio=evr,
        coords_2d=coords_2d.tolist() if coords_2d is not None else [],
        coords_3d=coords_3d.tolist() if coords_3d is not None else None,
        labels=labels.tolist() if isinstance(labels, np.ndarray) else list(labels),
    )


@router.get(
    "/profiles",
    response_model=ClusterProfilesResponse,
    summary="Get cluster personas, radar charts, ANOVA feature importance, and centroids",
)
def get_cluster_profiles() -> ClusterProfilesResponse:
    """
    Returns synthesized business personas, multi-axis radar chart coordinates,
    ANOVA F-statistic feature importance ranking, and cluster centroids.
    """
    cached = _ensure_active_clustering()
    df = state.get_dataset()
    X_prep = cached["X_prep"]
    labels = cached["labels"]
    pipeline = state.get_pipeline()

    feature_names = (
        pipeline.feature_names_
        if pipeline is not None and hasattr(pipeline, "feature_names_") and len(pipeline.feature_names_) > 0
        else [f"feat_{i}" for i in range(X_prep.shape[1])]
    )

    centroids_df = compute_cluster_centroids(df, labels)
    radar_profiles = compute_radar_profiles(df, labels)
    feat_importance = compute_feature_importance(X_prep, labels, feature_names)
    personas_raw = cached.get("personas_raw") or generate_cluster_personas(df, labels)

    personas: List[ClusterPersona] = []
    for p in personas_raw:
        dist_feats = [
            DistinguishingFeature(
                feature=d["feature"],
                cluster_mean=d["cluster_mean"],
                global_mean=d["global_mean"],
                z_score=d["z_score"],
            )
            for d in p.get("top_distinguishing_features", [])
        ]
        personas.append(
            ClusterPersona(
                cluster_id=p["cluster_id"],
                persona_name=p["persona_name"],
                sample_count=p["sample_count"],
                percentage=p["percentage"],
                business_description=p["business_description"],
                marketing_strategy=p["marketing_strategy"],
                top_distinguishing_features=dist_feats,
                radar_metrics=p.get("radar_metrics", {}),
            )
        )

    # Format centroids dict: {cluster_id: {feature: val}}
    centroids_dict: Dict[str, Dict[str, float]] = {}
    for cluster_id, row in centroids_df.iterrows():
        c_key = f"cluster_{cluster_id}" if cluster_id >= 0 else "noise"
        centroids_dict[c_key] = {col: round(float(val), 2) for col, val in row.items()}

    return ClusterProfilesResponse(
        personas=personas,
        radar_profiles=radar_profiles,
        feature_importance=feat_importance,
        centroids=centroids_dict,
    )


@router.get(
    "/silhouette-samples",
    response_model=SilhouetteSamplesResponse,
    summary="Get per-sample silhouette ribbon distribution",
)
def get_silhouette_samples(
    sample_limit: Optional[int] = Query(default=500, description="Max samples to return for ribbon rendering")
) -> SilhouetteSamplesResponse:
    """
    Computes and returns per-sample silhouette coefficients grouped by cluster
    for silhouette ribbon diagnostic visualization.
    """
    cached = _ensure_active_clustering()
    X_prep = cached["X_prep"]
    labels = cached["labels"]

    global_sil, sample_sils = compute_silhouette_score(X_prep, labels)

    samples_list: List[SilhouetteSampleItem] = []
    unique_labels = np.unique(labels)
    per_cluster_means: Dict[str, float] = {}

    for c in unique_labels:
        mask = labels == c
        c_sils = sample_sils[mask]
        c_key = f"cluster_{c}" if c >= 0 else "noise"
        per_cluster_means[c_key] = round(float(np.mean(c_sils)), 4) if len(c_sils) > 0 else 0.0

    # Collect sample points
    n_total = len(sample_sils)
    step = max(1, n_total // sample_limit) if sample_limit and n_total > sample_limit else 1

    for idx in range(0, n_total, step):
        samples_list.append(
            SilhouetteSampleItem(
                sample_index=idx,
                cluster_id=int(labels[idx]),
                silhouette_value=round(float(sample_sils[idx]), 4),
            )
        )

    return SilhouetteSamplesResponse(
        global_score=round(float(global_sil), 4),
        samples=samples_list,
        per_cluster_means=per_cluster_means,
    )


@router.get(
    "/elbow",
    response_model=ElbowResponse,
    summary="Get WCSS inertia curve and Kneedle elbow point",
)
def get_elbow_curve(
    k_min: int = Query(default=2, ge=2, le=5, description="Minimum k"),
    k_max: int = Query(default=10, ge=6, le=15, description="Maximum k"),
) -> ElbowResponse:
    """
    Computes within-cluster sum of squares (inertia) across k in [k_min, k_max]
    and applies the Kneedle algorithm to detect the optimal elbow knee point.
    """
    cached = _ensure_active_clustering()
    X_prep = cached["X_prep"]

    elbow_data = compute_inertia_elbow_curve(X_prep, k_min=k_min, k_max=k_max, random_state=42)

    return ElbowResponse(
        k_values=elbow_data["k_values"],
        inertias=elbow_data["inertias"],
        elbow_k=elbow_data["elbow_k"],
        normalized_differences=elbow_data["normalized_differences"],
    )
