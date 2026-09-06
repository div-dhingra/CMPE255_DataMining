"""
Comprehensive Unit Test Suite for CRISP-DM Clustering Pipeline (Milestone 1)
Tests:
- Data Ingestion, Synthetic Generation, Hopkins Statistic & EDA Summary
- Automated Preprocessing (Imputation, Outliers, Scalers, Feature Engineering, Pipeline)
- Multi-Paradigm Clustering Models (K-Means, K-Medoids, DBSCAN, HDBSCAN, Agglomerative, GMM)
- Internal & External Evaluation Metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, Kneedle Elbow, Stability ARI)
- Projections (PCA 2D/3D, UMAP 2D/3D, t-SNE 2D)
- Personas & Profiling (Centroids, Radars, Feature Importance, Business Personas)
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure backend/src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from crisp_dm.data_understanding import (
    CREDIT_CARD_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    generate_synthetic_credit_card_data,
    load_credit_card_data,
    compute_hopkins_statistic,
    compute_data_understanding_summary,
)
from crisp_dm.data_preparation import (
    PipelineConfig,
    Imputer,
    OutlierHandler,
    FeatureScaler,
    FeatureEngineer,
    DataPreparationPipeline,
)
from crisp_dm.models.partitioning import KMeansModel, KMedoidsModel
from crisp_dm.models.density import DBSCANModel, HDBSCANModel
from crisp_dm.models.hierarchical import AgglomerativeModel
from crisp_dm.models.probabilistic import GaussianMixtureModel
from crisp_dm.evaluation import (
    compute_silhouette_score,
    compute_davies_bouldin_index,
    compute_calinski_harabasz_score,
    compute_inertia_elbow_curve,
    compute_subsampling_stability,
    adjusted_rand_index,
    evaluate_clustering_solution,
)
from crisp_dm.projections import (
    compute_pca_projections,
    compute_umap_projections,
    compute_tsne_projections,
    project_coordinates,
)
from crisp_dm.profiling import (
    compute_cluster_centroids,
    compute_radar_profiles,
    compute_feature_importance,
    generate_cluster_personas,
)


# ============================================================================
# 1. Data Understanding & Ingestion Tests
# ============================================================================

def test_synthetic_data_generation_schema():
    df = generate_synthetic_credit_card_data(n_samples=200, random_state=42)
    assert len(df) == 200
    assert list(df.columns) == CREDIT_CARD_COLUMNS
    assert df["CUST_ID"].iloc[0].startswith("C1")
    # Check missingness injected
    assert df["MINIMUM_PAYMENTS"].isna().sum() > 0
    # Check numerical validity
    assert (df["BALANCE"] >= 0).all()
    assert (df["PURCHASES"] >= 0).all()
    assert (df["TENURE"] >= 6).all() and (df["TENURE"] <= 12).all()


def test_load_credit_card_data_fallback():
    df = load_credit_card_data(filepath="non_existent_file.csv", fallback_to_synthetic=True, n_synthetic=150)
    assert len(df) == 150
    assert "BALANCE" in df.columns


def test_hopkins_statistic_clustering_tendency():
    rng = np.random.default_rng(42)
    # 1. Highly clustered data (3 separated Gaussian blobs)
    blob1 = rng.normal(loc=-10, scale=0.5, size=(100, 4))
    blob2 = rng.normal(loc=0, scale=0.5, size=(100, 4))
    blob3 = rng.normal(loc=10, scale=0.5, size=(100, 4))
    clustered_data = np.vstack([blob1, blob2, blob3])
    h_clustered = compute_hopkins_statistic(clustered_data, m=30, random_state=42)
    assert h_clustered > 0.70  # Should indicate strong clustering tendency

    # 2. Uniform random data
    uniform_data = rng.uniform(low=0, high=1, size=(300, 4))
    h_uniform = compute_hopkins_statistic(uniform_data, m=30, random_state=42)
    assert 0.30 <= h_uniform <= 0.70  # Near 0.5 for uniform space


def test_data_understanding_summary_metrics():
    df = generate_synthetic_credit_card_data(n_samples=100, random_state=42)
    summary = compute_data_understanding_summary(df)
    assert summary["n_samples"] == 100
    assert summary["n_columns"] == 18
    assert "BALANCE" in summary["descriptive_statistics"]
    assert "mean" in summary["descriptive_statistics"]["BALANCE"]
    assert "skewness" in summary["descriptive_statistics"]["BALANCE"]
    assert "correlation_pearson" in summary
    assert "correlation_spearman" in summary
    assert 0.0 <= summary["hopkins_statistic"] <= 1.0


# ============================================================================
# 2. Data Preparation & Preprocessing Tests
# ============================================================================

@pytest.mark.parametrize("strategy", ["median", "mean", "knn", "mice"])
def test_imputation_strategies(strategy):
    X = np.array([
        [1.0, 2.0, np.nan],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
        [np.nan, 11.0, 12.0],
        [13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0],
    ])
    imputer = Imputer(strategy=strategy, knn_neighbors=2, mice_max_iter=5)
    X_imputed = imputer.fit_transform(X)
    assert not np.isnan(X_imputed).any()
    assert X_imputed.shape == X.shape


@pytest.mark.parametrize("strategy", ["winsorize", "iqr", "isolation_forest"])
def test_outlier_handlers(strategy):
    rng = np.random.default_rng(42)
    X = rng.normal(loc=0, scale=1, size=(200, 4))
    # Inject severe outliers
    X[0, :] = 100.0
    X[1, :] = -100.0

    handler = OutlierHandler(strategy=strategy, contamination=0.05, n_trees=20)
    X_capped = handler.fit_transform(X)
    assert X_capped.shape == X.shape
    # Check that extreme values were attenuated or clipped
    assert np.max(X_capped) < 90.0
    assert np.min(X_capped) > -90.0

    X_filt, inlier_mask = handler.filter(X)
    assert len(X_filt) <= len(X)
    assert inlier_mask.dtype == bool


@pytest.mark.parametrize("strategy", ["yeo_johnson", "standard", "robust", "minmax"])
def test_feature_scalers(strategy):
    rng = np.random.default_rng(42)
    # Heavy-tailed positive data
    X = rng.lognormal(mean=2.0, sigma=1.5, size=(100, 3))
    scaler = FeatureScaler(strategy=strategy)
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == X.shape
    assert not np.isnan(X_scaled).any()

    if strategy == "standard":
        assert np.allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-5)
        assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-5)
    elif strategy == "minmax":
        assert np.all(X_scaled >= -1e-5) and np.all(X_scaled <= 1.0 + 1e-5)


def test_feature_engineering_ratios():
    df = generate_synthetic_credit_card_data(n_samples=50, random_state=42)
    df_eng = FeatureEngineer.derive_ratios(df)
    assert "UTILIZATION_RATIO" in df_eng.columns
    assert "PAYMENT_MIN_PAYMENT_RATIO" in df_eng.columns
    assert "ONEOFF_PURCHASE_RATIO" in df_eng.columns
    assert "INSTALLMENT_PURCHASE_RATIO" in df_eng.columns
    assert "CASH_ADVANCE_RATIO" in df_eng.columns
    assert "PURCHASE_TRX_VELOCITY" in df_eng.columns
    assert (df_eng["UTILIZATION_RATIO"] >= 0).all()


def test_data_preparation_pipeline_end_to_end():
    df_train = generate_synthetic_credit_card_data(n_samples=150, random_state=42)
    df_test = generate_synthetic_credit_card_data(n_samples=50, random_state=99)

    config = PipelineConfig(
        imputer_strategy="median",
        outlier_strategy="winsorize",
        scaling_strategy="yeo_johnson",
        engineer_ratios=True,
    )
    pipeline = DataPreparationPipeline(config=config)
    X_train_scaled = pipeline.fit_transform(df_train)

    assert X_train_scaled.shape[0] == 150
    assert X_train_scaled.shape[1] > 17  # Includes engineered ratios
    assert not np.isnan(X_train_scaled).any()

    # Transform test set using fitted pipeline statistics
    X_test_scaled = pipeline.transform(df_test)
    assert X_test_scaled.shape[0] == 50
    assert X_test_scaled.shape[1] == X_train_scaled.shape[1]
    assert not np.isnan(X_test_scaled).any()


# ============================================================================
# 3. Multi-Paradigm Clustering Models Tests (6 Models across 4 Paradigms)
# ============================================================================

@pytest.fixture
def sample_clustered_data():
    """Generates 3 well-separated clusters in 4D space."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[-5.0, -5.0, -5.0, -5.0], scale=0.8, size=(60, 4))
    c2 = rng.normal(loc=[0.0, 0.0, 0.0, 0.0], scale=0.8, size=(60, 4))
    c3 = rng.normal(loc=[5.0, 5.0, 5.0, 5.0], scale=0.8, size=(60, 4))
    X = np.vstack([c1, c2, c3])
    return X


def test_kmeans_model(sample_clustered_data):
    model = KMeansModel(n_clusters=3, init="k-means++", n_init=5, random_state=42)
    labels = model.fit_predict(sample_clustered_data)

    assert len(labels) == len(sample_clustered_data)
    assert len(np.unique(labels)) == 3
    assert model.centroids_.shape == (3, 4)
    assert model.inertia_ > 0.0

    # Predict
    test_pt = np.array([[5.1, 4.9, 5.0, 5.2]])
    pred = model.predict(test_pt)
    assert pred.shape == (1,)

    # Predict Proba
    proba = model.predict_proba(test_pt)
    assert proba.shape == (1, 3)
    assert np.isclose(np.sum(proba), 1.0)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_kmedoids_model(sample_clustered_data, metric):
    model = KMedoidsModel(n_clusters=3, metric=metric, method="fasterpam", random_state=42)
    labels = model.fit_predict(sample_clustered_data)

    assert len(labels) == len(sample_clustered_data)
    assert len(np.unique(labels)) == 3
    assert model.centroids_.shape == (3, 4)
    assert len(model.medoid_indices_) == 3

    # Predict and Proba
    test_pt = np.array([[-4.8, -5.1, -4.9, -5.2]])
    pred = model.predict(test_pt)
    proba = model.predict_proba(test_pt)
    assert pred.shape == (1,)
    assert np.isclose(np.sum(proba), 1.0)


def test_dbscan_model(sample_clustered_data):
    model = DBSCANModel(eps=2.0, min_samples=5, metric="euclidean")
    labels = model.fit_predict(sample_clustered_data)

    assert len(labels) == len(sample_clustered_data)
    assert model.n_clusters_ >= 2
    assert model.core_sample_indices_ is not None

    test_pt = np.array([[0.1, -0.1, 0.0, 0.2]])
    pred = model.predict(test_pt)
    proba = model.predict_proba(test_pt)
    assert pred.shape == (1,)
    assert proba.shape[0] == 1


def test_hdbscan_model(sample_clustered_data):
    model = HDBSCANModel(min_cluster_size=10, min_samples=3)
    labels = model.fit_predict(sample_clustered_data)

    assert len(labels) == len(sample_clustered_data)
    assert model.n_clusters_ >= 2
    assert model.probabilities_ is not None
    assert model.outlier_scores_ is not None
    assert (model.probabilities_ >= 0).all() and (model.probabilities_ <= 1.0).all()

    test_pt = np.array([[5.0, 5.0, 5.0, 5.0]])
    pred = model.predict(test_pt)
    proba = model.predict_proba(test_pt)
    assert pred.shape == (1,)
    assert np.isclose(np.sum(proba), 1.0)


@pytest.mark.parametrize("linkage", ["ward", "complete", "average"])
def test_agglomerative_model(sample_clustered_data, linkage):
    metric = "euclidean"
    model = AgglomerativeModel(n_clusters=3, linkage=linkage, metric=metric)
    labels = model.fit_predict(sample_clustered_data)

    assert len(labels) == len(sample_clustered_data)
    assert len(np.unique(labels)) == 3
    assert model.centroids_.shape == (3, 4)

    test_pt = np.array([[0.0, 0.0, 0.0, 0.0]])
    pred = model.predict(test_pt)
    proba = model.predict_proba(test_pt)
    assert pred.shape == (1,)
    assert np.isclose(np.sum(proba), 1.0)


@pytest.mark.parametrize("cov_type", ["full", "tied", "diag", "spherical"])
def test_gmm_model(sample_clustered_data, cov_type):
    model = GaussianMixtureModel(n_clusters=3, covariance_type=cov_type, n_init=3, random_state=42)
    labels = model.fit_predict(sample_clustered_data)

    assert len(labels) == len(sample_clustered_data)
    assert len(np.unique(labels)) == 3
    assert model.means_.shape == (3, 4)
    assert model.bic_ is not None
    assert model.aic_ is not None
    assert model.log_likelihood_ is not None

    test_pt = np.array([[5.0, 5.0, 5.0, 5.0]])
    pred = model.predict(test_pt)
    proba = model.predict_proba(test_pt)
    assert pred.shape == (1,)
    assert np.isclose(np.sum(proba), 1.0)


# ============================================================================
# 4. Evaluation Metrics Tests
# ============================================================================

def test_internal_validation_metrics(sample_clustered_data):
    model = KMeansModel(n_clusters=3, random_state=42)
    labels = model.fit_predict(sample_clustered_data)

    sil_score, sample_sils = compute_silhouette_score(sample_clustered_data, labels)
    db_score = compute_davies_bouldin_index(sample_clustered_data, labels)
    ch_score = compute_calinski_harabasz_score(sample_clustered_data, labels)

    assert 0.5 <= sil_score <= 1.0  # Well separated clusters have high silhouette
    assert len(sample_sils) == len(sample_clustered_data)
    assert 0.0 <= db_score <= 1.5   # Low DB indicates tight, separated clusters
    assert ch_score > 50.0          # High CH indicates high between-cluster dispersion


def test_adjusted_rand_index_exactness():
    # Identical labelings
    labels_a = np.array([0, 0, 1, 1, 2, 2])
    labels_b = np.array([1, 1, 2, 2, 0, 0])  # Permutation of labels_a
    ari_identical = adjusted_rand_index(labels_a, labels_b)
    assert np.isclose(ari_identical, 1.0)

    # Independent random labelings
    labels_c = np.array([0, 1, 0, 1, 0, 1])
    ari_diff = adjusted_rand_index(labels_a, labels_c)
    assert ari_diff < 0.5


def test_kneedle_elbow_detection(sample_clustered_data):
    res = compute_inertia_elbow_curve(sample_clustered_data, k_min=2, k_max=6, random_state=42)
    assert res["k_values"] == [2, 3, 4, 5, 6]
    assert len(res["inertias"]) == 5
    assert res["inertias"][0] > res["inertias"][-1]  # Monotonically decreasing
    assert res["elbow_k"] == 3  # Ground truth cluster count is 3


def test_subsampling_bootstrap_stability(sample_clustered_data):
    def model_factory():
        return KMeansModel(n_clusters=3, n_init=3, random_state=42)

    stability = compute_subsampling_stability(
        model_factory, sample_clustered_data, n_bootstraps=5, subsample_ratio=0.8
    )
    assert 0.70 <= stability <= 1.0  # High stability on well-separated clusters


def test_evaluate_clustering_solution_summary(sample_clustered_data):
    model = KMeansModel(n_clusters=3, random_state=42)
    labels = model.fit_predict(sample_clustered_data)
    eval_dict = evaluate_clustering_solution(sample_clustered_data, labels, model=model)

    assert "silhouette_score" in eval_dict
    assert "davies_bouldin_index" in eval_dict
    assert "calinski_harabasz_score" in eval_dict
    assert "inertia" in eval_dict
    assert eval_dict["n_clusters"] == 3
    assert eval_dict["noise_points_count"] == 0


# ============================================================================
# 5. Projections & Dimensionality Reduction Tests
# ============================================================================

def test_pca_projections(sample_clustered_data):
    pca_res = compute_pca_projections(sample_clustered_data, n_components=3)
    assert pca_res["coords_2d"].shape == (len(sample_clustered_data), 2)
    assert pca_res["coords_3d"].shape == (len(sample_clustered_data), 3)
    assert len(pca_res["explained_variance_ratio"]) == 3
    assert np.isclose(np.sum(pca_res["explained_variance_ratio"]), 1.0, atol=0.2)


def test_umap_projections(sample_clustered_data):
    umap_res = compute_umap_projections(sample_clustered_data, n_components=3, n_neighbors=10)
    assert umap_res["coords_2d"].shape == (len(sample_clustered_data), 2)
    assert umap_res["coords_3d"].shape == (len(sample_clustered_data), 3)


def test_tsne_projections(sample_clustered_data):
    tsne_res = compute_tsne_projections(sample_clustered_data, n_components=2, perplexity=15.0, n_iter=100)
    assert tsne_res["coords_2d"].shape == (len(sample_clustered_data), 2)


def test_project_coordinates_dispatcher(sample_clustered_data):
    for method in ["pca", "umap", "tsne"]:
        res = project_coordinates(sample_clustered_data, method=method)
        assert "coords_2d" in res
        assert res["coords_2d"].shape == (len(sample_clustered_data), 2)


# ============================================================================
# 6. Personas & Profiling Tests
# ============================================================================

def test_profiling_and_personas():
    df = generate_synthetic_credit_card_data(n_samples=120, random_state=42)
    pipeline = DataPreparationPipeline(config=PipelineConfig(scaling_strategy="standard"))
    X_scaled = pipeline.fit_transform(df)

    model = KMeansModel(n_clusters=4, random_state=42)
    labels = model.fit_predict(X_scaled)

    # 1. Centroids
    centroids_df = compute_cluster_centroids(df, labels)
    assert centroids_df.shape[0] == 4

    # 2. Radar profiles
    radar = compute_radar_profiles(df, labels)
    assert len(radar) == 4
    for c_key, metrics in radar.items():
        assert "BALANCE" in metrics
        assert 0.0 <= metrics["BALANCE"] <= 1.0

    # 3. Feature importance
    importance = compute_feature_importance(X_scaled, labels, pipeline.feature_names_)
    assert len(importance) == len(pipeline.feature_names_)
    assert np.isclose(sum(importance.values()), 1.0, atol=1e-2)

    # 4. Personas
    personas = generate_cluster_personas(df, labels, feature_names=pipeline.feature_names_)
    assert len(personas) == 4
    for p in personas:
        assert "cluster_id" in p
        assert "persona_name" in p
        assert "business_description" in p
        assert "marketing_strategy" in p
        assert "top_distinguishing_features" in p
        assert len(p["top_distinguishing_features"]) > 0
