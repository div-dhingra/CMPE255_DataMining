"""
Adversarial Stress Testing & Edge Case Challenge Harness (Tier 5)
Executed by Challenger 1 (teamwork_preview_challenger)

Tests 4 Comprehensive Dimension Tracks:
1. Pathological Data & Clustering Model Stability
2. Autoresearch Hill-Climber Degenerate Parameter Stress
3. FastAPI Pydantic Schemas & Real-Time Inference Boundary Handling
4. Research Synthesis & Benchmark Matrix Edge Robustness
"""

import sys
import os
import time
import traceback
import importlib.util
import numpy as np
import pandas as pd
from pydantic import ValidationError

# Setup backend imports
BACKEND_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend/src"))
if BACKEND_SRC not in sys.path:
    sys.path.insert(0, BACKEND_SRC)

from crisp_dm.data_preparation import (
    DataPreparationPipeline,
    PipelineConfig,
    Imputer,
    OutlierHandler,
    FeatureScaler,
    FeatureEngineer,
)
from crisp_dm.data_understanding import (
    compute_data_understanding_summary,
    compute_hopkins_statistic,
    NUMERIC_FEATURE_COLUMNS,
)
from crisp_dm.models.partitioning import KMeansModel, KMedoidsModel
from crisp_dm.models.density import DBSCANModel, HDBSCANModel
from crisp_dm.models.hierarchical import AgglomerativeModel
from crisp_dm.models.probabilistic import GaussianMixtureModel
from crisp_dm.evaluation import (
    compute_silhouette_score,
    compute_davies_bouldin_index,
    compute_calinski_harabasz_score,
    adjusted_rand_index,
    compute_inertia_elbow_curve,
    compute_subsampling_stability,
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
from autoresearch.search_space import SearchSpace
from autoresearch.objective import (
    CompositeObjective,
    ObjectiveWeights,
    ObjectiveEvaluation,
)
from autoresearch.hill_climber import (
    HillClimbingOptimizer,
    StepLog,
    OptimizationResult,
)
from autoresearch.experiment_logger import (
    ExperimentLogger,
    export_ablation_analysis,
)
from research.literature import (
    LITERATURE_REPOSITORY,
    ALGORITHM_TAXONOMY,
    METRIC_TAXONOMY,
    AUTORESEARCH_PRINCIPLES,
    get_citations,
    get_citation_by_id,
    get_bibtex,
    get_algorithm_taxonomy,
    get_metric_formulations,
    get_autoresearch_principles,
)
from research.benchmark_matrix import (
    BenchmarkRunner,
    BenchmarkModelResult,
    BenchmarkMatrixSummary,
    AblationEntry,
    AblationMatrixSummary,
    run_benchmark_matrix,
    generate_ablation_matrix,
    to_latex_table,
    to_markdown_table,
    to_latex_ablation_table,
    to_markdown_ablation_table,
    to_dict,
    to_json,
)

# Load schemas dynamically without triggering ASGI app import
spec = importlib.util.spec_from_file_location(
    "api_schemas",
    os.path.join(BACKEND_SRC, "api/schemas.py")
)
api_schemas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_schemas)


class AdversarialHarness:
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def run(self, category: str, test_name: str, test_func):
        self.total_tests += 1
        header = f"[{category}] {test_name}"
        print(f"\n--- RUNNING: {header} ---")
        t0 = time.perf_counter()
        try:
            test_func()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.passed_tests += 1
            print(f"✅ PASS: {header} ({elapsed_ms:.2f}ms)")
            self.results.append({
                "category": category,
                "name": test_name,
                "status": "PASSED",
                "elapsed_ms": elapsed_ms,
                "error": None,
            })
        except Exception as ex:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.failed_tests += 1
            err_trace = traceback.format_exc()
            print(f"❌ FAIL: {header} ({elapsed_ms:.2f}ms)\n{err_trace}")
            self.results.append({
                "category": category,
                "name": test_name,
                "status": "FAILED",
                "elapsed_ms": elapsed_ms,
                "error": str(ex),
                "trace": err_trace,
            })


harness = AdversarialHarness()

# ==============================================================================
# 1. PATHOLOGICAL DATA & CLUSTERING MODEL STABILITY
# ==============================================================================

def test_all_zeros_dataset():
    """All-zeros dataset (zero variance across all 10 features, 60 samples)."""
    X = np.zeros((60, 10))
    df = pd.DataFrame(X, columns=[f"F{i}" for i in range(10)])

    # Preprocessing Pipeline
    pipe = DataPreparationPipeline(PipelineConfig(
        imputer_strategy="median",
        outlier_strategy="winsorize",
        scaling_strategy="standard",
        engineer_ratios=False,
    ))
    X_trans = pipe.fit_transform(df)
    assert X_trans.shape == (60, 10)
    assert not np.isnan(X_trans).any(), "NaNs detected in scaled all-zeros"
    assert not np.isinf(X_trans).any(), "Infs detected in scaled all-zeros"

    # All 6 Models
    km = KMeansModel(n_clusters=3, random_state=42)
    labels_km = km.fit_predict(X_trans)
    assert len(labels_km) == 60

    kmed = KMedoidsModel(n_clusters=3, metric="euclidean", random_state=42)
    labels_kmed = kmed.fit_predict(X_trans)
    assert len(labels_kmed) == 60

    for linkage in ["ward", "complete", "average", "single"]:
        agg = AgglomerativeModel(n_clusters=3, linkage=linkage)
        labels_agg = agg.fit_predict(X_trans)
        assert len(labels_agg) == 60

    for cov in ["full", "tied", "diag", "spherical"]:
        gmm = GaussianMixtureModel(n_clusters=3, covariance_type=cov, reg_covar=1e-3, random_state=42)
        labels_gmm = gmm.fit_predict(X_trans)
        assert len(labels_gmm) == 60
        assert not np.isnan(gmm.means_).any()

    # Evaluation metrics
    sil, sample_sils = compute_silhouette_score(X_trans, labels_km)
    assert -1.0 <= sil <= 1.0
    db = compute_davies_bouldin_index(X_trans, labels_km)
    assert db >= 0.0
    ch = compute_calinski_harabasz_score(X_trans, labels_km)
    assert ch >= 0.0

    # Projections
    pca_res = compute_pca_projections(X_trans, n_components=3)
    assert pca_res["coords_2d"].shape == (60, 2)
    assert not np.isnan(pca_res["coords_2d"]).any()

harness.run("Clustering", "All-Zeros Dataset (Zero Variance Matrix)", test_all_zeros_dataset)


def test_collinear_and_singular_covariance():
    """Rank-deficient matrices with perfectly duplicate and linear-combination columns."""
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, size=(60, 2))
    c3 = base[:, 0] * 3.1415
    c4 = base[:, 1] * -2.0 + base[:, 0]
    c5 = np.zeros(60)
    c6 = np.full(60, 100.0)
    X = np.column_stack([base, c3, c4, c5, c6])
    df = pd.DataFrame(X, columns=[f"C{i}" for i in range(6)])

    pipe = DataPreparationPipeline(PipelineConfig(
        imputer_strategy="mean",
        outlier_strategy="iqr",
        scaling_strategy="yeo_johnson",
        engineer_ratios=False,
    ))
    X_trans = pipe.fit_transform(df)
    assert not np.isnan(X_trans).any()
    assert not np.isinf(X_trans).any()

    # GMM Full Covariance Regularization
    gmm = GaussianMixtureModel(n_clusters=3, covariance_type="full", reg_covar=1e-5, random_state=42)
    labels_gmm = gmm.fit_predict(X_trans)
    assert len(labels_gmm) == 60
    proba = gmm.predict_proba(X_trans[:5])
    assert proba.shape == (5, 3)
    assert np.allclose(np.sum(proba, axis=1), 1.0)

    # PCA SVD stability on rank-deficient matrix
    pca_res = compute_pca_projections(X_trans, n_components=3)
    assert len(pca_res["explained_variance_ratio"]) == 3
    assert np.sum(pca_res["explained_variance_ratio"]) <= 1.0001

harness.run("Clustering", "Collinear & Singular Covariance Regularization", test_collinear_and_singular_covariance)


def test_high_dimensions_low_samples():
    """High dimensions (D=200) with few samples (N=12)."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(12, 200))
    df = pd.DataFrame(X)

    pipe = DataPreparationPipeline(PipelineConfig(scaling_strategy="standard", engineer_ratios=False))
    X_prep = pipe.fit_transform(df)
    assert X_prep.shape == (12, 200)

    agg = AgglomerativeModel(n_clusters=3, linkage="ward")
    labels = agg.fit_predict(X_prep)
    assert len(np.unique(labels)) == 3

    pca_res = compute_pca_projections(X_prep, n_components=3)
    assert pca_res["coords_2d"].shape == (12, 2)
    assert pca_res["coords_3d"].shape == (12, 3)

    umap_res = compute_umap_projections(X_prep, n_components=3)
    assert umap_res["coords_2d"].shape == (12, 2)

harness.run("Clustering", "High Dimensions (D=200) Low Samples (N=12)", test_high_dimensions_low_samples)


def test_extreme_floats_and_outlier_clamping():
    """Extreme magnitude values (+/- 1e12, subnormals, massive spikes)."""
    rng = np.random.default_rng(42)
    X = rng.normal(10, 2, size=(80, 5))
    X[0, 0] = 1e12
    X[1, 1] = -1e12
    X[2, 2] = 1e-28
    X[3, 3] = 1e15

    df = pd.DataFrame(X, columns=[f"F{i}" for i in range(5)])

    pipe = DataPreparationPipeline(PipelineConfig(
        outlier_strategy="winsorize",
        winsorize_limits=(0.05, 0.05),
        scaling_strategy="robust",
        engineer_ratios=False,
    ))
    X_scaled = pipe.fit_transform(df)
    assert not np.isnan(X_scaled).any()
    assert not np.isinf(X_scaled).any()
    assert np.max(np.abs(X_scaled)) < 1e6

    km = KMeansModel(n_clusters=4, random_state=42)
    labels = km.fit_predict(X_scaled)
    assert len(labels) == 80
    assert km.inertia_ > 0

harness.run("Clustering", "Extreme Float Magnitude Bounds & Outlier Clamping", test_extreme_floats_and_outlier_clamping)


def test_sample_size_less_than_k_strict_trapping():
    """Strict ValueError when N_samples < K_clusters."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # N=2, K=4
    for model_cls in [KMeansModel, KMedoidsModel, AgglomerativeModel, GaussianMixtureModel]:
        m = model_cls(n_clusters=4)
        try:
            m.fit_predict(X)
            assert False, f"{model_cls.__name__} should have raised ValueError when N < K"
        except ValueError as ve:
            assert "must be >= n_clusters" in str(ve)

harness.run("Clustering", "Strict ValueError on N_samples < K_clusters", test_sample_size_less_than_k_strict_trapping)


def test_density_100_percent_noise_and_single_cluster():
    """DBSCAN with 100% noise (eps=0.0001) and 1 giant cluster (eps=1000.0)."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 100, size=(50, 4))

    # 1. 100% Noise
    dbscan_noise = DBSCANModel(eps=0.0001, min_samples=10)
    labels_noise = dbscan_noise.fit_predict(X)
    assert np.all(labels_noise == -1)
    assert dbscan_noise.n_clusters_ == 0

    eval_res = evaluate_clustering_solution(X, labels_noise, dbscan_noise)
    assert eval_res["n_clusters"] == 0
    assert eval_res["noise_ratio"] == 1.0
    assert eval_res["silhouette_score"] == 0.0
    assert eval_res["davies_bouldin_index"] == 0.0
    assert eval_res["calinski_harabasz_score"] == 0.0

    personas = generate_cluster_personas(pd.DataFrame(X), labels_noise)
    assert len(personas) == 1
    assert personas[0]["cluster_id"] == -1
    assert personas[0]["persona_name"] == "Anomalies / Noise"

    # Prediction on noise model
    pred = dbscan_noise.predict(X[:5])
    assert np.all(pred == -1)
    proba = dbscan_noise.predict_proba(X[:5])
    assert proba.shape == (5, 1)

    # 2. Single Cluster
    dbscan_single = DBSCANModel(eps=1000.0, min_samples=2)
    labels_single = dbscan_single.fit_predict(X)
    assert np.all(labels_single == 0)
    assert dbscan_single.n_clusters_ == 1

    sil, _ = compute_silhouette_score(X, labels_single)
    assert sil == 0.0
    db = compute_davies_bouldin_index(X, labels_single)
    assert db == 0.0

harness.run("Clustering", "Density 100% Noise & Single-Cluster Edge Handling", test_density_100_percent_noise_and_single_cluster)


def test_nan_imputation_adversarial_patterns():
    """All-NaN columns and all-NaN transform queries."""
    X = np.array([
        [np.nan, 1.0, np.nan],
        [np.nan, 2.0, np.nan],
        [np.nan, 3.0, np.nan],
        [np.nan, 4.0, np.nan],
        [np.nan, 5.0, np.nan],
    ])

    imp_med = Imputer(strategy="median")
    X_med = imp_med.fit_transform(X)
    assert not np.isnan(X_med).any()
    assert np.all(X_med[:, 0] == 0.0)
    assert np.all(X_med[:, 2] == 0.0)

    imp_knn = Imputer(strategy="knn", knn_neighbors=2)
    X_knn = imp_knn.fit_transform(X)
    assert not np.isnan(X_knn).any()

    # Query with all NaNs
    all_nan_pt = np.array([[np.nan, np.nan, np.nan]])
    trans_pt = imp_knn.transform(all_nan_pt)
    assert not np.isnan(trans_pt).any()

harness.run("Clustering", "All-NaN Columns & Query Imputation Resilience", test_nan_imputation_adversarial_patterns)


def test_inertia_elbow_and_stability_adversarial():
    """Inertia elbow curve on degenerate datasets & subsampling bootstrap stability."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(40, 5))

    elbow_res = compute_inertia_elbow_curve(X, k_min=2, k_max=6, random_state=42)
    assert len(elbow_res["k_values"]) == 5
    assert 2 <= elbow_res["elbow_k"] <= 6
    assert elbow_res["inertias"][0] >= elbow_res["inertias"][-1]

    # Bootstrap stability
    stab = compute_subsampling_stability(
        model_factory=lambda: KMeansModel(n_clusters=3, random_state=42),
        X=X,
        n_bootstraps=3,
        subsample_ratio=0.7,
        random_state=42,
    )
    assert 0.0 <= stab <= 1.0

harness.run("Clustering", "Inertia Elbow Kneedle & Bootstrap Stability ARI", test_inertia_elbow_and_stability_adversarial)


# ==============================================================================
# 2. AUTORESEARCH HILL-CLIMBER DEGENERATE PARAMETER STRESS
# ==============================================================================

def test_autoresearch_zero_iterations():
    """HillClimber with max_steps=0."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(0, 1, size=(60, 6)), columns=[f"F{i}" for i in range(6)])
    obj = CompositeObjective(df, cache_preprocessing=False)

    opt = HillClimbingOptimizer(objective=obj, target_algorithm="kmeans", random_state=42)
    res0 = opt.run(max_steps=0)
    assert res0.total_steps == 0
    assert len(res0.history) == 0
    assert res0.fitness_improvement == 0.0
    assert res0.best_fitness == res0.initial_fitness

harness.run("Autoresearch", "Zero Iteration Execution (max_steps=0)", test_autoresearch_zero_iterations)


def test_autoresearch_extreme_temperatures():
    """Extreme temperatures: T=10000.0 (high entropy) vs T=1e-8 (near-zero/greedy)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(0, 1, size=(60, 6)), columns=[f"F{i}" for i in range(6)])
    obj = CompositeObjective(df, cache_preprocessing=False)

    # 1. Hot temperature
    opt_hot = HillClimbingOptimizer(
        objective=obj,
        initial_temperature=10000.0,
        cooling_rate=0.99,
        target_algorithm="kmeans",
        patience=20,
        random_state=42,
    )
    res_hot = opt_hot.run(max_steps=10)
    assert res_hot.total_steps == 10
    accept_hot = sum(1 for s in res_hot.history if s.accepted)
    assert accept_hot >= 7

    # 2. Frozen temperature
    opt_cold = HillClimbingOptimizer(
        objective=obj,
        initial_temperature=1e-8,
        min_temperature=1e-12,
        cooling_rate=0.5,
        target_algorithm="kmeans",
        patience=20,
        random_state=42,
    )
    res_cold = opt_cold.run(max_steps=10)
    assert res_cold.total_steps == 10
    for s in res_cold.history:
        if s.accepted and not s.restart:
            assert s.delta_fitness >= -1e-6

harness.run("Autoresearch", "Extreme Temperatures (T=10000 vs T=1e-8)", test_autoresearch_extreme_temperatures)


def test_autoresearch_fast_restart_triggers():
    """Patience=1: Every single non-improving step triggers a Global Random Restart."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(0, 1, size=(60, 6)), columns=[f"F{i}" for i in range(6)])
    obj = CompositeObjective(df, cache_preprocessing=False)

    opt = HillClimbingOptimizer(
        objective=obj,
        target_algorithm="kmeans",
        patience=1,
        cooling_rate=0.9,
        random_state=42,
    )
    res = opt.run(max_steps=15)
    assert res.total_steps == 15
    assert res.restart_count > 0
    restarts = [s for s in res.history if s.restart]
    assert len(restarts) == res.restart_count

harness.run("Autoresearch", "Fast Stagnation & Random Restarts (patience=1)", test_autoresearch_fast_restart_triggers)


def test_autoresearch_invalid_objective_weights():
    """Objective weights auto-normalization and zero weights handling."""
    w1 = ObjectiveWeights(silhouette=10.0, davies_bouldin=5.0, calinski_harabasz=3.0, stability=2.0)
    assert np.isclose(w1.silhouette + w1.davies_bouldin + w1.calinski_harabasz + w1.stability, 1.0)
    assert np.isclose(w1.silhouette, 0.5)

    w_zero = ObjectiveWeights(silhouette=0.0, davies_bouldin=0.0, calinski_harabasz=0.0, stability=0.0)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(0, 1, size=(50, 4)), columns=[f"F{i}" for i in range(4)])
    obj_zero = CompositeObjective(df, weights=w_zero)
    eval_res = obj_zero.evaluate({"algorithm": "kmeans", "hyperparameters": {"n_clusters": 3}})
    assert 0.0 <= eval_res.fitness <= 1.0 or eval_res.fitness == -1.0

harness.run("Autoresearch", "Objective Weights Auto-Normalization & Zero Weights", test_autoresearch_invalid_objective_weights)


def test_autoresearch_pathological_candidates_and_error_traps():
    """Graceful -1.0 penalty and error logging for invalid candidates."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(0, 1, size=(50, 4)), columns=[f"F{i}" for i in range(4)])
    obj = CompositeObjective(df)

    # 1. Non-existent algorithm
    e1 = obj.evaluate({"algorithm": "quantum_neural_clustering", "hyperparameters": {}})
    assert e1.fitness == -1.0
    assert "Unknown clustering algorithm" in str(e1.error)

    # 2. k=1 cluster (degenerate)
    e2 = obj.evaluate({"algorithm": "kmeans", "hyperparameters": {"n_clusters": 1}})
    assert e2.fitness == -1.0
    assert e2.error is not None

    # 3. DBSCAN 100% noise
    e3 = obj.evaluate({"algorithm": "dbscan", "hyperparameters": {"eps": 0.1, "min_samples": 500}})
    assert e3.fitness == -1.0
    assert "Degenerate clustering" in str(e3.error)

harness.run("Autoresearch", "Pathological Candidate Penalty Trapping (-1.0)", test_autoresearch_pathological_candidates_and_error_traps)


def test_autoresearch_experiment_logger_and_ablation_export():
    """Experiment logger ablation analysis on short and multi-step runs."""
    logger = ExperimentLogger(run_id="test_exp_001")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(0, 1, size=(50, 4)), columns=[f"F{i}" for i in range(4)])
    obj = CompositeObjective(df)
    opt = HillClimbingOptimizer(objective=obj, target_algorithm="kmeans", random_state=42)

    res = opt.run(max_steps=5, callback=lambda step: logger.log_step(step))
    assert len(logger.logs) == 5

    ablations = export_ablation_analysis(logger.logs)
    assert "parameter_stage_breakdown" in ablations or "error" not in ablations

harness.run("Autoresearch", "Experiment Logger & Ablation Decomposition", test_autoresearch_experiment_logger_and_ablation_export)


# ==============================================================================
# 3. FASTAPI PYDANTIC SCHEMAS & INFERENCE BOUNDARY HANDLING
# ==============================================================================

def test_pydantic_validation_error_rejections():
    """Verify strict Pydantic V2 schema validations and error rejections."""
    # 1. SingleInferenceRequest with invalid non-dict
    try:
        api_schemas.SingleInferenceRequest(features="invalid_string")
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass

    # 2. BatchInferenceRequest with invalid non-list
    try:
        api_schemas.BatchInferenceRequest(customers="not_a_list")
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass

    # 3. SingleInferenceRequest with non-numeric value in dictionary
    try:
        api_schemas.SingleInferenceRequest(features={"BALANCE": "not_a_number_string"})
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass

    # 4. Valid single request
    req = api_schemas.SingleInferenceRequest(features={"BALANCE": 1500.0, "PURCHASES": 200.0})
    assert req.features["BALANCE"] == 1500.0

harness.run("API Schemas", "Pydantic V2 Strict Type & Literal Rejections", test_pydantic_validation_error_rejections)


def test_inference_pipeline_sparse_and_extreme_inputs():
    """Inference scoring with 95% missing features, extreme floats, and batch sizing."""
    rng = np.random.default_rng(42)
    records = []
    for i in range(100):
        records.append({
            "CUST_ID": f"C{1000+i}",
            "BALANCE": float(rng.uniform(100, 5000)),
            "BALANCE_FREQUENCY": float(rng.uniform(0.5, 1.0)),
            "PURCHASES": float(rng.uniform(50, 3000)),
            "ONEOFF_PURCHASES": float(rng.uniform(0, 1500)),
            "INSTALLMENTS_PURCHASES": float(rng.uniform(0, 1500)),
            "CASH_ADVANCE": float(rng.uniform(0, 1000)),
            "PURCHASES_FREQUENCY": float(rng.uniform(0.1, 1.0)),
            "ONEOFF_PURCHASES_FREQUENCY": float(rng.uniform(0.0, 0.8)),
            "PURCHASES_INSTALLMENTS_FREQUENCY": float(rng.uniform(0.0, 0.8)),
            "CASH_ADVANCE_FREQUENCY": float(rng.uniform(0.0, 0.5)),
            "CASH_ADVANCE_TRX": int(rng.integers(0, 10)),
            "PURCHASES_TRX": int(rng.integers(1, 30)),
            "CREDIT_LIMIT": float(rng.uniform(1000, 15000)),
            "PAYMENTS": float(rng.uniform(100, 4000)),
            "MINIMUM_PAYMENTS": float(rng.uniform(50, 500)),
            "PRC_FULL_PAYMENT": float(rng.uniform(0.0, 1.0)),
            "TENURE": int(rng.integers(6, 13)),
        })
    df_raw = pd.DataFrame(records)

    pipe = DataPreparationPipeline(PipelineConfig(
        imputer_strategy="median",
        outlier_strategy="winsorize",
        scaling_strategy="yeo_johnson",
        engineer_ratios=True,
    ))
    X_prep = pipe.fit_transform(df_raw)
    km = KMeansModel(n_clusters=4, random_state=42)
    labels = km.fit_predict(X_prep)

    # 1. Sparse single customer (only BALANCE provided)
    sparse_customer = {"BALANCE": 3500.0}
    input_row = {}
    for col in NUMERIC_FEATURE_COLUMNS:
        if col in sparse_customer:
            input_row[col] = sparse_customer[col]
        else:
            input_row[col] = float(df_raw[col].median()) if col in df_raw.columns else 0.0

    df_sparse = pd.DataFrame([input_row])
    X_single_trans = pipe.transform(df_sparse)
    pred_single = km.predict(X_single_trans)
    assert len(pred_single) == 1
    proba_single = km.predict_proba(X_single_trans)
    assert proba_single.shape == (1, 4)
    assert np.allclose(np.sum(proba_single, axis=1), 1.0)

    # 2. Batch inference on 250 heterogeneous customer vectors
    batch_rows = []
    for i in range(250):
        row = {}
        for col in NUMERIC_FEATURE_COLUMNS:
            row[col] = float(rng.uniform(0, 10000))
        batch_rows.append(row)

    df_batch = pd.DataFrame(batch_rows)
    X_batch_trans = pipe.transform(df_batch)
    batch_preds = km.predict(X_batch_trans)
    assert len(batch_preds) == 250
    batch_proba = km.predict_proba(X_batch_trans)
    assert batch_proba.shape == (250, 4)
    assert np.allclose(np.sum(batch_proba, axis=1), 1.0)

harness.run("API Schemas", "Sparse Inference & 250-Sample Bulk Scoring", test_inference_pipeline_sparse_and_extreme_inputs)


def test_data_understanding_summary_resilience():
    """compute_data_understanding_summary with constant and zero-variance columns."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "CUST_ID": [f"C{i}" for i in range(40)],
        "BALANCE": rng.normal(1000, 100, 40),
        "PURCHASES": np.zeros(40),  # zero variance
        "CREDIT_LIMIT": np.full(40, 5000.0),  # constant
        "PAYMENTS": [np.nan if i % 5 == 0 else 500.0 for i in range(40)],  # partial NaNs
    })
    summary = compute_data_understanding_summary(df)
    assert summary["n_samples"] == 40
    assert summary["n_columns"] == 5
    assert "missing_summary" in summary
    assert "hopkins_statistic" in summary
    assert 0.0 <= summary["hopkins_statistic"] <= 1.0

harness.run("Data & EDA", "Data Understanding with Constant & Missing Columns", test_data_understanding_summary_resilience)


# ==============================================================================
# 4. RESEARCH SYNTHESIS & BENCHMARK MATRIX EDGE ROBUSTNESS
# ==============================================================================

def test_research_literature_query_and_fallbacks():
    """Literature queries with matching, non-matching, and topic filters."""
    # 1. Full literature
    all_lit = get_citations()
    assert len(all_lit) >= 9

    # 2. Filter by valid keyword (e.g. 'validation')
    val_lit = get_citations(topic="validation")
    assert len(val_lit) >= 3

    # 3. Query with non-existent topic (must return empty list, not crash)
    none_lit = get_citations(topic="quantum_superposition_clustering")
    assert len(none_lit) == 0

    # 4. Specific citation by key
    c_rouss = get_citation_by_id("rousseeuw1987")
    assert c_rouss is not None
    assert "Silhouettes" in c_rouss["title"]

    # 5. Non-existent citation key
    c_none = get_citation_by_id("nonexistent_key_xyz")
    assert c_none is None

    # 6. BibTeX generation
    bib = get_bibtex("rousseeuw1987")
    assert "@article" in bib or "@inproceedings" in bib

harness.run("Research", "Literature Citations Query & Fallback Protection", test_research_literature_query_and_fallbacks)


def test_benchmark_matrix_and_latex_markdown_exporters():
    """Benchmark matrix generation and publication export formatting."""
    rng = np.random.default_rng(42)
    records = []
    for i in range(50):
        records.append({
            "CUST_ID": f"C{1000+i}",
            "BALANCE": float(rng.uniform(100, 5000)),
            "BALANCE_FREQUENCY": 1.0,
            "PURCHASES": float(rng.uniform(50, 3000)),
            "ONEOFF_PURCHASES": float(rng.uniform(0, 1500)),
            "INSTALLMENTS_PURCHASES": float(rng.uniform(0, 1500)),
            "CASH_ADVANCE": 0.0,
            "PURCHASES_FREQUENCY": 0.8,
            "ONEOFF_PURCHASES_FREQUENCY": 0.4,
            "PURCHASES_INSTALLMENTS_FREQUENCY": 0.6,
            "CASH_ADVANCE_FREQUENCY": 0.0,
            "CASH_ADVANCE_TRX": 0,
            "PURCHASES_TRX": 15,
            "CREDIT_LIMIT": 5000.0,
            "PAYMENTS": 800.0,
            "MINIMUM_PAYMENTS": 200.0,
            "PRC_FULL_PAYMENT": 0.25,
            "TENURE": 12,
        })
    df = pd.DataFrame(records)

    runner = BenchmarkRunner(random_state=42)
    bm_summary = runner.run_benchmark(data=df, n_bootstraps=2)
    assert len(bm_summary.results) == 6
    for item in bm_summary.results:
        assert item.model_name in ["K-Means", "K-Medoids", "DBSCAN", "HDBSCAN", "Agglomerative", "GMM"]
        assert 0.0 <= item.composite_score <= 1.0

    # LaTeX Table Export
    latex_out = to_latex_table(bm_summary)
    assert r"\begin{table}" in latex_out
    assert r"\end{table}" in latex_out
    assert "K-Means" in latex_out

    # Markdown Table Export
    md_out = to_markdown_table(bm_summary)
    assert "| Model | Paradigm |" in md_out
    assert "| K-Means | Partitioning |" in md_out

    # Ablation Matrix Summary
    opt_models = runner.create_optimized_models()
    opt_summary = runner.run_benchmark(data=df, model_factories=opt_models, n_bootstraps=2)
    ab_summary = generate_ablation_matrix(bm_summary, opt_summary)
    assert len(ab_summary.ablation_entries) == 6

    latex_ab = to_latex_ablation_table(ab_summary)
    assert r"\begin{table}" in latex_ab

    md_ab = to_markdown_ablation_table(ab_summary)
    assert "| Model | Base Sil | Opt Sil |" in md_ab

harness.run("Research", "Benchmark Matrix Generation & Publication Exporters (LaTeX/MD)", test_benchmark_matrix_and_latex_markdown_exporters)


# ==============================================================================
# SUMMARY & FINAL VERDICT
# ==============================================================================

print("\n" + "=" * 80)
print(f"EMPIRICAL ADVERSARIAL STRESS TESTING SUMMARY:")
print(f"Total Tests Executed: {harness.total_tests}")
print(f"Passed: {harness.passed_tests}")
print(f"Failed: {harness.failed_tests}")
print(f"Success Rate: {harness.passed_tests / harness.total_tests * 100:.1f}%")
print("=" * 80)

if harness.failed_tests > 0:
    print(f"\n❌ VERDICT: DISPROVED ({harness.failed_tests} failure(s) observed)")
    sys.exit(1)
else:
    print("\n✅ VERDICT: CONFIRMED (100% of empirical adversarial stress challenges passed cleanly)")
    sys.exit(0)
