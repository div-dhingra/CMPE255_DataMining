"""
Comprehensive Unit & Integration Test Suite for High-Performance FastAPI Backend.
Tests:
- F-API-01: Health check, OpenAPI Swagger schema (/docs, /redoc, /api/v1/openapi.json), CORS
- F-API-02: Data understanding (/data/summary, /data/correlations, /data/distributions, /data/sample, /data/upload)
- F-API-03: Clustering execution across all 6 models (/cluster/run), projections (PCA/UMAP/t-SNE), profiles, personas, silhouette samples, elbow
- F-API-04: Autoresearch optimization lifecycle (/autoresearch/start, /status, /stream SSE, /pause, /stop, /leaderboard, /ablations)
- F-API-05: Real-time single and batch customer inference (/inference/predict, /inference/batch)
- Research Synthesis: Literature repository, benchmark matrix, ablations, LaTeX/Markdown export (/research/*)
- Error handling and boundary conditions (400, 404, 405, 422)
"""

import io
import json
import time
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.state import state
from crisp_dm.data_understanding import NUMERIC_FEATURE_COLUMNS


@pytest.fixture
def client():
    """Returns a TestClient instance bound to the FastAPI application."""
    return TestClient(app)


# ============================================================================
# 1. System Health & Documentation Tests (F-API-01)
# ============================================================================

def test_health_check_endpoint(client):
    """Verifies /api/v1/health and /health root alias return valid state metadata."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert data["dataset_loaded"] is True
    assert data["dataset_rows"] >= 100

    # Alias /health
    alias_resp = client.get("/health")
    assert alias_resp.status_code == 200
    assert alias_resp.json()["status"] == "ok"


def test_root_welcome_endpoint(client):
    """Verifies root endpoint / returns API metadata and documentation links."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert data["docs"] == "/docs"
    assert data["health"] == "/api/v1/health"


def test_openapi_documentation_endpoints(client):
    """Verifies /docs, /redoc, and /api/v1/openapi.json specifications."""
    # OpenAPI JSON
    resp_json = client.get("/api/v1/openapi.json")
    assert resp_json.status_code == 200
    schema = resp_json.json()
    assert schema["openapi"] == "3.1.0"
    assert "paths" in schema
    assert "/api/v1/health" in schema["paths"]
    assert "/api/v1/cluster/run" in schema["paths"]
    assert "/api/v1/autoresearch/start" in schema["paths"]

    # Swagger UI HTML
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200
    assert "SwaggerUIBundle" in docs_resp.text

    # ReDoc HTML
    redoc_resp = client.get("/redoc")
    assert redoc_resp.status_code == 200
    assert "redoc" in redoc_resp.text.lower()


# ============================================================================
# 2. Data & EDA Endpoints (F-API-02)
# ============================================================================

def test_data_summary_endpoint(client):
    """Verifies /api/v1/data/summary returns descriptive stats, missingness, and Hopkins tendency."""
    resp = client.get("/api/v1/data/summary")
    assert resp.status_code == 200
    data = resp.json()

    assert data["n_samples"] > 0
    assert data["n_columns"] >= 17
    assert len(data["numeric_columns"]) >= 17
    assert "descriptive_statistics" in data
    assert "BALANCE" in data["descriptive_statistics"]

    balance_stats = data["descriptive_statistics"]["BALANCE"]
    assert "mean" in balance_stats
    assert "std" in balance_stats
    assert "median" in balance_stats
    assert "skewness" in balance_stats
    assert balance_stats["count"] > 0

    assert "missing_summary" in data
    assert "total_missing_cells" in data["missing_summary"]
    assert "hopkins_statistic" in data
    assert 0.0 <= data["hopkins_statistic"] <= 1.0


def test_data_correlations_endpoint(client):
    """Verifies /api/v1/data/correlations returns Pearson and Spearman correlation matrices."""
    resp = client.get("/api/v1/data/correlations")
    assert resp.status_code == 200
    data = resp.json()

    assert "columns" in data
    assert "pearson" in data
    assert "spearman" in data
    assert "BALANCE" in data["pearson"]
    assert "PURCHASES" in data["pearson"]["BALANCE"]
    assert round(data["pearson"]["BALANCE"]["BALANCE"], 1) == 1.0


def test_data_distributions_endpoint(client):
    """Verifies /api/v1/data/distributions calculates histogram bins and distribution metrics."""
    resp = client.get("/api/v1/data/distributions?n_bins=12")
    assert resp.status_code == 200
    data = resp.json()

    assert "features" in data
    assert "BALANCE" in data["features"]
    bal_dist = data["features"]["BALANCE"]
    assert bal_dist["feature"] == "BALANCE"
    assert len(bal_dist["bins"]) == 12
    assert bal_dist["count"] > 0
    assert bal_dist["bins"][0]["bin_start"] <= bal_dist["bins"][0]["bin_end"]


def test_data_sample_preview_endpoint(client):
    """Verifies /api/v1/data/sample returns preview records matching query limit."""
    resp = client.get("/api/v1/data/sample?limit=15")
    assert resp.status_code == 200
    data = resp.json()

    assert data["sample_size"] == 15
    assert len(data["records"]) == 15
    assert "columns" in data


def test_data_upload_csv_endpoint(client):
    """Verifies /api/v1/data/upload accepts a CSV payload and updates active dataset in memory."""
    csv_content = (
        "CUST_ID,BALANCE,BALANCE_FREQUENCY,PURCHASES,ONEOFF_PURCHASES,INSTALLMENTS_PURCHASES,"
        "CASH_ADVANCE,PURCHASES_FREQUENCY,ONEOFF_PURCHASES_FREQUENCY,PURCHASES_INSTALLMENTS_FREQUENCY,"
        "CASH_ADVANCE_FREQUENCY,CASH_ADVANCE_TRX,PURCHASES_TRX,CREDIT_LIMIT,PAYMENTS,MINIMUM_PAYMENTS,"
        "PRC_FULL_PAYMENT,TENURE\n"
    )
    for i in range(25):
        csv_content += f"C{1000+i},{500+i*50},1.0,{200+i*20},{100+i*10},{100+i*10},0.0,0.8,0.5,0.5,0.0,0,{5+i},5000.0,{300+i*15},150.0,0.2,12\n"

    files = {"file": ("test_upload.csv", csv_content.encode("utf-8"), "text/csv")}
    resp = client.post("/api/v1/data/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["n_samples"] == 25
    assert 0.0 <= data["hopkins_statistic"] <= 1.0


def test_data_upload_invalid_file_error(client):
    """Verifies uploading an invalid non-CSV file returns 400 Bad Request."""
    files = {"file": ("invalid.json", b'{"key": "val"}', "application/json")}
    resp = client.post("/api/v1/data/upload", files=files)
    assert resp.status_code == 400


# ============================================================================
# 3. Clustering & Profiling Endpoints (F-API-03)
# ============================================================================

@pytest.mark.parametrize("algorithm,params", [
    ("kmeans", {"n_clusters": 3, "init": "k-means++", "max_iter": 100}),
    ("kmedoids", {"n_clusters": 3, "metric": "euclidean"}),
    ("dbscan", {"eps": 1.5, "min_samples": 5}),
    ("hdbscan", {"min_cluster_size": 10, "min_samples": 5}),
    ("agglomerative", {"n_clusters": 3, "linkage": "ward"}),
    ("gmm", {"n_clusters": 3, "covariance_type": "full"}),
])
def test_clustering_run_all_algorithms(client, algorithm, params):
    """Verifies /api/v1/cluster/run executes all 6 models across all 4 paradigms."""
    payload = {
        "algorithm": algorithm,
        "params": params,
        "pipeline": {
            "imputer": "median",
            "outlier_method": "winsorize",
            "scaler": "yeo_johnson",
            "feature_engineering": True,
        },
        "random_state": 42,
    }
    resp = client.post("/api/v1/cluster/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["algorithm"] == algorithm
    assert "labels" in data
    assert len(data["labels"]) > 0
    assert "silhouette_score" in data
    assert "davies_bouldin_index" in data
    assert "calinski_harabasz_score" in data
    assert len(data["personas"]) > 0
    assert "execution_time_ms" in data


def test_clustering_with_pca_reduction(client):
    """Verifies /api/v1/cluster/run handles dimensionality reduction within preprocessing."""
    payload = {
        "algorithm": "kmeans",
        "params": {"n_clusters": 3},
        "pipeline": {
            "imputer": "median",
            "outlier_method": "winsorize",
            "scaler": "standard",
            "pca_components": 3,
            "feature_engineering": True,
        },
    }
    resp = client.post("/api/v1/cluster/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_clusters"] == 3


def test_cluster_projections_endpoint(client):
    """Verifies /api/v1/cluster/projections returns PCA, UMAP, and t-SNE coordinate embeddings."""
    for method in ["pca", "umap", "tsne"]:
        resp = client.get(f"/api/v1/cluster/projections?method={method}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == method
        assert len(data["coords_2d"]) > 0
        assert len(data["labels"]) == len(data["coords_2d"])


def test_cluster_profiles_and_personas_endpoint(client):
    """Verifies /api/v1/cluster/profiles returns personas, radar profiles, and ANOVA importance."""
    resp = client.get("/api/v1/cluster/profiles")
    assert resp.status_code == 200
    data = resp.json()

    assert "personas" in data
    assert len(data["personas"]) > 0
    p0 = data["personas"][0]
    assert "persona_name" in p0
    assert "business_description" in p0
    assert "marketing_strategy" in p0
    assert "top_distinguishing_features" in p0

    assert "radar_profiles" in data
    assert "feature_importance" in data
    assert "centroids" in data


def test_cluster_silhouette_samples_endpoint(client):
    """Verifies /api/v1/cluster/silhouette-samples returns ribbon data per sample."""
    resp = client.get("/api/v1/cluster/silhouette-samples?sample_limit=100")
    assert resp.status_code == 200
    data = resp.json()

    assert "global_score" in data
    assert "samples" in data
    assert len(data["samples"]) > 0
    assert "silhouette_value" in data["samples"][0]
    assert "per_cluster_means" in data


def test_cluster_elbow_curve_endpoint(client):
    """Verifies /api/v1/cluster/elbow computes WCSS inertia and detects elbow k."""
    resp = client.get("/api/v1/cluster/elbow?k_min=2&k_max=5")
    assert resp.status_code == 200
    data = resp.json()

    assert data["k_values"] == [2, 3, 4, 5]
    assert len(data["inertias"]) == 4
    assert 2 <= data["elbow_k"] <= 5
    assert data["inertias"][0] > data["inertias"][-1]  # Monotonically decreasing


# ============================================================================
# 4. Autoresearch & Optimization Endpoints (F-API-04)
# ============================================================================

def test_autoresearch_start_and_status(client):
    """Verifies /api/v1/autoresearch/start triggers background search and /status tracks progress."""
    payload = {
        "max_steps": 4,
        "patience": 2,
        "target_algorithm": "kmeans",
        "allow_algorithm_mutation": False,
        "random_state": 42,
    }
    start_resp = client.post("/api/v1/autoresearch/start", json=payload)
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    job_id = start_data["job_id"]
    assert start_data["status"] == "running"

    # Allow background loop to execute steps
    time.sleep(1.0)

    status_resp = client.get(f"/api/v1/autoresearch/status?job_id={job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert status_data["current_step"] >= 0


def test_autoresearch_pause_and_stop(client):
    """Verifies /api/v1/autoresearch/pause and /stop control job execution."""
    start_resp = client.post("/api/v1/autoresearch/start", json={"max_steps": 10, "patience": 3})
    assert start_resp.status_code == 200
    job_id = start_resp.json()["job_id"]

    # Pause
    pause_resp = client.post(f"/api/v1/autoresearch/pause?job_id={job_id}")
    assert pause_resp.status_code == 200

    # Stop
    stop_resp = client.post(f"/api/v1/autoresearch/stop?job_id={job_id}")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"


def test_autoresearch_leaderboard_and_ablations(client):
    """Verifies /api/v1/autoresearch/leaderboard and /ablations query candidate ledger."""
    # Start a quick 3-step run and let it finish
    start_resp = client.post("/api/v1/autoresearch/start", json={"max_steps": 3, "patience": 2})
    job_id = start_resp.json()["job_id"]
    time.sleep(0.8)

    lb_resp = client.get(f"/api/v1/autoresearch/leaderboard?job_id={job_id}&top_n=5")
    assert lb_resp.status_code == 200
    lb_data = lb_resp.json()
    assert "top_entries" in lb_data

    ab_resp = client.get(f"/api/v1/autoresearch/ablations?job_id={job_id}")
    assert ab_resp.status_code == 200
    ab_data = ab_resp.json()
    assert "stage_contributions" in ab_data
    assert "parameter_importance_variance" in ab_data


def test_autoresearch_sse_streaming(client):
    """Verifies /api/v1/autoresearch/stream emits Server-Sent Events."""
    start_resp = client.post("/api/v1/autoresearch/start", json={"max_steps": 2, "patience": 2})
    job_id = start_resp.json()["job_id"]

    stream_resp = client.get(f"/api/v1/autoresearch/stream?job_id={job_id}")
    assert stream_resp.status_code == 200
    lines = stream_resp.iter_lines()
    assert len(lines) > 0
    # Check SSE data: format
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) > 0


# ============================================================================
# 5. Research Literature & Benchmark Synthesis Endpoints (R3)
# ============================================================================

def test_research_literature_endpoint(client):
    """Verifies /api/v1/research/literature returns citations, taxonomies, and principles."""
    resp = client.get("/api/v1/research/literature")
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["citations"]) >= 9
    assert "kmeans" in data["algorithms"]
    assert "silhouette" in data["metrics"]
    assert len(data["principles"]) >= 5

    # Filter by topic
    topic_resp = client.get("/api/v1/research/literature?topic=density")
    assert topic_resp.status_code == 200
    topic_data = topic_resp.json()
    assert len(topic_data["citations"]) > 0


def test_research_benchmark_matrix_endpoint(client):
    """Verifies /api/v1/research/benchmark-matrix returns 6-model cross-paradigm metrics."""
    resp = client.get("/api/v1/research/benchmark-matrix?sample_size=300&n_bootstraps=2")
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["models"]) == 6
    for m in data["models"]:
        assert "model_name" in m
        assert "paradigm" in m
        assert "silhouette_mean" in m
        assert "davies_bouldin_mean" in m
        assert "calinski_harabasz_mean" in m
        assert "stability_ari_mean" in m
        assert "composite_fitness" in m
        assert 0.0 <= m["composite_fitness"] <= 1.0


def test_research_ablations_endpoint(client):
    """Verifies /api/v1/research/ablations returns baseline vs. optimized deltas."""
    resp = client.get("/api/v1/research/ablations")
    assert resp.status_code == 200
    data = resp.json()

    assert "ablations" in data
    assert len(data["ablations"]) > 0
    ab0 = data["ablations"][0]
    assert "model_name" in ab0
    assert "baseline_fitness" in ab0
    assert "optimized_fitness" in ab0
    assert "delta_fitness" in ab0


def test_research_table_exports(client):
    """Verifies /api/v1/research/export produces publication LaTeX and Markdown tables."""
    # LaTeX Benchmark
    resp_l_bm = client.post("/api/v1/research/export", json={"format": "latex", "table_type": "benchmark"})
    assert resp_l_bm.status_code == 200
    assert r"\begin{table}" in resp_l_bm.json()["content"]

    # Markdown Benchmark
    resp_m_bm = client.post("/api/v1/research/export", json={"format": "markdown", "table_type": "benchmark"})
    assert resp_m_bm.status_code == 200
    assert "| Model | Paradigm |" in resp_m_bm.json()["content"]

    # LaTeX Ablation
    resp_l_ab = client.post("/api/v1/research/export", json={"format": "latex", "table_type": "ablation"})
    assert resp_l_ab.status_code == 200
    assert r"\begin{table}" in resp_l_ab.json()["content"]

    # Markdown Ablation
    resp_m_ab = client.post("/api/v1/research/export", json={"format": "markdown", "table_type": "ablation"})
    assert resp_m_ab.status_code == 200
    assert "| Model | Base Sil |" in resp_m_ab.json()["content"]


# ============================================================================
# 6. Real-Time Customer Inference Endpoints (F-API-05)
# ============================================================================

def test_single_customer_inference(client):
    """Verifies /api/v1/inference/predict scores a single customer vector and returns persona."""
    customer_vector = {
        "BALANCE": 1500.0,
        "BALANCE_FREQUENCY": 1.0,
        "PURCHASES": 600.0,
        "ONEOFF_PURCHASES": 300.0,
        "INSTALLMENTS_PURCHASES": 300.0,
        "CASH_ADVANCE": 0.0,
        "PURCHASES_FREQUENCY": 0.8,
        "ONEOFF_PURCHASES_FREQUENCY": 0.4,
        "PURCHASES_INSTALLMENTS_FREQUENCY": 0.6,
        "CASH_ADVANCE_FREQUENCY": 0.0,
        "CASH_ADVANCE_TRX": 0,
        "PURCHASES_TRX": 15,
        "CREDIT_LIMIT": 4000.0,
        "PAYMENTS": 800.0,
        "MINIMUM_PAYMENTS": 200.0,
        "PRC_FULL_PAYMENT": 0.25,
        "TENURE": 12,
    }
    resp = client.post("/api/v1/inference/predict", json={"features": customer_vector})
    assert resp.status_code == 200
    data = resp.json()

    assert "cluster_id" in data
    assert isinstance(data["cluster_id"], int)
    assert "probabilities" in data
    assert len(data["probabilities"]) > 0
    assert "persona" in data
    assert "persona_name" in data["persona"]
    assert "business_description" in data["persona"]


def test_batch_customer_inference(client):
    """Verifies /api/v1/inference/batch scores multiple customer records simultaneously."""
    batch_customers = [
        {"BALANCE": 200.0, "PURCHASES": 1500.0, "CREDIT_LIMIT": 5000.0},
        {"BALANCE": 4500.0, "PURCHASES": 50.0, "CASH_ADVANCE": 3000.0},
        {"BALANCE": 50.0, "PURCHASES": 20.0, "CREDIT_LIMIT": 1000.0},
    ]
    resp = client.post("/api/v1/inference/batch", json={"customers": batch_customers})
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_samples"] == 3
    assert len(data["predictions"]) == 3
    assert "cluster_counts" in data
    for pred in data["predictions"]:
        assert "cluster_id" in pred
        assert "probabilities" in pred
        assert "persona_name" in pred


def test_inference_empty_batch_error(client):
    """Verifies /api/v1/inference/batch rejects empty customer list with 400 Bad Request."""
    resp = client.post("/api/v1/inference/batch", json={"customers": []})
    assert resp.status_code == 400


def test_unknown_endpoint_404(client):
    """Verifies unmapped URLs return 404 Not Found."""
    resp = client.get("/api/v1/nonexistent_route")
    assert resp.status_code == 404
