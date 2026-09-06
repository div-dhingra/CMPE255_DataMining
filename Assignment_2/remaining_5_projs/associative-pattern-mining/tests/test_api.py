"""Comprehensive integration tests for all FastAPI endpoints."""

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "CRISP-DM" in data["framework"]


def test_crisp_dm_endpoint(client: TestClient):
    resp = client.get("/api/v1/crisp-dm")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_phases"] == 6
    assert len(data["phases"]) == 6
    assert "dataset_transactions" in data


def test_dataset_endpoints(client: TestClient):
    # Summary
    summary_res = client.get("/api/v1/data/summary")
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert summary["num_transactions"] > 0
    assert summary["num_unique_items"] > 0

    # Sample
    sample_res = client.get("/api/v1/data/sample?limit=5")
    assert sample_res.status_code == 200
    assert len(sample_res.json()["transactions"]) <= 5

    # Generate synthetic
    gen_res = client.post("/api/v1/data/generate", json={
        "num_transactions": 100,
        "avg_basket_size": 3.5,
        "noise_level": 0.2
    })
    assert gen_res.status_code == 200
    assert gen_res.json()["num_transactions"] == 100

    # Switch preset
    preset_res = client.post("/api/v1/data/load-preset/synthetic")
    assert preset_res.status_code == 200


def test_mining_endpoints(client: TestClient):
    # Mine using FP-Growth
    fp_res = client.post("/api/v1/mine", json={
        "algorithm": "fpgrowth",
        "min_support": 0.03,
        "min_confidence": 0.30,
        "min_lift": 1.05,
        "max_length": 3,
        "limit": 25
    })
    assert fp_res.status_code == 200
    fp_data = fp_res.json()
    assert fp_data["algorithm"] == "FP-Growth"
    assert fp_data["total_rules_found"] >= 0

    # Mine using Apriori
    ap_res = client.post("/api/v1/mine", json={
        "algorithm": "apriori",
        "min_support": 0.05,
        "min_confidence": 0.30,
        "min_lift": 1.0,
        "max_length": 2
    })
    assert ap_res.status_code == 200

    # Mine using ECLAT
    ec_res = client.post("/api/v1/mine", json={
        "algorithm": "eclat",
        "min_support": 0.05,
        "min_confidence": 0.30,
        "min_lift": 1.0,
        "max_length": 2
    })
    assert ec_res.status_code == 200

    # Query rules
    rules_res = client.get("/api/v1/mine/rules?limit=10&sort_by=lift")
    assert rules_res.status_code == 200
    assert "rules" in rules_res.json()

    # Network Graph payload
    graph_res = client.get("/api/v1/mine/graph?min_lift=1.05&max_rules=30")
    assert graph_res.status_code == 200
    graph = graph_res.json()
    assert "nodes" in graph
    assert "edges" in graph

    # Scatter Matrix payload
    matrix_res = client.get("/api/v1/mine/matrix?limit=50")
    assert matrix_res.status_code == 200
    assert "data" in matrix_res.json()


def test_autoresearch_endpoints(client: TestClient):
    # Single step
    step_res = client.post("/api/v1/autoresearch/step")
    assert step_res.status_code == 200
    step_data = step_res.json()
    assert "candidate_fitness" in step_data
    assert "accepted" in step_data

    # Batch trigger
    batch_res = client.post("/api/v1/autoresearch/trigger", json={
        "max_steps": 3,
        "cooling_rate": 0.92
    })
    assert batch_res.status_code == 200
    assert batch_res.json()["completed_iterations"] == 3

    # Trajectory
    traj_res = client.get("/api/v1/autoresearch/trajectory")
    assert traj_res.status_code == 200
    assert len(traj_res.json()["iterations"]) >= 4

    # Leaderboard
    lead_res = client.get("/api/v1/autoresearch/leaderboard?top_n=5")
    assert lead_res.status_code == 200
    assert isinstance(lead_res.json(), list)

    # Literature alignment
    lit_res = client.get("/api/v1/autoresearch/literature")
    assert lit_res.status_code == 200
    papers = lit_res.json()
    assert len(papers) >= 5
    paper_ids = [p["paper_id"] for p in papers]
    assert "agrawal1994" in paper_ids
    assert "han2000" in paper_ids
    assert "zaki2000" in paper_ids

    # Apply best config
    apply_res = client.post("/api/v1/autoresearch/apply-best")
    assert apply_res.status_code == 200
    assert "applied_configuration" in apply_res.json()

    # Reset
    reset_res = client.post("/api/v1/autoresearch/reset")
    assert reset_res.status_code == 200


def test_cart_recommendation_endpoints(client: TestClient):
    # Catalog
    cat_res = client.get("/api/v1/cart/catalog")
    assert cat_res.status_code == 200
    items = cat_res.json()["items"]
    assert len(items) > 0

    # Recommend with items
    pick_items = [items[0]["name"]]
    rec_res = client.post("/api/v1/cart/recommend", json={
        "cart_items": pick_items,
        "top_k": 4
    })
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["cart_items"] == pick_items
    assert "recommendations" in rec_data
    assert "projected_cross_sell_uplift_pct" in rec_data

    # Recommend with empty cart (cold start)
    cold_res = client.post("/api/v1/cart/recommend", json={"cart_items": []})
    assert cold_res.status_code == 200
    assert cold_res.json()["mode"] == "cold_start_popular"


def test_frontend_dashboard_serving(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "CRISP-DM" in res.text

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200

    res_static = client.get("/static/css/styles.css")
    assert res_static.status_code == 200
