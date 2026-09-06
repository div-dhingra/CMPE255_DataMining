"""Tier 4: FastAPI End-to-End API and Dashboard Integration Tests.

Verifies:
- Health and status endpoints
- Architecture diagnostics and telemetry
- KV-Cache footprint calculator
- Attention map extraction
- Chat completions and SSE streaming
- Autoresearch controls, step execution, and CSV ledger download
- Literature matrix, ablations, and CRISP-DM summary endpoints
- Admin Dashboard static file serving
"""

import pytest


class TestApiEndpoints:
    def test_health_check(self, api_client):
        res = api_client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["service"] == "llm-chatbot-autoresearch"

    def test_model_architecture_and_telemetry(self, api_client):
        res = api_client.get("/api/model/architecture")
        assert res.status_code == 200
        data = res.json()
        assert data["total_parameters"] > 0
        assert data["config"]["rope_theta"] == 10000.0
        assert len(data["layer_breakdown"]) > 0

        res_tel = api_client.get("/api/model/telemetry")
        assert res_tel.status_code == 200
        tel_data = res_tel.json()
        assert "device" in tel_data
        assert tel_data["status"] == "online"

    def test_kv_cache_calculator(self, api_client):
        payload = {
            "batch_size": 2,
            "seq_len": 2048,
            "dim": 384,
            "n_heads": 8,
            "n_layers": 8,
            "precision_bytes": 2,
        }
        res = api_client.post("/api/model/kv-cache-calculator", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "footprints_mb" in data
        assert data["footprints_mb"]["MHA (Full Heads)"] > data["footprints_mb"]["GQA-2 (Quarter KV Heads)"]
        assert data["footprints_mb"]["GQA-2 (Quarter KV Heads)"] > data["footprints_mb"]["MQA (Single KV Head)"]

    def test_attention_map_extraction(self, api_client):
        res = api_client.post(
            "/api/model/attention-map",
            json={"prompt": "Attention mechanisms in deep transformers", "layer_idx": 0},
        )
        assert res.status_code == 200
        data = res.json()
        assert "tokens" in data
        assert "matrix" in data
        assert len(data["matrix"]) == len(data["tokens"])

    def test_chat_completions(self, api_client):
        payload = {
            "messages": [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "What is SwiGLU?"},
            ],
            "max_tokens": 32,
            "temperature": 0.7,
        }
        res = api_client.post("/api/chat/completions", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data["choices"]) > 0
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert "telemetry" in data

    def test_chat_stream(self, api_client):
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            "stream": True,
        }
        res = api_client.post("/api/chat/stream", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        body = res.text
        assert "data: " in body
        assert "[DONE]" in body

    def test_autoresearch_endpoints(self, api_client):
        # 1. Check status
        res_stat = api_client.get("/api/autoresearch/status")
        assert res_stat.status_code == 200

        # 2. Execute a single step
        res_step = api_client.post("/api/autoresearch/step")
        assert res_step.status_code == 200
        step_data = res_step.json()
        assert step_data["status"] == "success"
        assert "entry" in step_data

        # 3. Retrieve ledger
        res_ledger = api_client.get("/api/autoresearch/ledger")
        assert res_ledger.status_code == 200
        assert res_ledger.json()["total_entries"] > 0

        # 4. Export CSV
        res_csv = api_client.get("/api/autoresearch/export/csv")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers["content-type"]

    def test_benchmarks_and_crisp_dm(self, api_client):
        res_mat = api_client.get("/api/benchmarks/matrix")
        assert res_mat.status_code == 200
        assert "papers" in res_mat.json()

        res_abl = api_client.get("/api/benchmarks/ablations")
        assert res_abl.status_code == 200
        assert len(res_abl.json()) >= 3

        res_crisp = api_client.get("/api/crisp-dm/summary")
        assert res_crisp.status_code == 200
        assert len(res_crisp.json()["phases"]) == 6

    def test_dashboard_serving(self, api_client):
        res = api_client.get("/")
        assert res.status_code == 200
        assert "SOTA LLM Studio" in res.text

        res_dash = api_client.get("/dashboard")
        assert res_dash.status_code == 200
        assert "SOTA LLM Studio" in res_dash.text
