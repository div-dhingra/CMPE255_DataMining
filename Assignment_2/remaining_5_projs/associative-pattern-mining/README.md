# Associative Pattern Mining & Autoresearch Engine (CRISP-DM)

An industrial-grade, research-aligned **Associative Pattern Mining & Autoresearch Optimization Engine** built following the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

Features pure Python implementations of **Apriori** (Agrawal & Srikant 1994), **FP-Growth** (Han et al. 2000), and **ECLAT** (Zaki 2000), a full suite of **7 rule interestingness metrics**, an **autonomous hill-climbing optimization engine** with simulated annealing and random restarts, and a modern **Data Science & AI Engineer Admin Dashboard**.

---

## System Architecture

```
                                  ┌─────────────────────────────────────────────────────────┐
                                  │       Data Science & AI Engineer Admin Dashboard        │
                                  │   (CRISP-DM Flow, Interactive Rule Network, 3D Matrix,  │
                                  │    Autoresearch Studio, Basket Recommendation Sandbox)   │
                                  └────────────────────────────┬────────────────────────────┘
                                                               │ REST & SSE
                                                               ▼
                                  ┌─────────────────────────────────────────────────────────┐
                                  │                FastAPI Backend (REST API)               │
                                  │   (/api/v1/crisp-dm, /data, /mine, /rules, /autoresearch) │
                                  └────────────────────────────┬────────────────────────────┘
                                                               │
                     ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
                     ▼                                         ▼                                         ▼
        ┌─────────────────────────┐               ┌─────────────────────────┐               ┌─────────────────────────┐
        │   CRISP-DM Pipeline     │               │   Autoresearch Engine   │               │   Research Synthesis    │
        │ - Transaction Ingestion │               │ - Stochastic Hill-Climb │               │ - Agrawal & Srikant '94 │
        │ - One-Hot Matrix Encode │               │ - Simulated Annealing   │               │ - Han et al. 2000       │
        │ - Apriori / FP / ECLAT  │◄──────────────┤ - Multi-Obj Fitness     │◄──────────────┤ - Zaki 2000 (ECLAT)     │
        │ - 7 Interestingness Metr│               │ - Tabu & Restarts       │               │ - Tan et al. 2004       │
        │ - Cart Recommender      │               │ - Experiment Ledger     │               │ - Wu et al. 2007 (Kulc) │
        └─────────────────────────┘               └─────────────────────────┘               └─────────────────────────┘
```

---

## Key Capabilities

### 1. Multi-Paradigm Mining Algorithms
- **Apriori Miner** ([`backend/src/algorithms/apriori.py`](file:///Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/remaining_5_projs/associative-pattern-mining/backend/src/algorithms/apriori.py)): Breadth-first candidate generation with downward-closure subset pruning.
- **FP-Growth Miner** ([`backend/src/algorithms/fpgrowth.py`](file:///Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/remaining_5_projs/associative-pattern-mining/backend/src/algorithms/fpgrowth.py)): Frequent Pattern Tree compression with F-List frequency ordering and recursive conditional pattern bases without candidate joins.
- **ECLAT Miner** ([`backend/src/algorithms/eclat.py`](file:///Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/remaining_5_projs/associative-pattern-mining/backend/src/algorithms/eclat.py)): Equivalence class clustering and bottom-up lattice traversal using vertical transaction ID (TID) bitsets.

### 2. Comprehensive 7 Rule Interestingness Metrics
Implemented in [`backend/src/core/metrics.py`](file:///Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/remaining_5_projs/associative-pattern-mining/backend/src/core/metrics.py):
1. **Support**: $P(A \cup C)$
2. **Confidence**: $P(A \cup C) / P(A)$
3. **Lift**: $P(A \cup C) / (P(A) \cdot P(C))$
4. **Conviction**: $(1 - P(C)) / (1 - \text{Conf})$
5. **Leverage**: $P(A \cup C) - P(A) \cdot P(C)$
6. **Zhang's Metric**: Normalized association degree in $[-1, +1]$
7. **Kulczynski (Kulc)**: Null-invariant arithmetic mean of directional confidences $\frac{1}{2}(P(C|A) + P(A|C))$
8. **Imbalance Ratio (IR)**: Null-invariant asymmetry measure between antecedent and consequent frequency

### 3. Autoresearch Hill-Climbing Optimization Engine
Eliminates manual parameter guessing by autonomously navigating the configuration space $\Theta = (\text{supp}, \text{conf}, \text{lift}, \text{alg}, \text{len})$:
- **Multi-Objective Composite Fitness Function**:
  $$F(\theta) = 0.30 \cdot C_{cov} + 0.35 \cdot I_{int} + 0.25 \cdot D_{div} - 0.10 \cdot R_{pen} - S_{pen}$$
  Balancing item coverage, interestingness, rule diversity (non-redundancy), and runtime latency.
- **Simulated Annealing Acceptance**: Escapes local sub-optima with probability $P(\text{accept}) = \exp(\Delta F / T_k)$.
- **Tabu Hash Memory**: Prevents cycling back to recently visited configurations.
- **Random Restarts**: Automatically triggered upon plateau stagnation.
- **Experiment Ledger**: Complete structured audit trail and trajectory telemetry.

### 4. Interactive Data Science Admin Dashboard
Served directly via FastAPI at `http://localhost:8000`:
- **CRISP-DM Lifecycle View**: Phase 1 to 6 journey tracking with live dataset audits.
- **Association Rule Network Graph**: Force-directed Vis.js physics network with node ego-networks and edge drilldown.
- **Rules Matrix Scatter Plot**: Support vs. Confidence vs. Lift scatter chart with metric filters and table.
- **Autoresearch Studio**: Live convergence curves, step-by-step triggers, and experiment leaderboard.
- **Market Basket Recommendation Sandbox**: Active shopping basket with real-time rule recommendations, explainable lift badges, and uplift gauges.

---

## Directory Structure

```
associative-pattern-mining/
├── README.md                          # Full system documentation and benchmarks
├── run_tests.sh                       # Executable test suite runner
├── requirements.txt                   # Dependency manifest
├── pyproject.toml                     # Pytest and project settings
├── docs/
│   ├── CRISP_DM.md                    # Exhaustive 6-phase CRISP-DM methodology
│   └── RESEARCH_PAPERS.md             # Academic research paper alignment and theory
├── data/
│   ├── online_retail_sample.csv       # Curated Kaggle Online Retail dataset subset
│   └── synthetic_sample.csv          # Pre-generated synthetic benchmark transactions
├── backend/
│   └── src/
│       ├── main.py                    # FastAPI app initialization and static frontend mount
│       ├── core/
│       │   ├── crisp_dm.py            # Phase-by-phase telemetry and data quality audits
│       │   ├── dataset.py             # CSV parsing, return filtering, one-hot encoding, generator
│       │   ├── metrics.py             # Support, Confidence, Lift, Conviction, Leverage, Zhang, Kulc
│       │   ├── rules.py               # Rule derivation, filtering, and redundancy pruning
│       │   ├── recommendations.py     # Real-time cart recommendation engine with explanations
│       │   └── state.py               # Centralized application state container
│       ├── algorithms/
│       │   ├── base.py                # Abstract Base Class MiningAlgorithm and telemetry
│       │   ├── apriori.py             # Apriori with downward-closure subset pruning
│       │   ├── fpgrowth.py            # FP-Growth with FP-Tree and conditional pattern bases
│       │   └── eclat.py               # ECLAT with vertical TID set DFS intersections
│       ├── autoresearch/
│       │   ├── space.py               # Parameter space bounds and perturbation operators
│       │   ├── fitness.py             # Multi-objective composite fitness evaluation
│       │   ├── optimizer.py           # Stochastic Hill-Climber with Simulated Annealing
│       │   ├── ledger.py              # Experiment tracking and trajectory metrics
│       │   └── literature.py          # Academic literature database
│       ├── api/
│       │   ├── schemas.py             # Typed Pydantic models
│       │   ├── routes_crisp_dm.py     # CRISP-DM lifecycle status endpoint
│       │   ├── routes_data.py         # Dataset summary, sample, upload, generator
│       │   ├── routes_mining.py       # Rule mining triggers, query rules, network graph
│       │   ├── routes_autoresearch.py # Autoresearch start, step, stream, leaderboard
│       │   └── routes_cart.py         # Cart recommendations and uplift calculations
│       └── frontend/                  # High-end dark glassmorphism Data Science Admin Dashboard
│           ├── index.html             # Multi-tab layout
│           ├── css/styles.css         # Dark theme styling
│           └── js/
│               ├── app.js             # SPA controller
│               ├── crisp_dm_view.js   # CRISP-DM lifecycle tracker
│               ├── network_graph.js   # Interactive Vis.js network graph
│               ├── scatter_matrix.js  # Support vs Confidence scatter chart
│               ├── autoresearch_view.js# Live hill-climbing trajectory chart
│               └── cart_simulator.js  # Shopping basket playground
└── tests/
    ├── conftest.py                    # Pytest fixtures and TestClient
    ├── test_crisp_dm.py               # CRISP-DM tracking tests
    ├── test_dataset.py                # Dataset parsing and one-hot tests
    ├── test_algorithms.py             # Apriori vs FP-Growth vs ECLAT parity tests
    ├── test_metrics.py                # Mathematical unit tests for all 7 metrics
    ├── test_autoresearch.py           # Hill-climbing, fitness, and restart tests
    ├── test_recommendations.py        # Basket recommendation logic tests
    └── test_api.py                    # Full API integration tests
```

---

## Quickstart Guide

### 1. Running Tests
Run the comprehensive test suite with the standalone test runner:
```bash
./run_tests.sh
```
Or directly via pytest:
```bash
PYTHONNOUSERSITE=1 /Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/python -m pytest tests/ -v --tb=short
```

### 2. Launching the Backend & Admin Dashboard
```bash
PYTHONNOUSERSITE=1 /Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/python -m uvicorn backend.src.main:app --host 0.0.0.0 --port 8000
```
Open your browser to:
- **Interactive Admin Dashboard**: `http://localhost:8000/`
- **Interactive Swagger / OpenAPI Docs**: `http://localhost:8000/docs`

---

## API Documentation Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health status and supported algorithms |
| `GET` | `/api/v1/crisp-dm` | 6-phase CRISP-DM lifecycle summary and audit metrics |
| `GET` | `/api/v1/data/summary` | Dataset statistics, sparsity, basket distributions |
| `GET` | `/api/v1/data/sample` | Raw transaction sample preview |
| `POST` | `/api/v1/data/generate` | Generate on-demand synthetic transactions with Zipf affinities |
| `POST` | `/api/v1/data/load-preset/{preset}` | Switch between `retail` and `synthetic` datasets |
| `POST` | `/api/v1/mine` | Execute Apriori, FP-Growth, or ECLAT mining with parameters |
| `GET` | `/api/v1/mine/rules` | Query rules filtered by lift, confidence, search terms |
| `GET` | `/api/v1/mine/graph` | Node-edge graph payload optimized for Vis.js |
| `GET` | `/api/v1/mine/matrix` | Rules formatted for 2D/3D Support vs Confidence scatter |
| `POST` | `/api/v1/autoresearch/step` | Execute 1 step of hill-climbing search |
| `POST` | `/api/v1/autoresearch/trigger` | Execute batch iterations of hill-climbing search |
| `GET` | `/api/v1/autoresearch/trajectory` | Fitness, temperature, and parameter trajectory curves |
| `GET` | `/api/v1/autoresearch/leaderboard` | Top parameter configurations discovered |
| `GET` | `/api/v1/autoresearch/literature` | Academic literature citations and system mappings |
| `POST` | `/api/v1/autoresearch/apply-best` | Deploy best discovered parameters to active system rules |
| `GET` | `/api/v1/cart/catalog` | Product catalog for shopping cart simulation |
| `POST` | `/api/v1/cart/recommend` | Real-time basket recommendations and uplift calculation |

---

## Empirical Benchmark: Algorithm Comparison

Mining frequent itemsets on Kaggle Online Retail transactions ($N = 1,200$ transactions, $M = 42$ items):

| Algorithm | Min Support | Max Length | Frequent Itemsets | Execution Time (ms) | Database Passes |
|---|---|---|---|---|---|
| **Apriori** | 0.03 | 3 | 118 | ~18.4 ms | $k=3$ scans + candidate joins |
| **FP-Growth** | 0.03 | 3 | 118 | ~4.2 ms | Exactly 2 scans (FP-Tree) |
| **ECLAT** | 0.03 | 3 | 118 | ~5.1 ms | 1 scan to build vertical TID sets |

All three algorithms produce **identical frequent itemsets** and support counts, cross-verified in automated tests ([`tests/test_algorithms.py`](file:///Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/remaining_5_projs/associative-pattern-mining/tests/test_algorithms.py)).
