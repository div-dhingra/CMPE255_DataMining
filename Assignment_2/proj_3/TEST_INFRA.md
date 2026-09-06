# E2E Test Infrastructure & Test Architecture

## 1. Overview & Architecture

The **CRISP-DM Clustering & Autoresearch Engine** test suite is built on a four-tier, opaque-box, requirement-driven testing architecture. It validates the complete data mining lifecycle, autonomous optimization engine, research synthesis, and high-performance API backend against the authoritative requirements in `PROJECT.md` and `ORIGINAL_REQUEST.md`.

```
backend/tests/
├── e2e/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures & mathematical verification oracles
│   ├── fixtures/
│   │   ├── __init__.py
│   │   └── synthetic_data.py         # High-fidelity Kaggle Credit Card dataset generator
│   ├── test_tier1_features.py        # Tier 1: Feature Coverage (170 tests across 34 features)
│   ├── test_tier2_boundaries.py      # Tier 2: Boundary & Corner Cases (13 tests)
│   ├── test_tier3_interactions.py    # Tier 3: Cross-Feature Pairwise Interactions (7 tests)
│   └── test_tier4_applications.py    # Tier 4: Real-World Application Scenarios (4 tests)
```

---

## 2. Test Harness & Fixtures

### 2.1 Synthetic Kaggle Credit Card Dataset Generator (`fixtures/synthetic_data.py`)
- Generates fully compliant synthetic datasets modeled after the Kaggle Credit Card / Customer Segmentation benchmark.
- **18-Attribute Schema**: `CUST_ID` + 17 continuous numerical behavioral features (`BALANCE`, `BALANCE_FREQUENCY`, `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `CASH_ADVANCE`, `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX`, `PURCHASES_TRX`, `CREDIT_LIMIT`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT`, `TENURE`).
- **5 Realistic Archetypes**: Transactors, Revolvers, Cash Advance Seekers, Inactive Users, and VIP Spenders.
- **Injected Realism**: Configurable missingness rates (e.g. 3.5% in `MINIMUM_PAYMENTS`, 0.1% in `CREDIT_LIMIT`), heavy-tailed skewness, and controllable anomaly/outlier rates.

### 2.2 Mathematical Verification Oracles (`conftest.py`)
Provides deterministic reference calculators as independent ground truth:
1. **Silhouette Coefficient Oracle**: Computes $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ and global mean $S \in [-1, 1]$.
2. **Davies-Bouldin Index Oracle**: Computes $DB = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} \left(\frac{s_i + s_j}{d(\mu_i, \mu_j)}\right)$.
3. **Calinski-Harabasz Index Oracle**: Computes $CH = \frac{\text{Tr}(B_k)/(k-1)}{\text{Tr}(W_k)/(n-k)}$.
4. **Hopkins Statistic Oracle**: Computes spatial clustering tendency $H = \frac{\sum u_i}{\sum u_i + \sum w_i} \in [0, 1]$.
5. **Normalized Composite Fitness Oracle**: Computes $F(\theta) = w_1 S_{\text{norm}} + w_2 (1 - DB_{\text{norm}}) + w_3 CH_{\text{norm}} + w_4 \text{Stability} - P_{\text{noise}} - P_{\text{imbalance}}$.

---

## 3. Four-Tier Test Suite Structure

### Tier 1: Feature Coverage (`test_tier1_features.py`)
- **Coverage**: All 34 features in `PROJECT.md § Feature Inventory` (5+ tests per feature = 170 tests total).
- **Features Tested**:
  - **F-DATA-01 to F-DATA-07**: Dataset Ingestion, Data Understanding & Profiling, Imputation Strategies, Outlier Handling, Feature Transformations & Scaling, Feature Engineering, Dimensionality Reduction Projections.
  - **F-MOD-01 to F-MOD-06**: K-Means, K-Medoids (PAM/FasterPAM), DBSCAN, HDBSCAN, Agglomerative (Ward/Complete/Average), Gaussian Mixture Models.
  - **F-EVAL-01 to F-EVAL-04**: Internal Validation Metrics, Inertia & Kneedle Elbow Detection, Cluster Stability Analysis (ARI), Personas & Radar Profiles.
  - **F-AUTO-01 to F-AUTO-05**: Search Space Parameterization, Normalized Composite Objective $F(\theta)$, Hill-Climbing Optimization Engine, Perturbation & Random Restarts, Experiment Ledger & Ablations.
  - **F-RES-01 to F-RES-02**: Literature Alignment Synthesis & Citations, Comparative Benchmark Matrix & LaTeX/Markdown Export.
  - **F-API-01 to F-API-05**: FastAPI Infrastructure & OpenAPI Docs, Data & EDA Endpoints, Clustering & Profiling Endpoints, Autoresearch & Streaming Endpoints, Real-Time Inference & Prediction Endpoint.
  - **F-UI-01 to F-UI-05**: Overview & CRISP-DM View Schemas, Interactive Cluster Explorer Schemas, Autoresearch Studio Live Trajectory Schemas, Research Benchmark Matrix Schemas, Customer Profiler Playground Schemas.

### Tier 2: Boundary & Corner Cases (`test_tier2_boundaries.py`)
- **Coverage**: 13 extreme edge cases and failure modes.
- **Scenarios Tested**: Empty dataset matrix (0 rows), $N < k$ sample validation, $k=1$ undefined Silhouette handling, zero-variance / constant column scaling, singular/collinear covariance matrix regularization (`reg_covar`), 100% NaN column rejection, 1000x extreme outlier clamping, noise-only density clustering penalty, Hopkins uniform distribution bounds, incomplete inference input imputation, and autoresearch plateau patience exhaustion.

### Tier 3: Cross-Feature Pairwise Interactions (`test_tier3_interactions.py`)
- **Coverage**: 7 integration points testing cross-module synergy.
- **Scenarios Tested**: Yeo-Johnson + GMM covariance conditioning, KNN/MICE Imputation + DBSCAN + UMAP, Winsorization + Agglomerative Ward trees, RobustScaler + K-Medoids Manhattan metric, PCA + Hopkins + K-Means, Hill-climber multi-step state transitions + ledger logging, and Cluster labels $\to$ Personas $\to$ Radar profiles $\to$ Inbound query scoring.

### Tier 4: Real-World Application Scenarios (`test_tier4_applications.py`)
- **Coverage**: 4 complete end-to-end user workflows.
- **Scenarios Tested**:
  1. Full 6-Phase CRISP-DM customer segmentation workflow.
  2. Autonomous autoresearch optimization convergence ($F(\theta^*) > F(\theta_0)$).
  3. Academic paper benchmark matrix reproduction across all 6 models with LaTeX output.
  4. Real-time customer profiler & multi-archetype inference playground scoring.

---

## 4. Test Execution Guide

### Using Standalone Test Runner Script
```bash
# Run full E2E test suite (194 tests)
./run_tests.sh

# Run individual tiers
./run_tests.sh --tier1
./run_tests.sh --tier2
./run_tests.sh --tier3
./run_tests.sh --tier4
```

### Using Pytest Directly
```bash
PYTHONPATH=backend:backend/src python3 -m pytest backend/tests/e2e -v
```
