# Project: CRISP-DM Autonomous Clustering & Autoresearch Engine

## Architecture

```
                                  ┌─────────────────────────────────────────┐
                                  │      Next.js + TypeScript Dashboard      │
                                  │  (CRISP-DM Flow, 2D/3D Cluster Explorer, │
                                  │   Autoresearch Studio, Benchmark Matrix, │
                                  │            Inference Playground)        │
                                  └────────────────────┬────────────────────┘
                                                       │ REST & SSE
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │       High-Performance FastAPI API      │
                                  │ (/api/v1/data, /cluster, /autoresearch) │
                                  └────────────────────┬────────────────────┘
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     ▼                                 ▼                                 ▼
        ┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
        │    CRISP-DM Pipeline    │       │   Autoresearch Engine   │       │    Research Synthesis   │
        │ - Data Cleaning & Prep  │       │ - Hill-Climber Loop     │       │ - Literature Citations  │
        │ - Scaling & Power Trans │       │ - Perturbations/Restart │       │ - Comparative Matrix    │
        │ - 4 Paradigms (6 Models)│◄──────┤ - Composite Fitness     │◄──────┤ - Ablation Breakdown    │
        │ - Metrics (S, DB, CH)   │       │ - Experiment Ledger     │       │ - LaTeX/Markdown Tables │
        │ - Projections (PCA/UMAP)│       │ - State Search Space    │       └─────────────────────────┘
        └─────────────────────────┘       └─────────────────────────┘
```

## Feature Inventory

| # | Feature ID | Feature Name | Description | Milestone | Source |
|---|---|---|---|---|---|
| 1 | F-DATA-01 | Dataset Ingestion & Validation | Loads Kaggle Credit Card / Customer Segmentation dataset with schema validation & synthetic fallback | M1 | Survey |
| 2 | F-DATA-02 | Data Understanding & Stats | Computes statistical summaries, missingness percentages, skewness, correlation matrices, Hopkins statistic | M1 | Survey |
| 3 | F-DATA-03 | Missing Value Imputation | Automated imputation strategies: Median, Mean, KNN, MICE/Iterative | M1 | Survey |
| 4 | F-DATA-04 | Outlier Detection & Handling | Winsorization (quantile capping), IQR trimming, Isolation Forest filtering | M1 | Survey |
| 5 | F-DATA-05 | Feature Transformations | Yeo-Johnson power transform, Standard, Robust, and MinMax scaling | M1 | Survey |
| 6 | F-DATA-06 | Feature Engineering | Behavioral ratio derivation (Purchase-to-Limit, Cash-Advance-to-Limit, Payment-to-MinPayment) | M1 | Survey |
| 7 | F-DATA-07 | Dimensionality Reduction | 2D/3D coordinate projections using PCA, UMAP, and t-SNE | M1 | Survey |
| 8 | F-MOD-01 | Partitioning: K-Means | K-Means++ clustering with variable $k$, convergence tracking, and inertia calculation | M1 | Survey |
| 9 | F-MOD-02 | Partitioning: K-Medoids | Medoid-based clustering using FasterPAM / PAM with Manhattan/Euclidean distances | M1 | Survey |
| 10 | F-MOD-03 | Density-Based: DBSCAN | Density contour clustering with $\epsilon$ and `min_samples`, automatic noise separation | M1 | Survey |
| 11 | F-MOD-04 | Density-Based: HDBSCAN | Hierarchical density clustering with variable density support and GLOSH outlier scoring | M1 | Survey |
| 12 | F-MOD-05 | Hierarchical: Agglomerative | Agglomerative clustering with Ward, Complete, and Average linkages | M1 | Survey |
| 13 | F-MOD-06 | Probabilistic: GMM | Gaussian Mixture Models with Expectation-Maximization, covariance types (full, tied, diag, spherical) | M1 | Survey |
| 14 | F-EVAL-01 | Internal Validation Metrics | Silhouette score, Davies-Bouldin index, Calinski-Harabasz score | M1 | Survey |
| 15 | F-EVAL-02 | Inertia & Elbow Analysis | Inertia curve across $k \in [2, 15]$ with automated Kneedle elbow detection | M1 | Survey |
| 16 | F-EVAL-03 | Cluster Stability Analysis | Subsampling bootstrap stability computing mean Adjusted Rand Index (ARI) | M1 | Survey |
| 17 | F-EVAL-04 | Personas & Radar Profiles | Cluster-level centroid summaries, dominant behavioral archetypes, radar normalization | M1 | Survey |
| 18 | F-AUTO-01 | Search Space Parameterization | Formal configuration space $\Theta$ over preprocessing, feature flags, algorithm choice, and hyperparameters | M2 | Survey |
| 19 | F-AUTO-02 | Composite Fitness Function | Bounded multi-objective objective $F(\theta)$ balancing Silhouette, DB, CH, Stability, and Noise Penalties | M2 | Survey |
| 20 | F-AUTO-03 | Hill-Climbing Optimization Engine | Stochastic hill-climbing with step transitions, simulated annealing acceptance, tabu hash cache | M2 | Survey |
| 21 | F-AUTO-04 | Perturbation & Random Restarts | Neighborhood mutation operators and global random restarts upon convergence/plateau | M2 | Survey |
| 22 | F-AUTO-05 | Experiment Ledger & Ablations | JSONL/CSV structured experiment tracking, best-so-far trajectory logging, ablation export | M2 | Survey |
| 23 | F-RES-01 | Literature Alignment Synthesis | Academic literature mapping (Rousseeuw 1987, Davies-Bouldin 1979, Calinski-Harabasz 1974, Campello 2013, McInnes 2018, Sakana AI 2024) | M3 | Survey |
| 24 | F-RES-02 | Benchmark Matrix & Ablation | Comparative benchmark tables across all 6 models, baseline vs. hill-climbed ablation tables | M3 | Survey |
| 25 | F-API-01 | FastAPI Infrastructure & Docs | Structured FastAPI application with typed Pydantic V2 models, CORS, OpenAPI Swagger at `/docs` | M4 | Survey |
| 26 | F-API-02 | Data & EDA Endpoints | REST endpoints for dataset upload, summary statistics, correlation matrices, and distribution metrics | M4 | Survey |
| 27 | F-API-03 | Clustering & Profiling Endpoints | REST endpoints for model execution, cluster assignments, 2D/3D projections, silhouette samples, radar profiles | M4 | Survey |
| 28 | F-API-04 | Autoresearch & Streaming Endpoints | REST & SSE endpoints to trigger, pause, monitor, and stream real-time hill-climbing search progress | M4 | Survey |
| 29 | F-API-05 | Inference & Prediction Endpoint | REST endpoint to classify new single/batch customer vectors, returning cluster IDs, distances, and probabilities | M4 | Survey |
| 30 | F-UI-01 | Overview & CRISP-DM View | Interactive 6-phase CRISP-DM flow tracker, dataset health metrics, pipeline telemetry | M5 | Survey |
| 31 | F-UI-02 | Cluster Explorer View | Multi-model switcher, 2D/3D interactive projection canvas, cluster radar charts, silhouette distributions | M5 | Survey |
| 32 | F-UI-03 | Autoresearch Studio View | Live hill-climbing monitor, trajectory curves, parameter deltas, experiment leaderboard | M5 | Survey |
| 33 | F-UI-04 | Research Benchmark Matrix View | Paper-style comparative tables with LaTeX/Markdown export, standard deviation bounds, ablation charts | M5 | Survey |
| 34 | F-UI-05 | Inference Playground View | Interactive feature sliders, customer persona presets, real-time prediction, soft membership probabilities | M5 | Survey |

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| E2E | E2E Testing Suite Track | Test harness, synthetic dataset generation, Tier 1-4 test suites across all 34 features, publishing `TEST_READY.md` | none | **DONE** |
| M1 | CRISP-DM Clustering Pipeline | Data cleaning, preprocessing, 6 models across 4 paradigms, internal metrics, projections, personas | none | **DONE** |
| M2 | Autoresearch & Hill-Climbing Engine | Search space $\Theta$, composite fitness $F(\theta)$, hill-climber with restarts, experiment ledger | M1 | **DONE** |
| M3 | Research Paper Alignment & Benchmarks | Literature synthesis, benchmark matrix, baseline vs. hill-climbed ablation tables | M1, M2 | **DONE** |
| M4 | High-Performance FastAPI Backend | REST API endpoints, SSE stream, Pydantic schemas, OpenAPI docs, inference endpoint | M1, M2, M3 | **DONE** |
| M5 | Next.js + TypeScript Admin Dashboard | 5 views (CRISP-DM Flow, Cluster Explorer, Autoresearch Studio, Benchmark Matrix, Playground), mock fallback | M4 | **DONE** |
| M6 | Final Verification & Adversarial Hardening | Pass 100% of E2E test suite (Tiers 1-4) + Tier 5 whitebox adversarial testing + Clean Forensic Audit | E2E, M1-M5 | **DONE** |

## Interface Contracts

### Data Preprocessing & Model Interface (`src/crisp_dm/`)
- `PipelineConfig`: Preprocessing and feature engineering options (`imputer`, `outlier_method`, `scaler`, `power_transform`, `pca_components`).
- `ClusteringModelBase`: Abstract base class implementing `fit_predict(X) -> np.ndarray`, `get_params() -> dict`, `predict(X_new) -> np.ndarray`, `predict_proba(X_new) -> np.ndarray`.
- `ClusterResult`: Container with `labels`, `n_clusters`, `silhouette`, `davies_bouldin`, `calinski_harabasz`, `stability_ari`, `inertia`, `personas`, `projections_2d`, `projections_3d`.

### Autoresearch Interface (`src/autoresearch/`)
- `SearchSpace`: Parameter ranges for preprocessing, algorithms, and hyperparameters.
- `CompositeObjective`: Computes scalar fitness $F(\theta) = w_1 S + w_2 (1/(1+DB)) + w_3 CH_{norm} + w_4 ARI - P_{noise} - P_{imbalance}$.
- `HillClimbingOptimizer`: Methods `start(dataset)`, `step() -> StepLog`, `run(max_steps, callback)`, `get_history() -> List[StepLog]`, `export_ablations() -> dict`.

### REST API Schemas (`src/api/schemas/`)
- `DatasetSummaryResponse`: Shape, columns, missing values, skewness, Hopkins statistic.
- `ClusteringRequest`: Model name, parameters, preprocessing config.
- `ClusteringResponse`: Metrics, cluster counts, persona summaries, sample projections.
- `AutoresearchTriggerRequest`: Max iterations, restart threshold, target objective.
- `AutoresearchStepResponse`: Step index, candidate params, fitness delta, best fitness, accept flag, restart flag.
- `InferenceRequest`: Feature vector dictionary or CSV string.
- `InferenceResponse`: Assigned cluster, distances to centroids, soft probabilities, persona description.
