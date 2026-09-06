# CRISP-DM Autonomous Clustering & Autoresearch Engine

This document serves as the formal Implementation Plan detailing the architecture, components, and evaluation strategy for the end-to-end clustering project using the Kaggle Customer Segmentation dataset.

## Proposed Changes (System Architecture)

The system is separated into a high-performance Python FastAPI backend and a Next.js TypeScript frontend dashboard.

### Backend Infrastructure (`backend/`)

#### [NEW] `src/crisp_dm/`
Implementation of the standard data mining lifecycle.
- `data_understanding.py` & `data_preparation.py`: Handles ingestion, missing value imputation (Median, KNN, MICE), outlier capping, and feature scaling/transformations (Yeo-Johnson).
- `models/`: Modular wrappers for 6 algorithms across 4 clustering paradigms: K-Means, K-Medoids, DBSCAN, HDBSCAN, Agglomerative, and Gaussian Mixture Models (GMM).
- `evaluation.py`: Computes validation metrics including Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, and bootstrap stability (ARI).
- `projections.py` & `profiling.py`: Generates 2D/3D coordinates (PCA, UMAP, t-SNE) and cluster persona narratives (Radar profiles).

#### [NEW] `src/autoresearch/`
Autonomous hill-climbing optimization engine.
- `search_space.py`: Defines the bounded configuration space $\Theta$ over preprocessing flags and model hyperparameters.
- `objective.py`: Computes the composite fitness function balancing separation, stability, and noise penalty.
- `optimizer.py`: The stochastic hill-climbing loop with simulated annealing acceptance and random restarts to escape local optima.

#### [NEW] `src/api/`
FastAPI REST interfaces.
- `routes/data.py` & `routes/cluster.py`: Endpoints for fetching stats and executing clustering pipelines.
- `routes/autoresearch.py`: Server-Sent Events (SSE) stream for monitoring live optimization trajectories.
- `routes/inference.py`: Endpoint for scoring new customer vectors into soft cluster probabilities.

### Frontend Dashboard (`frontend/`)

#### [NEW] `src/app/`
Next.js web views mapping to core features.
- `overview/page.tsx`: CRISP-DM 6-phase tracker and dataset health telemetry.
- `clusters/page.tsx`: Interactive 2D/3D scatter projections and silhouette ribbon charts.
- `autoresearch/page.tsx`: Real-time trajectory monitor for the hill-climbing engine.
- `benchmarks/page.tsx`: Academic literature alignment matrix comparing baseline vs optimized configurations.
- `playground/page.tsx`: 17 interactive sliders for real-time customer cluster prediction.

## Verification Plan

### Automated Tests
- Execution of the full E2E canonical test suite using pytest: `bash run_tests.sh`
- Component tests spanning boundaries, missing data handling, and feature interactions (Tier 1-4 tests).

### Manual Verification
- Start FastAPI backend (`uvicorn src.api.main:app`) and Next.js frontend (`npm run dev`).
- Verify the Autoresearch Studio successfully iterates and improves the composite fitness score over time.
- Verify 3D interactions and radar chart rendering on the Cluster Explorer view.
