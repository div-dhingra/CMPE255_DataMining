# Original User Request

## 2026-08-28T08:16:30Z

An end-to-end CRISP-DM clustering and segmentation system built on a popular Kaggle benchmark dataset (Customer Segmentation / Credit Card data), driven by an autonomous autoresearch hill-climbing optimization engine that references clustering research literature, served via a high-performance FastAPI backend and visualized through a modern Next.js + TypeScript Data Science & AI Engineer Admin Dashboard.

Working directory: /Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3
Integrity mode: demo

## Requirements

### R1. CRISP-DM Clustering Pipeline
Implement the full CRISP-DM lifecycle (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment) for the dataset.
- Support multiple clustering paradigms: Partitioning (K-Means, K-Medoids), Density-based (DBSCAN, HDBSCAN), Hierarchical (Agglomerative), and Probabilistic (Gaussian Mixture Models).
- Provide automated preprocessing pipelines: missing value imputation, outlier detection, scaling/power transformations, and dimensionality reduction (PCA, UMAP, t-SNE).
- Compute comprehensive internal and external cluster evaluation metrics (Silhouette score, Davies-Bouldin index, Calinski-Harabasz score, cluster stability, inertia elbow analysis).
- Output detailed cluster personas and feature importance radar/profile summaries.

### R2. Autoresearch & Hill-Climbing Optimization Engine
An autonomous research and experimentation loop that iteratively searches hyperparameter spaces, feature sets, and preprocessing transformations to maximize cluster separation and stability.
- Maintain an experiment log tracking iteration history, configuration state, step transitions, metric improvements, and convergence criteria.
- Support hill-climbing search with restart/perturbation capabilities to escape local optima.
- Export benchmark comparisons and ablation logs for downstream analysis and visualization.

### R3. Research Paper Alignment & Benchmark Synthesis
Conduct literature research on clustering benchmarks and state-of-the-art evaluation methodologies.
- Provide a structured research synthesis linking experimental results to published clustering standards, algorithm trade-offs, and empirical findings.
- Generate paper-style comparative analysis tables and ablation studies comparing baseline vs. hill-climbed models.

### R4. FastAPI Analytical Backend
A structured REST API serving the data science backend:
- Endpoints for dataset ingestion, summary statistics, feature correlation matrices, and distribution metrics.
- Endpoints to trigger, monitor, and stream autoresearch hill-climbing runs.
- Endpoints for querying cluster profiles, 2D/3D projection coordinates, silhouette sample analyses, and metric comparisons.

### R5. Next.js & TypeScript Admin Dashboard
A modern, responsive Data Science & AI Engineering admin interface built with Next.js and TypeScript:
- **Overview & CRISP-DM Flow**: Visual breakdown of the 6 CRISP-DM phases and dataset health indicators.
- **Cluster Explorer**: Interactive 2D/3D projection plots (PCA/UMAP), cluster radar charts, silhouette sample plots, and per-cluster feature distributions.
- **Autoresearch Studio**: Real-time / interactive hill-climbing experiment monitor showing step trajectories, convergence curves, parameter deltas, and experiment leaderboard.
- **Research Benchmark Matrix**: Paper-style tables comparing algorithms across metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, Runtime, Stability).
- **Inference & Profiler Playground**: Interactive tool to pass new data points and inspect cluster assignment probabilities and segment interpretations.

---

## Acceptance Criteria

### Data & Pipeline Verification
- [ ] Dataset loads cleanly with full automated data cleaning, missing value handling, and feature scaling verified by automated pipeline tests.
- [ ] At least 4 distinct clustering algorithms (including K-Means, DBSCAN/HDBSCAN, Agglomerative, GMM) execute and produce reproducible clustering outputs.
- [ ] Evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) are computed and logged for all algorithms.

### Autoresearch Verification
- [ ] Autoresearch engine runs hill-climbing iterations, successfully tracks metric progression across steps, and outputs a structured experiment history JSON/CSV.
- [ ] The best configuration found by hill climbing achieves a measurable improvement in the target clustering validation objective over the default baseline.

### Backend API Verification
- [ ] FastAPI service runs and passes automated endpoint tests (health check, dataset stats, clustering execution/results, autoresearch status, and cluster profiling endpoints).
- [ ] All API responses conform to typed Pydantic models with OpenAPI/Swagger documentation generated at `/docs`.

### Frontend Dashboard Verification
- [ ] Next.js + TypeScript frontend builds cleanly with zero TypeScript compilation or lint errors (`npm run build`).
- [ ] Dashboard contains functional views for CRISP-DM overview, interactive cluster visualization, autoresearch experiment tracking, and research paper benchmarking.

### Documentation & Reproducibility
- [ ] Project includes automated test suites covering pipeline logic and API endpoints.
- [ ] Comprehensive documentation detailing CRISP-DM phases, architecture, research paper references, and startup instructions for both backend and frontend.
