# CRISP-DM Clustering & Autoresearch System
## Authoritative Technical Specification & Requirements Baseline

**Document Version:** 1.0.0  
**Author:** teamwork_preview_spec_miner (Specification Mining Agent)  
**Date:** 2026-08-28  
**Working Directory:** `/Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3`  
**Status:** Approved Specification Baseline

---

## 1. Executive Summary & Scope

The **CRISP-DM Clustering & Autoresearch System** is an end-to-end data mining and machine learning platform engineered to discover actionable customer segments on high-dimensional behavioral datasets (specifically the popular Kaggle Credit Card / Customer Segmentation benchmark). 

The platform integrates three core capabilities:
1. **CRISP-DM Industrial Lifecycle**: Complete data mining lifecycle executing Business Understanding, Data Understanding, Data Preparation, Multi-Paradigm Modeling (Partitioning, Density-Based, Hierarchical, Probabilistic), Multi-Metric Evaluation, and Deployment.
2. **Autonomous Autoresearch Hill-Climbing Engine**: A self-driving metaheuristic optimization engine that navigates high-dimensional pipeline search spaces (preprocessing transformations, feature combinations, algorithm families, and hyperparameters) to maximize composite cluster quality, stability, and separation while logging full telemetry.
3. **Full-Stack Analytical & Engineering Dashboard**: A high-throughput, typed FastAPI backend coupled with a modern Next.js + TypeScript engineering interface providing interactive 2D/3D projections, silhouette ribbon diagnostics, persona radar charts, real-time autoresearch monitoring, and academic paper-style benchmark matrices.

---

## 2. Requirements Analysis (R1 to R5 & Acceptance Criteria)

### R1. CRISP-DM Clustering Pipeline
- **R1.1 Business Understanding**: Define customer segmentation objectives (transactors, revolvers, cash-advance heavy, inactive, VIP spenders) and translate them into measurable data mining KPIs.
- **R1.2 Data Understanding**: Ingest the 18-attribute customer dataset (`CUST_ID`, `BALANCE`, `BALANCE_FREQUENCY`, `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `CASH_ADVANCE`, `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX`, `PURCHASES_TRX`, `CREDIT_LIMIT`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT`, `TENURE`). Perform descriptive statistical profiling, kurtosis/skewness detection, missing value quantification, and Pearson/Spearman correlation analysis.
- **R1.3 Data Preparation**: Automated, configurable preprocessing pipelines:
  - *ID Handling*: Exclusion of non-informative identifiers (`CUST_ID`).
  - *Imputation*: Median, Mean, or KNN Imputation for missing values (`MINIMUM_PAYMENTS`, `CREDIT_LIMIT`).
  - *Outlier Mitigation*: Configurable IQR truncation (Winsorization) or Isolation Forest filtering.
  - *Scaling & Transforms*: `StandardScaler`, `RobustScaler`, `MinMaxScaler`, and `PowerTransformer` (Yeo-Johnson) for heavy-tailed distributions.
  - *Dimensionality Reduction*: PCA (2D, 3D, and scree variance cutoff), UMAP, and t-SNE coordinate embedding for visualization and noise reduction.
- **R1.4 Modeling**: Multi-paradigm clustering coverage across 4 distinct paradigms and 6 algorithms:
  - *Partitioning*: K-Means (k-means++ initialization, Lloyd/Elkan algorithms), K-Medoids (PAM / FasterPAM with Manhattan/Euclidean metrics).
  - *Density-Based*: DBSCAN (density reachability with $\epsilon$ and $min\_samples$), HDBSCAN (hierarchical density estimates with variable density and soft noise assignment).
  - *Hierarchical*: Agglomerative Clustering (Ward, Complete, Average, Single linkage trees).
  - *Probabilistic*: Gaussian Mixture Models (EM algorithm with Full, Tied, Diagonal, and Spherical covariance structures, BIC/AIC model selection).
- **R1.5 Evaluation**: Exhaustive internal and stability validation metrics:
  - Silhouette Coefficient (global mean and per-cluster sample profiles).
  - Davies-Bouldin Index (cluster separation ratio).
  - Calinski-Harabasz Index (Variance Ratio Criterion).
  - Within-Cluster Sum of Squares (Inertia) & Automated Kneedle Elbow Detection.
  - Cluster Stability Index via bootstrap resampling and noise perturbation under Adjusted Rand Index (ARI).
  - Hopkins Statistic for spatial clustering tendency.
- **R1.6 Deployment & Persona Profiling**: Cluster persona extraction (centroid/median feature profiles, relative z-score deviations, persona naming heuristics) and real-time inference scoring.

### R2. Autoresearch & Hill-Climbing Optimization Engine
- **R2.1 Search Space Formulation**: Parametrize the joint space $\Theta = \mathcal{P}_{prep} \times \mathcal{A}_{alg} \times \mathcal{H}_{hyper}$.
- **R2.2 Objective Function**: Formulate a normalized composite fitness metric $F(\theta) \in [0, 1]$ balancing Silhouette, Davies-Bouldin, Calinski-Harabasz, and Cluster Stability with penalty terms for excessive noise or severe cluster size imbalance.
- **R2.3 Search Metaheuristic**:
  - Implement First-Choice / Steepest-Ascent Hill Climbing with neighborhood mutation operators.
  - Random Restarts upon reaching local optima (stagnation for $M$ steps).
  - Stochastic Perturbation / Simulated Annealing cooling step ($P = \exp(\Delta F / T)$) to escape shallow basins.
- **R2.4 Experiment Telemetry & State Tracking**: Log every step (iteration, timestamp, configuration vector, proposed mutated state, acceptance status, metric vector, execution latency, best-so-far candidate).
- **R2.5 Artifact Generation**: Export comprehensive experiment trajectories (`autoresearch_log.json`, `leaderboard.json`, `ablation_table.json`).

### R3. Research Paper Alignment & Benchmark Synthesis
- **R3.1 Methodological Grounding**: Reference peer-reviewed literature (Rousseeuw 1987, Davies-Bouldin 1979, Calinski-Harabasz 1974, Campello et al. 2013, Arthur-Vassilvitskii 2007, Von Luxburg 2010, Satopaa et al. 2011).
- **R3.2 Benchmark Synthesis Matrix**: Generate academic-style comparison tables evaluating default baselines vs. Autoresearch-optimized configurations across all paradigms.
- **R3.3 Ablation Studies**: Provide systematic ablation tables isolating the performance contributions of preprocessing transformations, dimensionality reduction, and hyperparameter tuning.

### R4. FastAPI Analytical Backend
- **R4.1 Architecture & Performance**: Asynchronous, typed FastAPI application with modular routers, CORS support, OpenAPI/Swagger auto-documentation, and low-latency JSON serialization.
- **R4.2 Endpoints**:
  - `/api/v1/health`: System health and model cache status.
  - `/api/v1/dataset/*`: Data ingestion, statistical summaries, correlation matrices, feature distributions.
  - `/api/v1/clustering/*`: Model execution, 2D/3D projection coordinates, cluster personas, silhouette sample ribbons, elbow inertia curves.
  - `/api/v1/autoresearch/*`: Asynchronous/synchronous experiment start, live status/step polling, abort execution, and leaderboard queries.
  - `/api/v1/benchmark/*`: Comparative benchmark matrix and ablation data.
  - `/api/v1/inference/predict`: Real-time single/batch inference returning cluster assignment, soft membership probabilities, and persona card.

### R5. Next.js & TypeScript Admin Dashboard
- **R5.1 UI/UX Architecture**: Next.js 14+ App Router, TypeScript, Tailwind CSS, Lucide icons, responsive layout with dark/light themes.
- **R5.2 Views**:
  - **Overview & CRISP-DM Lifecycle View (`/`)**: Visual 6-phase interactive flow diagram, dataset health metrics, missingness audit, feature distribution histograms, and correlation heatmaps.
  - **Cluster Explorer View (`/clusters`)**: Interactive 2D/3D scatter plots (PCA/UMAP toggle), silhouette plot with per-cluster ribbons, feature importance radar charts, cluster persona cards.
  - **Autoresearch Studio View (`/autoresearch`)**: Real-time experiment monitor, step trajectory timeline, convergence curves ($F(\theta)$, Silhouette, DB, CH vs. step), parameter diff viewer, and experiment leaderboard.
  - **Research Benchmark Matrix View (`/benchmarks`)**: Academic paper-style comparative tables, baseline vs. optimized deltas, ablation matrix, and literature bibliography.
  - **Inference & Profiler Playground (`/playground`)**: Interactive input forms/sliders for customer parameters, real-time prediction calling, soft probability distributions, and persona narrative summaries.

---

## Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|---|---|---|---|---|---|---|
| 1 | Data Understanding | Dataset Ingestion & Validation | Loads raw customer data, validates column schemas, checks data types and row integrity. | CSV file path / stream | Clean DataFrame & ingestion metadata | Raises `DataIngestionError` on missing required columns. | ORIGINAL_REQUEST R1.2 |
| 2 | Data Understanding | Statistical Profiling & Distribution Analysis | Computes mean, std, median, IQR, skewness, kurtosis, and missing value rates. | Clean DataFrame | Summary stats dictionary & histogram bins | Returns warning on zero-variance columns. | ORIGINAL_REQUEST R1.2 |
| 3 | Data Understanding | Correlation & Covariance Matrix | Calculates Pearson and Spearman correlation matrices across all behavioral attributes. | Clean DataFrame | $17 \times 17$ correlation matrix | Replaces NaN correlations with 0.0. | ORIGINAL_REQUEST R1.2 |
| 4 | Data Preparation | Missing Value Imputation | Imputes missing values in `MINIMUM_PAYMENTS` and `CREDIT_LIMIT` using Median, Mean, or KNN. | Raw feature matrix, strategy | Fully imputed feature matrix | Raises error if entire column is NaN. | ORIGINAL_REQUEST R1.3 |
| 5 | Data Preparation | Outlier Detection & Capping | Detects and treats extreme anomalies via IQR Winsorization or Isolation Forest. | Imputed matrix, method, threshold | Outlier-bounded matrix & outlier mask | Falls back to no capping if threshold $\le 0$. | ORIGINAL_REQUEST R1.3 |
| 6 | Data Preparation | Multi-Strategy Feature Scaling | Standardizes or transforms skewed distributions into normalized scales (Standard, Robust, Yeo-Johnson). | Cleaned matrix, scaler type | Scaled feature matrix ($N \times D$) | Raises `ScalingError` on infinite values. | ORIGINAL_REQUEST R1.3 |
| 7 | Data Preparation | Dimensionality Reduction | Projects high-dimensional feature space into 2D/3D coordinates for visualization & clustering. | Scaled matrix, method (PCA/UMAP/t-SNE) | 2D/3D coordinates, explained variance ratios | Falls back to PCA if UMAP graph fails to converge. | ORIGINAL_REQUEST R1.3 |
| 8 | Modeling | K-Means Partitioning | Partitions data into $k$ spherical Voronoi cells using k-means++ seeding. | Scaled matrix, $k$, max_iter, random_state | Cluster labels ($0 \dots k-1$), centroids | Error if $k < 2$ or $k > N$. | ORIGINAL_REQUEST R1.4 |
| 9 | Modeling | K-Medoids Partitioning | Partitions data using actual exemplar data points as medoids (PAM/FasterPAM). | Scaled matrix, $k$, metric | Cluster labels, medoid indices | Error on non-symmetric distance matrix. | ORIGINAL_REQUEST R1.4 |
| 10 | Modeling | DBSCAN Density Clustering | Finds arbitrarily shaped clusters and labels sparse points as noise (-1). | Scaled matrix, $\epsilon$, min_samples | Cluster labels (including -1 for noise) | Warns if all points labeled noise. | ORIGINAL_REQUEST R1.4 |
| 11 | Modeling | HDBSCAN Hierarchical Density | Computes cluster hierarchy across varying densities and extracts stable clusters. | Scaled matrix, min_cluster_size, min_samples | Cluster labels, cluster probabilities, outlier scores | Soft assignment for unassigned noise. | ORIGINAL_REQUEST R1.4 |
| 12 | Modeling | Agglomerative Hierarchical | Builds hierarchical bottom-up cluster tree with Ward, Complete, or Average linkage. | Scaled matrix, $k$, linkage | Cluster labels, linkage matrix | Error if metric != 'euclidean' with Ward. | ORIGINAL_REQUEST R1.4 |
| 13 | Modeling | Gaussian Mixture Models | Fits $k$ multivariate Gaussian distributions via Expectation-Maximization. | Scaled matrix, $k$, covariance_type | Cluster labels, posterior probabilities, BIC/AIC | Regularizes covariance to prevent singularity. | ORIGINAL_REQUEST R1.4 |
| 14 | Evaluation | Silhouette Analysis | Computes cohesion vs. separation silhouette coefficients globally and per sample. | Scaled matrix, cluster labels | Global Silhouette score, per-sample silhouette array | Undefined for $k=1$ or all noise; returns 0.0. | ORIGINAL_REQUEST R1.5 |
| 15 | Evaluation | Davies-Bouldin Index | Evaluates similarity between each cluster and its most similar counterpart. | Scaled matrix, cluster labels | Davies-Bouldin score ($DB \ge 0$) | Returns infinity/error if cluster has 0 variance. | ORIGINAL_REQUEST R1.5 |
| 16 | Evaluation | Calinski-Harabasz Index | Ratio of between-cluster dispersion to within-cluster dispersion (Variance Ratio). | Scaled matrix, cluster labels | Calinski-Harabasz score ($CH \ge 0$) | Undefined for $k=1$; returns 0.0. | ORIGINAL_REQUEST R1.5 |
| 17 | Evaluation | Inertia & Kneedle Elbow Analysis | Computes within-cluster sum of squares over $k \in [2, 10]$ and detects knee point. | Scaled matrix, range of $k$ | Inertia curve array, optimal $k$ elbow | Returns lowest $k$ if curve is strictly linear. | ORIGINAL_REQUEST R1.5 |
| 18 | Evaluation | Cluster Stability Analysis | Assesses cluster robustness under bootstrap subsampling and Gaussian jitter. | Scaled matrix, pipeline config, n_bootstraps | Stability index $\in [0, 1]$ | Warns if bootstrap sample fails to converge. | ORIGINAL_REQUEST R1.5 |
| 19 | Evaluation | Cluster Persona & Profiler | Computes feature centroid means, relative deviation from global mean, and persona tags. | Original feature matrix, cluster labels | Cluster persona cards, radar chart dimensions | Handles noise cluster (-1) as separate segment. | ORIGINAL_REQUEST R1.6 |
| 20 | Autoresearch | Search Space Parametrization | Encodes pipeline hyperparameter choices and continuous ranges into discrete/driftable state vectors. | Search config definition | Config dictionary & parameter bounds | Validates constraint dependencies per algorithm. | ORIGINAL_REQUEST R2.1 |
| 21 | Autoresearch | Composite Objective Evaluator | Evaluates pipeline candidate and computes normalized composite score $F(\theta)$. | Feature matrix, candidate state $\theta$ | Float fitness score $F \in [0, 1]$, metrics breakdown | Returns $F=0.0$ if pipeline execution fails. | ORIGINAL_REQUEST R2.2 |
| 22 | Autoresearch | Hill-Climbing Optimization Engine | Executes iterative steepest-ascent search with mutation, restarts, and perturbation. | Search space, initial state, max_iter, patience | Complete execution log, best config, metric trajectory | Catches exceptions per candidate and proceeds to next neighbor. | ORIGINAL_REQUEST R2.3 |
| 23 | Autoresearch | Experiment Telemetry & Persistence | Tracks iteration history, step transitions, metric improvements, and serializes to JSON. | Autoresearch state stream | `autoresearch_log.json`, `leaderboard.json` | Flushes logs to disk incrementally on each step. | ORIGINAL_REQUEST R2.4 |
| 24 | Benchmarks | Research Synthesis & Literature Alignment | Formulates comparative performance tables linking empirical results to literature benchmarks. | Benchmark execution results | Academic benchmark comparison matrix | Displays N/A for metrics not supported by specific models. | ORIGINAL_REQUEST R3.1, R3.2 |
| 25 | Benchmarks | Systematic Ablation Engine | Evaluates pipeline components in isolation to quantify marginal utility of each stage. | Dataset, baseline config, ablation factors | Ablation summary table and delta bar chart | Handles invalid pipeline combinations safely. | ORIGINAL_REQUEST R3.3 |
| 26 | Backend API | REST Dataset & Statistics Endpoints | Provides endpoints for dataset health, summary stats, distributions, and correlation matrices. | HTTP GET requests | JSON response (`DatasetSummaryResponse`, `CorrelationResponse`) | 404 on missing dataset; 500 on calculation error. | ORIGINAL_REQUEST R4.2 |
| 27 | Backend API | REST Clustering Execution Endpoints | Runs clustering models on-demand and returns labels, metrics, projections, and personas. | HTTP POST `{algorithm, config}` | JSON response (`ClusteringRunResponse`, `ClusterProfileResponse`) | 422 on invalid parameters; 400 on clustering failure. | ORIGINAL_REQUEST R4.2 |
| 28 | Backend API | REST Autoresearch Engine Endpoints | Triggers, monitors, queries status, and aborts autoresearch optimization jobs. | HTTP POST/GET `{max_iter, weights, patience}` | JSON response (`AutoresearchStartResponse`, `StatusResponse`) | 404 on invalid job ID; 409 if job already running. | ORIGINAL_REQUEST R4.2 |
| 29 | Backend API | REST Real-Time Inference Endpoint | Evaluates an arbitrary customer feature vector and predicts cluster label & persona card. | HTTP POST `{features: Dict[str, float]}` | JSON response (`InferenceResponse`) | 422 on missing/invalid numeric fields. | ORIGINAL_REQUEST R4.2 |
| 30 | Frontend UI | CRISP-DM Flow & Dataset Health View | Visual interactive breakdown of 6 CRISP-DM stages with live dataset metrics and distributions. | User navigation (`/`) | Rendered React Dashboard page | Displays fallback error alert if API is unreachable. | ORIGINAL_REQUEST R5.2 |
| 31 | Frontend UI | Interactive 2D/3D Cluster Explorer | High-performance interactive scatter plot with PCA/UMAP toggles, silhouette ribbons, and radars. | User selection, API projections | Rendered Cluster Explorer page (`/clusters`) | Renders placeholder if no model has been executed. | ORIGINAL_REQUEST R5.2 |
| 32 | Frontend UI | Autoresearch Studio & Live Monitor | Real-time trajectory monitor displaying fitness convergence, parameter diffs, and leaderboard. | Autoresearch API polling stream (`/autoresearch`) | Rendered Autoresearch Studio page | Displays reconnecting badge on network blips. | ORIGINAL_REQUEST R5.2 |
| 33 | Frontend UI | Research Benchmark Matrix View | Academic paper-style comparative table with baseline vs. hill-climbed deltas & citations. | Benchmark API data (`/benchmarks`) | Rendered Benchmark Matrix page | Renders empty state guide if benchmarks not yet generated. | ORIGINAL_REQUEST R5.2 |
| 34 | Frontend UI | Customer Profiler & Inference Playground | Interactive input sliders/forms to test arbitrary customer profiles with instant segment feedback. | User input values (`/playground`) | Rendered Playground page with persona badge & radar | Displays input validation warnings on out-of-range inputs. | ORIGINAL_REQUEST R5.2 |

---

## Edge Cases

| # | Feature | Input | Observed Behavior |
|---|---|---|---|
| 1 | Silhouette Analysis (`FEAT-EVAL-01`) | $k=1$ (single cluster or all points assigned to same cluster) | Silhouette score is mathematically undefined ($b(i)$ does not exist). Evaluator returns 0.0 with warning log. |
| 2 | DBSCAN Density Clustering (`FEAT-CLUST-03`) | $\epsilon = 0.01$, $\text{min\_samples} = 50$ (overly restrictive) | All dataset points classified as noise (label $-1$). Pipeline sets $F(\theta) = 0.0$ and applies full noise penalty. |
| 3 | Feature Scaling (`FEAT-PIPE-03`) | Constant / zero-variance feature column | Zero division during standardization $\frac{x-\mu}{\sigma}$. Imputer/scaler imputes $\sigma=1.0$ or drops zero-variance columns. |
| 4 | Gaussian Mixture Models (`FEAT-CLUST-06`) | Highly collinear features or tiny cluster size | Covariance matrix $\Sigma_k$ becomes singular/non-invertible. EM fails without covariance regularization (`reg_covar=1e-6`). |
| 5 | Dimensionality Reduction (`FEAT-PIPE-04`) | UMAP with small disconnected components | UMAP raises disconnected nearest-neighbor graph warning. System adds small epsilon connectivity or falls back to PCA. |
| 6 | Missing Value Imputation (`FEAT-PIPE-01`) | Entire column has missing values (100% NaN) | Imputation fails. Pipeline validation rejects dataset with descriptive schema error. |
| 7 | Real-Time Inference (`FEAT-API-04`) | Incomplete user input vector (e.g. missing `MINIMUM_PAYMENTS`) | API uses pipeline's pre-fitted imputer (`SimpleImputer(strategy='median')`) to safely impute before scaling and scoring. |
| 8 | Autoresearch Hill-Climber (`FEAT-AUTO-03`) | Search trapped in local plateau ($\Delta F = 0$ for $P$ steps) | Triggers Random Restart by sampling a new random point in parameter space $\Theta$ and resets patience counter. |

---

## 3. CRISP-DM Lifecycle Detailed Specification

### Phase 1: Business Understanding
- **Objective**: Identify credit card customer behavioral archetypes to drive data-informed credit limits, APR incentives, cash-back marketing, and churn risk mitigation.
- **Key Personas Defined**:
  1. *Transactors (Active Spenders)*: High purchase frequency, high one-off purchases, low cash advance, high percentage of full payment.
  2. *Revolvers (Interest Payers)*: High balance, low purchase frequency, low percentage of full payment, moderate cash advance.
  3. *Cash Advance Users (Liquidity Seekers)*: High cash advance frequency and transactions, high balance, low purchases.
  4. *Low-Engagement / Inactive*: Low balance, low purchases, low payments, near-zero transaction counts.
  5. *VIP / High-Limit Spenders*: High credit limit, high payments, high total purchases.

### Phase 2: Data Understanding
- **Dataset Structure**: 8,950 records, 18 features (`CUST_ID` + 17 numerical behavioral features).
- **Missing Value Profile**:
  - `CREDIT_LIMIT`: 1 missing record ($0.01\%$).
  - `MINIMUM_PAYMENTS`: 313 missing records ($3.50\%$).
- **Distributional Skewness**: Attributes like `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, and `PAYMENTS` exhibit extreme positive skewness ($> 3.0$), requiring power transformations (Yeo-Johnson) or robust scaling.

### Phase 3: Data Preparation
- **Pipeline Stages**:
  1. Column filtering: Remove non-predictive `CUST_ID`.
  2. Imputation: Strategy $\in \{\text{'median'}, \text{'mean'}, \text{'knn'}\}$.
  3. Outlier handling: Winsorization capping at $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$ or Isolation Forest.
  4. Transformation / Scaling:
     - `StandardScaler`: $z = (x - \mu) / \sigma$
     - `RobustScaler`: $z = (x - \text{median}) / \text{IQR}$
     - `PowerTransformer` (Yeo-Johnson): Log/power monotonic transformation stabilizing variance.
  5. Projection / Dimensionality Reduction: PCA (2D/3D), UMAP (2D/3D), t-SNE (2D).

### Phase 4: Modeling Paradigms & Formulations
1. **Partitioning**:
   - **K-Means**: $\min_{S} \sum_{i=1}^k \sum_{x \in S_i} \|x - \mu_i\|^2$ with k-means++ seeding.
   - **K-Medoids**: $\min_{S} \sum_{i=1}^k \sum_{x \in S_i} d(x, m_i)$ with exemplar medoids $m_i \in X$.
2. **Density-Based**:
   - **DBSCAN**: Core density reachability with parameters $(\epsilon, min\_samples)$.
   - **HDBSCAN**: Condensed hierarchical trees over mutual reachability distance $d_{\text{mreach-}k}(a, b) = \max(\text{core}_k(a), \text{core}_k(b), d(a,b))$.
3. **Hierarchical**:
   - **Agglomerative Clustering**: Ward variance minimization ($\Delta ESS_{AB} = \frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2$), Complete, and Average linkage.
4. **Probabilistic**:
   - **Gaussian Mixture Models (GMM)**: Soft mixture modeling $p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$ via EM.

### Phase 5: Evaluation Standards & Formulas
1. **Silhouette Coefficient**:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad S = \frac{1}{N}\sum_{i=1}^N s(i) \in [-1, 1]$$
2. **Davies-Bouldin Index**:
   $$DB = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} \left(\frac{s_i + s_j}{d(\mu_i, \mu_j)}\right) \ge 0 \quad (\text{lower is better})$$
3. **Calinski-Harabasz Index (Variance Ratio Criterion)**:
   $$CH = \frac{\text{Tr}(B_k)/(k-1)}{\text{Tr}(W_k)/(n-k)} \ge 0 \quad (\text{higher is better})$$
4. **Inertia & Kneedle Elbow Detection**:
   $$\text{Inertia}(k) = \sum_{j=1}^k \sum_{x \in C_j} \|x - \mu_j\|^2, \quad k^* = \arg\max_k D(k)$$
5. **Cluster Stability Index**:
   $$\text{Stability} = \frac{1}{B}\sum_{b=1}^B \text{ARI}(Y_{\text{orig}}, Y_{\text{boot}, b}) \in [0, 1]$$
6. **Hopkins Statistic**:
   $$H = \frac{\sum u_i}{\sum u_i + \sum w_i} \in [0, 1]$$

---

## 4. Autoresearch & Hill-Climbing Optimization Engine

### Search Space Definition
$$\Theta = \mathcal{P}_{\text{impute}} \times \mathcal{P}_{\text{outlier}} \times \mathcal{P}_{\text{scale}} \times \mathcal{P}_{\text{dimred}} \times \mathcal{A} \times \mathcal{H}(\mathcal{A})$$

### Normalized Composite Objective Function $F(\theta)$
$$F(\theta) = w_1 S_{\text{norm}}(\theta) + w_2 (1 - DB_{\text{norm}}(\theta)) + w_3 CH_{\text{norm}}(\theta) + w_4 \text{Stability}(\theta) - P_{\text{noise}}(\theta) - P_{\text{imbalance}}(\theta)$$
Where:
- $S_{\text{norm}} = \frac{S + 1}{2}$
- $DB_{\text{norm}} = \min(1.0, DB / 5.0)$
- $CH_{\text{norm}} = \min\left(1.0, \frac{\ln(1 + CH)}{\ln(1 + 10000)}\right)$
- $P_{\text{noise}} = 0.5 \times \max(0, \text{noise\_frac} - 0.10)$
- $P_{\text{imbalance}} = 0.2 \times \left(1 - \frac{\mathcal{H}(\text{sizes})}{\ln(k)}\right)$
- Default weights: $w_1 = 0.40, w_2 = 0.25, w_3 = 0.15, w_4 = 0.20$.

### Hill-Climbing Optimization Metaheuristic
1. **Neighborhood Mutation**: Mutate one continuous parameter by Gaussian drift ($\pm 10\%$) or toggle one discrete choice (e.g. Scaler: Standard $\to$ Yeo-Johnson).
2. **First-Choice / Steepest Ascent**: Evaluate mutation candidate $F(\theta')$.
3. **Simulated Annealing Acceptance**: If $\Delta F = F(\theta') - F(\theta) > 0$, accept; else accept with probability $P = \exp(\Delta F / T)$.
4. **Stagnation Detection & Random Restart**: If no improvement after $P$ steps, jump to a random configuration $\theta_{\text{rand}} \in \Theta$ and reset patience.
5. **Telemetry Logging**: Emits step JSON `{iteration, state, proposed, delta_F, metrics, latency_ms, accepted}`.

---

## 5. REST API Endpoints & Pydantic Data Contracts

### Endpoint Map

| HTTP Method | Route | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health, dataset status, and active model cache |
| `GET` | `/api/v1/dataset/summary` | Descriptive statistics (mean, std, IQR, skewness, kurtosis) |
| `GET` | `/api/v1/dataset/correlation` | Pearson and Spearman correlation matrices |
| `GET` | `/api/v1/dataset/distributions` | Histogram & KDE data per feature |
| `POST` | `/api/v1/clustering/run` | Execute custom clustering pipeline |
| `GET` | `/api/v1/clustering/projections` | 2D/3D projection coordinates (PCA/UMAP) |
| `GET` | `/api/v1/clustering/profiles` | Cluster personas and radar chart profiles |
| `GET` | `/api/v1/clustering/silhouette-samples` | Per-sample silhouette ribbon arrays |
| `GET` | `/api/v1/clustering/elbow` | WCSS inertia and elbow curve |
| `POST` | `/api/v1/autoresearch/start` | Launch autonomous hill-climbing run |
| `GET` | `/api/v1/autoresearch/status/{job_id}` | Live poll experiment trajectory and steps |
| `POST` | `/api/v1/autoresearch/stop/{job_id}` | Terminate running autoresearch task |
| `GET` | `/api/v1/autoresearch/leaderboard` | Ranked top-performing configurations |
| `GET` | `/api/v1/benchmark/matrix` | Comparative research benchmark table |
| `GET` | `/api/v1/benchmark/ablation` | Preprocessing & parameter ablation breakdown |
| `POST` | `/api/v1/inference/predict` | Real-time customer cluster assignment |

---

## 6. Next.js & TypeScript Admin Dashboard Specification

### Routing & UI View Architecture
1. **CRISP-DM Flow & Dataset Health (`/`)**:
   - Interactive SVG flow diagram of 6 CRISP-DM stages.
   - Dataset health cards: Row count (8,950), missing value audit, outlier summary.
   - Feature distribution histograms with KDE toggle.
   - Interactive Pearson/Spearman correlation heatmap.
2. **Cluster Explorer (`/clusters`)**:
   - Interactive 2D/3D Canvas/WebGL scatter plot with PCA/UMAP toggle.
   - Silhouette ribbon plot with global mean line.
   - Multi-axis persona radar chart.
   - Per-cluster descriptive statistics table.
3. **Autoresearch Studio (`/autoresearch`)**:
   - Real-time trajectory charts ($F(\theta)$, Silhouette, DB, CH vs. iteration).
   - Visual step parameter delta diff viewer.
   - Interactive control panel (Start, Pause, Abort, Weight sliders).
   - Live experiment leaderboard with "Deploy to Production" action.
4. **Research Benchmark Matrix (`/benchmarks`)**:
   - Academic paper-style comparative table with sortable columns.
   - Preprocessing & parameter ablation delta cards.
   - Research bibliography with direct paper references.
5. **Inference & Profiler Playground (`/playground`)**:
   - Slider and numeric form for customer financial parameters.
   - Preset customer archetypes (Transactor, Revolver, Cash Advance, VIP).
   - Instant segment classification badge, radar chart, and persona card.

---

## 7. Verification & Acceptance Criteria Standards

| Verification Track | Acceptance Criteria Target | Validation Standard |
|---|---|---|
| **Data & Pipeline** | Dataset loads cleanly; 0 NaNs after pipeline; scaling verified. | `pytest tests/test_pipeline.py` passes 100%. |
| **Clustering Algorithms** | All 6 algorithms execute and produce valid labels $\in [-1, K-1]$. | `pytest tests/test_clustering.py` passes 100%. |
| **Evaluation Metrics** | Silhouette, DB, CH, Inertia, and Stability match standard formulas. | `pytest tests/test_evaluation.py` passes 100%. |
| **Autoresearch Engine** | Hill-climbing runs, logs telemetry, achieves $F(\theta^*) > F(\theta_0)$. | `pytest tests/test_autoresearch.py` passes 100%. |
| **FastAPI Backend** | All 16 endpoints return valid Pydantic responses with HTTP 200. | `pytest tests/test_api.py` passes 100%. |
| **Next.js Frontend** | Builds cleanly with zero TypeScript/ESLint errors (`npm run build`). | `npm run build` exits 0. |

---
