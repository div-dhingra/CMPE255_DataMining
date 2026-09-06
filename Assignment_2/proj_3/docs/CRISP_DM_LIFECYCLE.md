# Industrial Data Mining Lifecycle: The 6-Phase CRISP-DM Methodology for Customer Segmentation

**Project:** Autonomous Customer Segmentation & Autoresearch Clustering Engine  
**Dataset:** Kaggle Credit Card Customer Behavioral Dataset (8,950 accounts, 18 features)  
**Standard:** Cross-Industry Standard Process for Data Mining (CRISP-DM)  
**Author:** AI & Data Science Engineering Team  
**Date:** 2026-08-28  

---

## 1. Executive Summary & Lifecycle Architecture

The **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology provides a structured, hierarchical, and iterative framework for translating raw transactional credit card records into actionable business segments. While traditional clustering projects often treat algorithm selection as an isolated trial-and-error exercise, this platform operationalizes all six phases of CRISP-DM into an automated, reproducible machine learning system integrated with an autonomous heuristic optimization engine (Autoresearch).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CRISP-DM Lifecycle Loop                           │
│                                                                             │
│      ┌─────────────────────────┐             ┌─────────────────────────┐    │
│      │ 1. Business             │◄───────────►│ 2. Data                 │    │
│      │    Understanding        │             │    Understanding        │    │
│      └────────────┬────────────┘             └────────────┬────────────┘    │
│                   │                                       │                 │
│                   ▼                                       ▼                 │
│      ┌─────────────────────────┐             ┌─────────────────────────┐    │
│      │ 6. Deployment &         │             │ 3. Data                 │    │
│      │    Inference API        │             │    Preparation          │    │
│      └────────────▲────────────┘             └────────────┬────────────┘    │
│                   │                                       │                 │
│                   │                                       ▼                 │
│      ┌────────────┴────────────┐             ┌─────────────────────────┐    │
│      │ 5. Evaluation &         │◄───────────►│ 4. Modeling             │    │
│      │    Research Benchmarks  │             │    (4 Paradigms)        │    │
│      └─────────────────────────┘             └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: Business Understanding

### 2.1 Problem Statement & Commercial Objectives
Modern credit card issuers manage portfolios containing thousands of heterogeneous cardholder accounts. A monolithic marketing and credit policy leads to elevated credit default, customer churn, sub-optimal APR revenue, and low merchant interchange fees. 

The primary business objective is to discover **unsupervised behavioral archetypes** across active customer accounts to enable:
1. **Targeted APR and Credit Limit Optimization**: Tailor credit limits to transacting volume vs. liquidity risk.
2. **Promotional Cross-Selling**: Offer installment loan promotions to balance-heavy revolvers and rewards promotions to transactors.
3. **Liquidity Risk Monitoring**: Detect cash-advance-dependent customers before default delinquency.
4. **Reactivation Strategies**: Re-engage dormant cardholders with zero fees or point incentives.

### 2.2 Target Behavioral Archetypes (Personas)
Our segmentation framework establishes five formal behavioral archetypes:

| Persona | Primary Behavioral Markers | Business Strategy |
|---|---|---|
| **Transactors (Active Spenders)** | High `PURCHASES_FREQUENCY`, high `ONEOFF_PURCHASES`, low `CASH_ADVANCE`, high `PRC_FULL_PAYMENT` | Premium cashback rewards, travel cards, credit limit extensions. |
| **Revolvers (Interest Payers)** | High `BALANCE`, low `PURCHASES_FREQUENCY`, low `PRC_FULL_PAYMENT`, high balance-to-limit ratio | 0% APR balance transfer promotions, installment conversion programs. |
| **Cash Advance Users (Liquidity Seekers)** | High `CASH_ADVANCE_FREQUENCY`, high `CASH_ADVANCE_TRX`, low purchases, low payment ratio | Risk monitoring, fee adjustments, personal loan cross-sell. |
| **Low-Engagement / Inactive** | Low `BALANCE`, near-zero `PURCHASES_TRX`, low `PAYMENTS`, tenure stability | Activation bonuses, waived annual fees, low-usage surveys. |
| **VIP / High-Limit Spenders** | High `CREDIT_LIMIT`, high `PAYMENTS`, high total `PURCHASES`, high transaction velocity | Concierge tier upgrades, bespoke limits, partner co-branded perks. |

### 2.3 Business KPIs & Data Mining Success Criteria
- **Separation Quality**: Mean Silhouette Coefficient $S \ge 0.40$ on transformed feature spaces.
- **Cluster Compactness**: Davies-Bouldin Index $DB \le 1.00$.
- **Partition Stability**: Subsampling bootstrap Adjusted Rand Index (ARI) $\ge 0.85$.
- **Cluster Balance**: Avoid degenerate singletons ($< 1\%$ cohort size) and massive single catch-all clusters ($> 80\%$).

---

## 3. Phase 2: Data Understanding

### 3.1 Kaggle Credit Card Dataset Schema (18 Features)
The benchmark dataset contains 8,950 cardholder records across 18 features (1 identifier + 17 continuous behavioral dimensions):

| Feature Name | Type | Description | Distribution Characteristics |
|---|---|---|---|
| `CUST_ID` | String | Unique Cardholder ID (Excluded from modeling) | Nominal Identifier |
| `BALANCE` | Float | Outstanding account balance ($) | Highly right-skewed ($\gamma_1 > 2.3$) |
| `BALANCE_FREQUENCY` | Float | Frequency of balance updates ($0.0 \dots 1.0$) | Left-skewed, multimodal peak at $1.0$ |
| `PURCHASES` | Float | Total purchase volume over 6 months ($) | Extreme Pareto right-skew ($\gamma_1 > 7.0$) |
| `ONEOFF_PURCHASES` | Float | Maximum single one-off payment volume ($) | Heavy right tail |
| `INSTALLMENTS_PURCHASES` | Float | Purchases made in installment transactions ($) | Heavy right tail |
| `CASH_ADVANCE` | Float | Cash in advance given by the user ($) | Extreme right tail ($\gamma_1 > 5.0$) |
| `PURCHASES_FREQUENCY` | Float | Frequency of purchase transactions ($0.0 \dots 1.0$) | Bimodal (0.0 inactive vs 1.0 daily) |
| `ONEOFF_PURCHASES_FREQUENCY` | Float | Frequency of one-off purchases ($0.0 \dots 1.0$) | Concentration near $0.0$ |
| `PURCHASES_INSTALLMENTS_FREQUENCY`| Float | Frequency of installment purchases ($0.0 \dots 1.0$) | Concentration near $0.0$ |
| `CASH_ADVANCE_FREQUENCY` | Float | Frequency of cash-advance cashouts ($0.0 \dots 1.0$) | Severe zero-inflation |
| `CASH_ADVANCE_TRX` | Integer | Total number of cash advance transactions | Severe zero-inflation |
| `PURCHASES_TRX` | Integer | Total count of purchase transactions | Skewed integer count |
| `CREDIT_LIMIT` | Float | Maximum approved credit limit line ($) | Moderately right-skewed |
| `PAYMENTS` | Float | Total payments made by the customer ($) | Heavy right tail |
| `MINIMUM_PAYMENTS` | Float | Minimum payment amounts required ($) | Heavy right tail with missing values |
| `PRC_FULL_PAYMENT` | Float | Percentage of full payment balance cleared | High concentration at $0.0$ and $1.0$ |
| `TENURE` | Integer | Cardholder account duration (6 to 12 months) | Overwhelmingly 12 months ($> 85\%$) |

### 3.2 Missing Value Profile
- `CREDIT_LIMIT`: 1 missing record ($0.01\%$).
- `MINIMUM_PAYMENTS`: 313 missing records ($3.50\%$).
- **Handling Strategy**: Missingness in `MINIMUM_PAYMENTS` occurs primarily on brand-new or zero-balance accounts that have never generated an active billing statement.

### 3.3 Spatial Clustering Tendency: Hopkins Statistic
Before running clustering algorithms, we evaluate the **Hopkins Statistic** ($H$) to confirm whether the dataset possesses natural cluster structures or is uniformly distributed noise:

$$H = \frac{\sum_{i=1}^m u_i^d}{\sum_{i=1}^m u_i^d + \sum_{i=1}^m w_i^d}$$

Where $u_i$ is the distance from a synthetic uniformly generated point to its nearest real neighbor, and $w_i$ is the distance from a subsampled real point to its nearest neighbor.
- **Empirical Result**: $H = 0.82 \pm 0.04 \gg 0.50$, proving high spatial clustering tendency and rejecting the null hypothesis of uniform spatial dispersion.

### 3.4 Correlation & Multicollinearity
- High positive correlation between `PURCHASES` and `ONEOFF_PURCHASES` ($r \approx 0.91$).
- Strong positive correlation between `PURCHASES_FREQUENCY` and `PURCHASES_INSTALLMENTS_FREQUENCY` ($r \approx 0.86$).
- Negative correlation between `BALANCE` and `PRC_FULL_PAYMENT` ($r \approx -0.32$).

---

## 4. Phase 3: Data Preparation

Data preparation is the most influential phase for clustering performance on heavy-tailed financial data. We engineered a configurable, modular preprocessing pipeline (`DataPreparationPipeline`):

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Raw DataFrame   │────►│ 1. Imputation   │────►│ 2. Outlier      │────►│ 3. Feature      │
│ (8950 x 18)     │     │ (Median / MICE) │     │ (Winsor / IF)   │     │ Engineering     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                                                 │
                                                                                 ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Clean Trans-    │◄────│ 6. Dimensional  │◄────│ 5. Power Trans  │◄────│ 4. Normalization│
│ formed Matrix   │     │ Reduction (PCA) │     │ (Yeo-Johnson)   │     │ & Scaling       │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 4.1 Missing Value Imputation
1. **Median Imputation** (Default): Non-parametric, robust against extreme financial outliers.
2. **Mean Imputation**: Traditional baseline.
3. **KNN Imputation**: Replaces missing values using Euclidean weighted average of $k=5$ nearest neighbors.
4. **MICE (Multivariate Imputation by Chained Equations)**: Iterative Bayesian Ridge chained regression modeling each incomplete attribute as a linear combination of all other observed attributes.

### 4.2 Outlier Detection & Capping
1. **Quantile Winsorization**: Clips extreme tails to $[P_1, P_{99}]$ percentiles to preserve sample size while eliminating distortion.
2. **Interquartile Range (IQR) Trimming**: Caps at $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$.
3. **Isolation Forest**: Isolates multi-attribute anomalous accounts with random partitioning hyperplanes.

### 4.3 Feature Scaling & Power Transformations
Because Euclidean distances in K-Means and GMM are dominated by large-magnitude variables (e.g. `CREDIT_LIMIT` in thousands vs `PRC_FULL_PAYMENT` in decimals), transformations are essential:

- **Yeo-Johnson Power Transformation**: Optimizes parameter $\lambda$ via profile maximum likelihood estimation (MLE) over non-positive and positive values:
  $$\psi(\lambda, y) = \begin{cases} \frac{(y + 1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, y \ge 0 \\ \ln(y + 1) & \text{if } \lambda = 0, y \ge 0 \\ -\frac{(-y + 1)^{2 - \lambda} - 1}{2 - \lambda} & \text{if } \lambda \neq 2, y < 0 \\ -\ln(-y + 1) & \text{if } \lambda = 2, y < 0 \end{cases}$$
- **StandardScaler**: Standardizes features to zero mean and unit variance ($z = (x - \mu)/\sigma$).
- **RobustScaler**: Uses median and IQR ($z = (x - \text{median})/\text{IQR}$).
- **MinMaxScaler**: Bounds values to $[0, 1]$.

### 4.4 Financial Feature Engineering (Behavioral Ratios)
We derive six domain-specific behavioral ratios that reveal normalized cardholder habits:

1. **Credit Utilization Ratio**:
   $$\text{UTILIZATION\_RATIO} = \frac{\text{BALANCE}}{\text{CREDIT\_LIMIT} + \epsilon}$$
2. **Payment to Minimum Payment Ratio**:
   $$\text{PAYMENT\_MIN\_PAYMENT\_RATIO} = \frac{\text{PAYMENTS}}{\text{MINIMUM\_PAYMENTS} + \epsilon}$$
3. **One-Off Purchase Ratio**:
   $$\text{ONEOFF\_PURCHASE\_RATIO} = \frac{\text{ONEOFF\_PURCHASES}}{\text{PURCHASES} + \epsilon}$$
4. **Installment Purchase Ratio**:
   $$\text{INSTALLMENT\_PURCHASE\_RATIO} = \frac{\text{INSTALLMENTS\_PURCHASES}}{\text{PURCHASES} + \epsilon}$$
5. **Cash Advance Share Ratio**:
   $$\text{CASH\_ADVANCE\_RATIO} = \frac{\text{CASH\_ADVANCE}}{\text{PURCHASES} + \text{CASH\_ADVANCE} + \epsilon}$$
6. **Purchase Transaction Velocity**:
   $$\text{PURCHASE\_TRX\_VELOCITY} = \frac{\text{PURCHASES\_TRX}}{\text{TENURE}}$$

### 4.5 Dimensionality Reduction & Projections
- **PCA (Principal Component Analysis)**: Eigen-decomposition of sample covariance matrix. First 2 principal components capture $\sim 48\%$ of total variance; 3 components capture $\sim 62\%$.
- **UMAP (Uniform Manifold Approximation & Projection)**: Preserves local manifold topology and global clustering separation for 2D/3D interactive exploration.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Student-t distribution KL-divergence minimization.

---

## 5. Phase 4: Modeling Across 4 Paradigms

We benchmark six models across four mathematical clustering paradigms:

```
                                  Clustering Paradigms
                                           │
         ┌───────────────────┬─────────────┴───────┬───────────────────┐
         ▼                   ▼                     ▼                   ▼
   Partitioning        Density-Based         Hierarchical        Probabilistic
   ├── K-Means         ├── DBSCAN            └── Agglomerative   └── GMM (EM)
   └── K-Medoids       └── HDBSCAN
```

### 5.1 Partitioning Models
1. **K-Means++** (Arthur & Vassilvitskii, 2007):
   - Objective: $\min_S \sum_{i=1}^k \sum_{x \in S_i} \|x - \mu_i\|^2$
   - Seeding: Proportional probability $P(x) = \frac{D(x)^2}{\sum_{x'} D(x')^2}$ guarantees $O(\log k)$ expected optimality bound.
2. **K-Medoids (FasterPAM)**:
   - Centroids constrained to actual exemplar records ($m_i \in X$).
   - Optimized via FasterPAM with Manhattan ($L_1$) distance, making it impervious to extreme financial power-law outliers.

### 5.2 Density-Based Models
1. **DBSCAN**:
   - Core point condition: $|N_\epsilon(p)| \ge \text{min\_samples}$. Identifies arbitrary non-linear contours and isolates noise as label $-1$.
2. **HDBSCAN** (Campello et al., 2013):
   - Constructs mutual reachability distance graph: $d_{\text{mreach-}k}(a, b) = \max(\text{core}_k(a), \text{core}_k(b), d(a,b))$.
   - Builds condensed cluster tree and extracts flat clusters via Excess of Mass (EOM) stability.

### 5.3 Hierarchical Models
1. **Agglomerative Hierarchical Clustering**:
   - Bottom-up merge tree utilizing Ward's minimum variance criterion:
     $$\Delta ESS_{AB} = \frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2$$
   - Deterministic and dendrogram-interpretable.

### 5.4 Probabilistic Models
1. **Gaussian Mixture Models (GMM)**:
   - Models data as a linear superposition of $K$ Gaussian distributions:
     $$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
   - Fits parameters via Expectation-Maximization (EM), returning soft posterior membership probabilities $P(C_k \mid x)$.

---

## 6. Phase 5: Multi-Metric Evaluation & Persona Profiling

### 6.1 Internal Validation Indices
1. **Silhouette Coefficient** ($S \in [-1, 1]$) (Rousseeuw, 1987):
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad S = \frac{1}{N}\sum_{i=1}^N s(i)$$
2. **Davies-Bouldin Index** ($DB \ge 0$) (Davies & Bouldin, 1979):
   $$DB = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} \left(\frac{s_i + s_j}{d(\mu_i, \mu_j)}\right)$$
3. **Calinski-Harabasz Variance Ratio Criterion** ($CH \ge 0$) (Calinski & Harabasz, 1974):
   $$CH = \frac{\text{Tr}(B_k)/(k-1)}{\text{Tr}(W_k)/(n-k)}$$

### 6.2 Model Selection & Elbow Analysis
- Automated knee detection on Within-Cluster Sum of Squares (Inertia) curve across $k \in [2, 15]$ using the **Kneedle Algorithm** (Satopaa et al., 2011), identifying the global inflection knee at $k^* = 4 \dots 5$.

### 6.3 Subsampling Bootstrap Stability
- Following Von Luxburg (2010), we evaluate partition reproducibility across $B=10$ random subsamples ($80\%$ sample ratio) using Adjusted Rand Index (ARI):
  $$\text{Stability} = \frac{1}{B}\sum_{b=1}^B \text{ARI}(Y_{\text{orig} \mid S_b}, Y_{\text{boot}, b})$$

### 6.4 Persona Profiling & ANOVA Feature Importance
- Centroids are un-scaled back to native financial dollar values.
- One-way ANOVA $F$-statistics identify the top distinguishing features (typically `UTILIZATION_RATIO`, `PURCHASES_FREQUENCY`, `CASH_ADVANCE_RATIO`, `PAYMENTS`).
- Z-score relative deviations are mapped into 5-axis radar charts for visual persona interpretation.

---

## 7. Phase 6: Deployment & Real-Time Inference

### 7.1 Production API Architecture
The system is deployed as a high-throughput, asynchronous FastAPI backend microservice:
- `/api/v1/health`: System health and model cache status.
- `/api/v1/dataset/*`: Ingestion, statistics, distributions, and correlation matrices.
- `/api/v1/clustering/*`: Model execution, 2D/3D projections, persona cards, silhouette ribbons.
- `/api/v1/autoresearch/*`: Autonomous optimization engine triggering, SSE streaming, leaderboard.
- `/api/v1/benchmark/*`: Academic comparative matrices and ablation tables.
- `/api/v1/inference/predict`: Real-time vector scoring.

### 7.2 Real-Time Inference Lifecycle
```
Incoming Customer Vector (JSON)
          │
          ▼
Imputation (Fitted Median/MICE)
          │
          ▼
Feature Engineering (Compute 6 Ratios)
          │
          ▼
Power Transformation (Fitted Yeo-Johnson + Scaling)
          │
          ▼
Active Clustering Model (Centroid Distance / GMM Posterior)
          │
          ▼
JSON Output: {cluster_id, distances, probabilities, persona_card}
```

### 7.3 Model Monitoring & Concept Drift
- **Covariate Shift**: Monitor Kolmogorov-Smirnov (KS) test statistics on incoming customer feature distributions weekly.
- **Cluster Centroid Drift**: Trigger automated re-clustering and autoresearch optimization when average distance to assigned centroid exceeds $2\sigma$ above baseline.
