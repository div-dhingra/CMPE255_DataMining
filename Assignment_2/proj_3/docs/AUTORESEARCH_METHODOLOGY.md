# Autonomous Autoresearch Methodology: Heuristic Hill-Climbing, Simulated Annealing, & Multi-Objective Search for Unsupervised Pipelines

**Project:** Autonomous Customer Segmentation & Autoresearch Clustering Engine  
**Theoretical Baseline:** Inspired by Sakana AI's "The AI Scientist" (Lu et al., 2024), Auto-sklearn, and Metaheuristic Global Optimization  
**Author:** AI & Machine Learning Research Team  
**Date:** 2026-08-28  

---

## 1. Executive Overview & Theoretical Motivation

Automated Machine Learning (AutoML) has achieved widespread maturity in supervised regimes where ground-truth labels ($y$) provide an unambiguous scalar loss gradient. In unsupervised customer segmentation, however, AutoML faces fundamental theoretical hurdles:

1. **Absence of Ground Truth**: No objective target label exists; performance must be evaluated through geometric, topological, and statistical proxies.
2. **Conflicting Multi-Objective Landscape**: Maximizing the Calinski-Harabasz index often produces miniature spherical fragments, while maximizing the Silhouette score can degenerate toward trivial $k=2$ splits. Density algorithms can optimize silhouette by discarding $90\%$ of points as noise.
3. **High-Dimensional Discrete-Continuous Search Space**: The permutation of missing value imputers, outlier trimmers, power transforms, behavioral feature flags, clustering paradigms, and continuous hyperparameters creates a non-convex, discontinuous search space $\Theta$.

To address these challenges, our **Autoresearch Engine** implements a stochastic, self-directed experimentation loop. Inspired by recent advances in automated scientific discovery (Sakana AI's *The AI Scientist*, Lu et al., 2024), the engine iteratively hypothesizes pipeline configurations, evaluates multi-metric cluster validity, navigates local optima via simulated annealing and random restarts, and persists complete experimental telemetry for downstream ablation analysis.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Autonomous Autoresearch Engine Loop                      │
│                                                                             │
│      ┌─────────────────────────┐             ┌─────────────────────────┐    │
│      │ State Initializer       │────►────────│ Pipeline Execution      │    │
│      │ θ_0 ~ Uniform(Θ)        │             │ (DataPrep + Model Fit)  │    │
│      └─────────────────────────┘             └────────────┬────────────┘    │
│                   ▲                                       │                 │
│                   │ (Random Restart on Plateau)           ▼                 │
│      ┌────────────┴────────────┐             ┌─────────────────────────┐    │
│      │ Stagnation & Patience   │             │ Multi-Metric Evaluator  │    │
│      │ Monitor (P > M steps)   │             │ (Sil, DB, CH, ARI, Time)│    │
│      └────────────▲────────────┘             └────────────┬────────────┘    │
│                   │                                       │                 │
│                   │                                       ▼                 │
│      ┌────────────┴────────────┐             ┌─────────────────────────┐    │
│      │ Mutation / Perturbation │◄────────────│ Composite Fitness       │    │
│      │ θ' ~ Mutate(θ)          │  Annealing  │ F(θ) & Ledger Logging   │    │
│      └─────────────────────────┘  Acceptance └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Formal Search Space Parameterization $\Theta$

The pipeline configuration state is formally defined as a tuple $\theta \in \Theta$:

$$\Theta = \mathcal{P}_{\text{impute}} \times \mathcal{P}_{\text{outlier}} \times \mathcal{P}_{\text{scale}} \times \mathcal{P}_{\text{engineer}} \times \mathcal{A} \times \mathcal{H}(\mathcal{A})$$

### 2.1 Search Space Decomposition Table

| Subspace | Parameter | Type | Domain / Values | Default Baseline |
|---|---|---|---|---|
| $\mathcal{P}_{\text{impute}}$ | `imputer_strategy` | Discrete | `['median', 'mean', 'knn', 'mice']` | `'median'` |
| | `knn_neighbors` | Integer | $[3, 10]$ | $5$ |
| $\mathcal{P}_{\text{outlier}}$ | `outlier_strategy` | Discrete | `['winsorize', 'iqr', 'isolation_forest', 'none']` | `'winsorize'` |
| | `winsorize_limits` | Continuous | $[(0.005, 0.005) \dots (0.05, 0.05)]$ | $(0.01, 0.01)$ |
| | `isolation_forest_contamination` | Continuous | $[0.01, 0.08]$ | $0.02$ |
| $\mathcal{P}_{\text{scale}}$ | `scaling_strategy` | Discrete | `['yeo_johnson', 'standard', 'robust', 'minmax']` | `'standard'` |
| $\mathcal{P}_{\text{engineer}}$ | `engineer_ratios` | Boolean | `[True, False]` | `False` |
| $\mathcal{A}$ | `model_name` | Discrete | `['kmeans', 'kmedoids', 'dbscan', 'hdbscan', 'agglomerative', 'gmm']` | `'kmeans'` |

### 2.2 Algorithm-Specific Hyperparameter Spaces $\mathcal{H}(\mathcal{A})$

| Model | Hyperparameter | Type | Domain | Mutation Operator |
|---|---|---|---|---|
| **K-Means** | `n_clusters` ($k$) | Integer | $[2, 10]$ | $\pm 1$ integer step |
| | `init` | Categorical | `['k-means++', 'random']` | Discrete flip |
| | `max_iter` | Integer | $[100, 500]$ | $\pm 50$ step |
| **K-Medoids** | `n_clusters` ($k$) | Integer | $[2, 10]$ | $\pm 1$ integer step |
| | `metric` | Categorical | `['euclidean', 'manhattan']` | Discrete toggle |
| **DBSCAN** | `eps` ($\epsilon$) | Continuous | $[0.3, 2.5]$ | Gaussian drift $\mathcal{N}(0, 0.15^2)$ |
| | `min_samples` | Integer | $[3, 20]$ | $\pm 1$ integer step |
| **HDBSCAN** | `min_cluster_size` | Integer | $[5, 30]$ | $\pm 2$ integer step |
| | `min_samples` | Integer | $[2, 15]$ | $\pm 1$ integer step |
| **Agglomerative** | `n_clusters` ($k$) | Integer | $[2, 10]$ | $\pm 1$ integer step |
| | `linkage` | Categorical | `['ward', 'complete', 'average']` | Discrete transition |
| **GMM** | `n_clusters` ($k$) | Integer | $[2, 10]$ | $\pm 1$ integer step |
| | `covariance_type` | Categorical | `['full', 'tied', 'diag', 'spherical']` | Categorical sample |
| | `reg_covar` | Log-Scale | $[10^{-6}, 10^{-2}]$ | Multiplicative $\times 10^{\pm 0.5}$ |

---

## 3. Normalized Multi-Objective Composite Fitness Function $F(\theta)$

To prevent single-metric exploitation, we formulate a scalar composite fitness function $F(\theta) \in [0, 1]$ that balances cohesion, separation, dispersion, and stability while penalizing degenerate solutions.

$$F(\theta) = w_1 S_{\text{norm}}(\theta) + w_2 (1 - DB_{\text{norm}}(\theta)) + w_3 CH_{\text{norm}}(\theta) + w_4 \text{Stability}(\theta) - P_{\text{noise}}(\theta) - P_{\text{imbalance}}(\theta)$$

### 3.1 Normalization Formulations

1. **Silhouette Normalization**:
   $$S_{\text{norm}}(\theta) = \max\left(0.0, \min\left(1.0, \frac{S(\theta) + 1.0}{2.0}\right)\right)$$
   Maps theoretical domain $[-1.0, 1.0]$ linearly to $[0.0, 1.0]$.

2. **Davies-Bouldin Normalization**:
   $$DB_{\text{norm}}(\theta) = \max\left(0.0, \min\left(1.0, \frac{DB(\theta)}{5.0}\right)\right)$$
   Since $DB \in [0, \infty)$ where lower is better, $1 - DB_{\text{norm}}$ bounds optimal separation at $1.0$ and penalizes poorly separated solutions ($DB \ge 5.0$).

3. **Calinski-Harabasz Logarithmic Normalization**:
   $$CH_{\text{norm}}(\theta) = \min\left(1.0, \frac{\ln(1 + \max(0, CH(\theta)))}{\ln(1 + 10000.0)}\right)$$
   Compresses unbounded variance ratio scale into $[0.0, 1.0]$ via natural logarithmic scaling with an empirical ceiling of $10,000$.

4. **Cluster Stability**:
   $$\text{Stability}(\theta) = \frac{1}{B}\sum_{b=1}^B \text{ARI}(Y_{\text{orig} \mid S_b}, Y_{\text{boot}, b}) \in [0.0, 1.0]$$
   Evaluated over $B=5$ subsampled bootstrap draws.

### 3.2 Penalty Formulations

1. **Excessive Noise Penalty ($P_{\text{noise}}$)**:
   $$P_{\text{noise}}(\theta) = 0.50 \times \max(0.0, \text{noise\_ratio}(\theta) - 0.10)$$
   Penalizes density-based models (DBSCAN/HDBSCAN) if outlier noise classification exceeds $10\%$ of the dataset.

2. **Cluster Imbalance Penalty ($P_{\text{imbalance}}$)**:
   $$P_{\text{imbalance}}(\theta) = 0.20 \times \max\left(0.0, 1.0 - \frac{\mathcal{H}(\mathbf{p})}{\ln(k)}\right)$$
   Where $\mathcal{H}(\mathbf{p}) = -\sum_{i=1}^k p_i \ln(p_i)$ is the Shannon entropy of cluster proportions $p_i = |C_i| / N$. Penalizes degenerate singletons or dominant catch-all partitions.

### 3.3 Default Weight Allocation
- $w_1 = 0.40$ (Silhouette Cohesion & Separation)
- $w_2 = 0.25$ (Davies-Bouldin Compactness)
- $w_3 = 0.15$ (Calinski-Harabasz Variance Ratio)
- $w_4 = 0.20$ (Subsampling Bootstrap Stability)
- Constraint: $\sum_{i=1}^4 w_i = 1.00$.

---

## 4. Optimization Engine Metaheuristic Mechanics

The optimization engine executes a hybrid **Stochastic First-Choice Hill Climber with Simulated Annealing and Random Restarts**:

```
Algorithm 1: Autoresearch Stochastic Hill-Climber
─────────────────────────────────────────────────────────────────────────────
Input : Dataset D, Max Iterations I_max, Patience M, Initial Temp T_0, Cooling Rate α
Output: Best Configuration θ*, Best Fitness F*, Experiment Ledger L

1:  θ ← SampleUniformInitialState(Θ)
2:  F ← EvaluatePipeline(D, θ)
3:  θ* ← θ; F* ← F; patience_counter ← 0; T ← T_0
4:  L ← [LogStep(0, θ, F, accepted=True)]
5:  
6:  for step = 1 to I_max do:
7:      θ' ← MutateNeighbor(θ)
8:      F' ← EvaluatePipeline(D, θ')
9:      ΔF ← F' - F
10:     
11:     # Acceptance Criteria (Metropolis-Hastings Criterion)
12:     if ΔF > 0 then:
13:         accept ← True
14:     else:
15:         P_accept ← exp(ΔF / max(1e-4, T))
16:         accept ← (Uniform(0, 1) < P_accept)
17:     
18:     if accept then:
19:         θ ← θ'; F ← F'
20:     
21:     if F > F* then:
22:         θ* ← θ; F* ← F; patience_counter ← 0
23:     else:
24:         patience_counter ← patience_counter + 1
25:     
26:     # Random Restart upon Plateau / Stagnation
27:     if patience_counter >= M then:
28:         θ ← SampleUniformInitialState(Θ)
29:         F ← EvaluatePipeline(D, θ)
30:         T ← T_0
31:         patience_counter ← 0
32:         restart_flag ← True
33:     else:
34:         restart_flag ← False
35:         T ← T * α   # Geometric Cooling
36:     
37:     L.append(LogStep(step, θ', F', ΔF, F*, accept, restart_flag))
38: 
39: return θ*, F*, L
─────────────────────────────────────────────────────────────────────────────
```

### 4.1 Neighborhood Mutation Operators
1. **Discrete Stage Mutation**: With probability $p_{\text{stage}} = 0.50$, selects one categorical dimension (e.g. `scaling_strategy: standard` $\to$ `yeo_johnson` or `imputer: median` $\to$ `mice`).
2. **Continuous Drift Mutation**: With probability $p_{\text{cont}} = 0.50$, adds Gaussian noise $\delta \sim \mathcal{N}(0, \sigma^2)$ to continuous hyperparameters, clipping to valid parameter bounds.
3. **Cross-Paradigm Jump**: Allows occasional algorithm switching ($\mathcal{A}_i \to \mathcal{A}_j$) to explore radically different geometric partitions.

### 4.2 State Hashing & Tabu Cache
To eliminate redundant computational latency, candidate states are canonicalized into JSON representations and hashed using SHA-256:

$$\text{hash}(\theta) = \text{SHA256}(\text{Canonicalize}(\theta))$$

Fitted metrics and fitness scores are cached in a hash map $\mathcal{M}: \text{hash} \mapsto F(\theta)$. If a proposed mutated state $\theta'$ already exists in $\mathcal{M}$, the cached evaluation is retrieved instantly with $0\text{ ms}$ fitting overhead.

---

## 5. Experiment Telemetry, Ledger, & Artifacts

Every experimental transition is serialized into structured telemetry records:

```json
{
  "iteration": 14,
  "timestamp": "2026-08-28T01:32:45Z",
  "state_hash": "a8f3c912e5",
  "candidate": {
    "imputer": "median",
    "outlier": "winsorize",
    "scaler": "yeo_johnson",
    "engineer_ratios": true,
    "model": "gmm",
    "n_clusters": 5,
    "covariance_type": "diag"
  },
  "metrics": {
    "silhouette": 0.472,
    "davies_bouldin": 0.884,
    "calinski_harabasz": 1540.2,
    "stability_ari": 0.895,
    "noise_ratio": 0.0,
    "runtime_ms": 18.4
  },
  "fitness": 0.6842,
  "delta_fitness": 0.0418,
  "best_fitness_so_far": 0.6842,
  "accepted": true,
  "restart_triggered": false,
  "temperature": 0.428
}
```

### 5.1 Real-Time Streaming (SSE) & UI Telemetry
The FastAPI backend serves Server-Sent Events (SSE) from `/api/v1/autoresearch/stream/{job_id}`, streaming step telemetry directly into the Next.js Autoresearch Studio. The dashboard plots:
- Multi-metric trajectory curves ($F(\theta)$, Silhouette, DB, CH vs. Step).
- Parameter mutation diffs showing exact toggled knobs.
- Real-time Pareto Leaderboard ranking top candidate configurations.

---

## 6. Systematic Ablation Engine

To scientifically validate that performance gains stem from specific engineering decisions rather than random chance, the engine performs component-wise ablation comparisons:

$$\Delta F_{\text{Factor}} = F(\theta_{\text{optimized}}) - F(\theta_{\text{ablated}})$$

### Evaluated Ablation Dimensions
1. **Ablation 1 (Power Transform)**: `Yeo-Johnson` vs. `StandardScaler` (measures variance stabilization on heavy tails).
2. **Ablation 2 (Feature Engineering)**: `6 Derived Ratios` vs. `Raw 17 Features` (measures behavioral normalization).
3. **Ablation 3 (Outlier Handling)**: `Winsorization (1%-99%)` vs. `No Outlier Treatment` (measures stability against Pareto extremes).
4. **Ablation 4 (Hyperparameter Tuning)**: Autoresearch optimal vs. Default framework defaults.
