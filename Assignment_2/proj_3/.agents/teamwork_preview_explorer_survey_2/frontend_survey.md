# Comprehensive Frontend Architecture Survey & Technical Blueprint
**Project**: CRISP-DM Autonomous Clustering & Autoresearch Dashboard  
**Author**: Explorer 2 (`teamwork_preview_explorer`) — Frontend Dashboard & Full-Stack Integration Architect  
**Date**: 2026-08-28  
**Status**: COMPLETE / READY FOR MILESTONE PLANNING  

---

## 1. Executive Summary

This survey defines the frontend architecture, UI component systems, interactive data visualization pipeline, API client integration, and offline mock resilience strategies for the **CRISP-DM Autonomous Clustering & Autoresearch Admin Dashboard**.

The frontend will be built with **Next.js 14+ (App Router)** and **TypeScript 5+**, styled with **Tailwind CSS**, energized by **Lucide React** icons, and powered by a dual-engine visualization stack (**Recharts** for radar/metric charts and **Canvas 2D/3D / Plotly** for high-performance dimensionality reduction projections).

### Core Architectural Pillars
1. **Zero-Downtime Resilience (Dual-Mode Operation)**: The dashboard seamlessly connects to the high-performance FastAPI backend (`http://localhost:8000`). If the backend is starting up, offline, or in demo mode, the dashboard automatically falls back to an embedded, high-fidelity TypeScript mock simulation engine with realistic clustering metrics, 2D/3D projections, interactive hill-climbing steps, and real-time client-side inference.
2. **5-View Specialized Data Science & AI Suite**:
   - **Overview & CRISP-DM Flow**: Interactive 6-phase lifecycle tracker, dataset health metrics, and pipeline status.
   - **Cluster Explorer**: Multi-algorithm switcher, 2D/3D PCA/UMAP/t-SNE projection scatter, cluster persona radar profiles, silhouette sample distributions, and feature distribution histograms.
   - **Autoresearch Studio**: Live hill-climbing monitor, step-by-step trajectory curves, parameter delta heatmaps, perturbation trackers, and trial leaderboards.
   - **Research Benchmark Matrix**: Paper-style comparative tables with LaTeX/Markdown export, standard deviation bounds, ablation matrices, and academic literature alignment.
   - **Inference & Profiler Playground**: Interactive single-sample feature sliders, persona presets, batch CSV upload, real-time cluster assignment, soft membership probabilities, and radar overlay comparison.
3. **Flawless Build & Static Export Option**: Designed for rapid TypeScript compilation (`npm run build` / `npm run typecheck`) with zero lint errors and optional static export (`output: 'export'`) allowing static file serving directly from FastAPI or CDN.

---

## 2. Technology Stack & Framework Architecture

### 2.1 Next.js App Router vs. Pages Router Trade-off Analysis

| Dimension | Next.js App Router (`src/app/`) | Next.js Pages Router (`src/pages/`) | Decision & Rationale |
|---|---|---|---|
| **Layout Hierarchy** | First-class nested layouts (`layout.tsx`, `template.tsx`), persistent nav shells, zero layout flicker. | Manual `getLayout` wrappers or `_app.tsx` state overhead. | **App Router**: Provides persistent navbar, system health status banner, and clean routing structure. |
| **Component Model** | React Server Components (RSC) by default; `"use client"` for interactive charts/state. | Client-side only or full SSR hydration. | **App Router**: RSC allows lightweight static shells while client components isolate stateful Recharts and Canvas 3D renderers. |
| **Build & Export** | Native support for `output: 'export'` or standalone Node server (`output: 'standalone'`). | Static HTML export via `next export`. | **App Router**: Enables single-command static export (`dist/` or `out/`) which FastAPI can mount directly as `StaticFiles` if needed. |
| **Code Organization** | Colocated components, hooks, and types per route segment. | Rigid `/pages` vs `/components` split. | **App Router**: Cleaner modular organization. |

**Decision**: Standardize on **Next.js 14+ with App Router (`src/app/`)**, strict TypeScript configuration, and `"use client"` directives for interactive visualization widgets.

### 2.2 Directory & Project Layout

```
frontend/
├── package.json
├── tsconfig.json
├── next.config.mjs
├── tailwind.config.ts
├── postcss.config.mjs
├── public/
│   ├── favicon.ico
│   └── dataset/sample_credit_card.csv
└── src/
    ├── app/
    │   ├── layout.tsx                 # Root layout with Dark Theme & Navbar
    │   ├── page.tsx                   # Redirect / default view (CRISP-DM Overview)
    │   ├── overview/
    │   │   └── page.tsx               # View 1: CRISP-DM Lifecycle & Data Health
    │   ├── clusters/
    │   │   └── page.tsx               # View 2: Multi-Model Cluster Explorer (2D/3D)
    │   ├── autoresearch/
    │   │   └── page.tsx               # View 3: Autoresearch Hill-Climbing Studio
    │   ├── benchmarks/
    │   │   └── page.tsx               # View 4: Research Benchmark Matrix & Ablation
    │   └── playground/
    │       └── page.tsx               # View 5: Inference & Profiler Playground
    ├── components/
    │   ├── layout/
    │   │   ├── Navbar.tsx             # Main navigation bar with route links
    │   │   ├── SystemStatusBar.tsx    # Live backend status / Demo Mode toggle
    │   │   └── Footer.tsx             # Footer with system telemetry & links
    │   ├── ui/
    │   │   ├── Card.tsx               # Glassmorphic card container
    │   │   ├── Button.tsx             # Primary, secondary, outline, danger buttons
    │   │   ├── Badge.tsx              # Algorithm & metric status badges
    │   │   ├── Tabs.tsx               # Animated tab switcher
    │   │   ├── Slider.tsx             # Parameter & feature input slider
    │   │   ├── StatCard.tsx           # Telemetry metric indicator card
    │   │   ├── Table.tsx              # Styled data table with sort/filter
    │   │   ├── Modal.tsx              # Dialog modal for LaTeX export / details
    │   │   └── Tooltip.tsx            # Contextual metric explainer tooltips
    │   ├── crisp-dm/
    │   │   ├── PhaseTracker.tsx       # 6-phase interactive flow diagram
    │   │   ├── DatasetHealthCard.tsx  # Missingness, rows, outlier rates
    │   │   ├── FeatureSummaryTable.tsx# 18-feature stats, distributions, types
    │   │   └── CorrelationHeatmap.tsx # Interactive feature correlation grid
    │   ├── clusters/
    │   │   ├── ModelSelector.tsx      # K-Means, DBSCAN, Agglomerative, GMM
    │   │   ├── ProjectionScatter2D.tsx# 2D PCA/UMAP/t-SNE scatter with hover info
    │   │   ├── ProjectionScatter3D.tsx# 3D Canvas orbital scatter projection
    │   │   ├── PersonaRadarChart.tsx  # Multi-cluster normalized radar profile
    │   │   ├── SilhouettePlot.tsx     # Silhouette score per sample distribution
    │   │   └── FeatureHistogram.tsx   # Per-cluster attribute distribution bins
    │   ├── autoresearch/
    │   │   ├── TrajectoryChart.tsx    # Real-time Silhouette/DB score step curve
    │   │   ├── ParameterDeltaView.tsx # Step-by-step diff and perturbation log
    │   │   ├── RunControls.tsx        # Start, Pause, Step, Perturb, Reset buttons
    │   │   ├── LiveTelemetry.tsx      # Current step, best score, state machine
    │   │   └── LeaderboardTable.tsx   # Ranked trials & Pareto frontier
    │   ├── benchmarks/
    │   │   ├── PaperTable.tsx         # Academic comparison table with bold bests
    │   │   ├── AblationMatrix.tsx     # Component impact breakdown chart
    │   │   ├── LiteraturePanel.tsx    # Academic citations & theoretical backing
    │   │   └── ExportModal.tsx        # LaTeX / Markdown / BibTeX code copy
    │   └── playground/
    │       ├── FeatureInputForm.tsx   # Sliders & number inputs for 18 features
    │       ├── PresetSelector.tsx     # 1-click customer archetype loader
    │       ├── PredictionCard.tsx     # Predicted cluster & persona summary
    │       ├── ProbabilityChart.tsx   # Soft assignment probability bar chart
    │       ├── RadarOverlay.tsx       # Input sample vs centroid profile radar
    │       └── BatchUploadTester.tsx  # CSV dropzone and batch table inference
    ├── lib/
    │   ├── api.ts                     # Typed REST API client with auto-fallback
    │   ├── mockData.ts                # Deterministic fallback datasets & simulation
    │   ├── clusteringUtils.ts         # Softmax, Euclidean distance, normalization
    │   └── formatters.ts              # Number, percentage, LaTeX table formatters
    ├── types/
    │   └── api.ts                     # Pydantic-mirrored TypeScript types
    └── styles/
        └── globals.css                # Tailwind directives and custom scrollbars
```

---

## 3. Detailed Page & View Hierarchy

```
+---------------------------------------------------------------------------------------------------+
|  [Logo] CRISP-DM Autoresearch Dashboard      [🟢 Connected: localhost:8000] [Demo Mode: OFF] [v1.0] |
|  [Overview & CRISP-DM]  [Cluster Explorer]  [Autoresearch Studio]  [Benchmarks]  [Playground]     |
+---------------------------------------------------------------------------------------------------+
```

### 3.1 View 1: Overview & CRISP-DM Flow (`/` or `/overview`)
**Objective**: Guide the user through the 6 CRISP-DM phases with visual progression, dataset health telemetry, and high-level execution triggers.

#### Layout & UI Components
1. **Header Banner**: Project metadata, Kaggle dataset description (Credit Card Customer Segmentation, 8,950 accounts, 18 behavioral attributes), and quick-action buttons:
   - `[ Run Baseline Benchmark ]`
   - `[ Launch Autoresearch ]`
   - `[ Download Report ]`
2. **Interactive 6-Phase CRISP-DM Tracker**:
   - Phase 1: **Business Understanding** (Customer value tiering, revolving credit risk segmentation, marketing targeting).
   - Phase 2: **Data Understanding** (8,950 rows, 18 features, missingness in `MINIMUM_PAYMENTS` [3.5%] and `CREDIT_LIMIT` [0.01%], skewness analysis).
   - Phase 3: **Data Preparation** (Median/KNN imputation, Log1p/Power transformations, RobustScaler normalization, PCA/UMAP dimensionality reduction).
   - Phase 4: **Modeling** (Partitioning [K-Means], Density [DBSCAN/HDBSCAN], Hierarchical [Agglomerative], Probabilistic [GMM]).
   - Phase 5: **Evaluation** (Silhouette score, Davies-Bouldin index, Calinski-Harabasz score, Bootstrap stability ARI).
   - Phase 6: **Deployment** (FastAPI REST service, real-time inference playground, automated model export).
   - *Interactive Behavior*: Clicking any phase card filters data summaries and navigates directly to the relevant sub-view.
3. **Dataset Health Indicators (Stat Cards)**:
   - Total Records: `8,950`
   - Features: `18` (Numeric behavioral signals)
   - Missing Value Rate: `0.35%` (Handled via Robust Imputation)
   - Outlier Detection Rate: `4.2%` (Identified via Isolation Forest & Mahalanobis distance)
   - Memory Footprint: `1.23 MB`
4. **Feature Distribution & Correlation Heatmap**:
   - Interactive correlation matrix showing strong correlations (e.g. `PURCHASES` vs `ONEOFF_PURCHASES` $r=0.91$, `PURCHASES_TRX` vs `PURCHASES_FREQUENCY` $r=0.69$).

---

### 3.2 View 2: Cluster Explorer (`/clusters`)
**Objective**: Enable deep multi-dimensional analysis of clustering results across algorithms with 2D/3D projections, persona radar charts, and silhouette distributions.

```
+---------------------------------------------------------------------------------------------------+
| Algorithm: [ K-Means (k=4) v ]   Projection: [ PCA 2D | UMAP 2D | PCA 3D (Orbital) ]              |
+-----------------------------------------------------------------+---------------------------------+
|                                                                 |  Cluster Personas (Radar Chart) |
|   2D / 3D Projection Scatter Plot                               |                                 |
|   - 8,950 points colored by cluster                             |   [Radar chart showing 6        |
|   - Centroids marked with glowing badges                        |    key financial dimensions:    |
|   - Hover tooltip: Customer ID, Balance, Purchases, Cluster     |    Balance, Purchases, CashAdv, |
|   - Zoom / Pan / Lasso Selection Controls                       |    CreditLimit, Payments, Freq] |
|                                                                 |                                 |
+-----------------------------------------------------------------+---------------------------------+
|  Silhouette Sample Distribution (Per-Cluster)                   |  Per-Cluster Feature Histograms |
|  [Horizontal ribbon plot showing silhouette values for each    |  [Side-by-side distribution of  |
|   sample vs. mean threshold line = 0.51]                        |   Balance & Purchase Frequency] |
+-----------------------------------------------------------------+---------------------------------+
```

#### Detailed Features
1. **Algorithm & Parameter Switcher**:
   - Dropdown: `K-Means (k=3..8)`, `DBSCAN (eps=0.5, min_samples=5)`, `Agglomerative (Ward, k=4)`, `GMM (n=4, full covariance)`, `Autoresearch Optimized (Best)`.
2. **Dual-Mode Projection Scatter (2D / 3D)**:
   - **2D Mode**: Fast Canvas/Plotly scatter plotting PCA (Dim 1 vs Dim 2) or UMAP (Dim 1 vs Dim 2) with cluster color palette (Emerald, Cyan, Violet, Amber, Rose).
   - **3D Mode**: Interactive 3D Canvas / Three.js orbital projection (PCA 1, 2, 3) with mouse drag rotation, scroll zoom, and hover raycaster.
3. **Cluster Persona Summary & Radar Profile**:
   - Multi-axis radar comparing normalized features:
     - Cluster 0: **"Transactors / Low Balance"** (High purchase frequency, low balance, low cash advance).
     - Cluster 1: **"Cash Advance Borrowers"** (High cash advance, low purchase frequency, high balance).
     - Cluster 2: **"High Spenders / VIP"** (High balance, high purchases, high credit limit, high payments).
     - Cluster 3: **"Inactive / Budget Accounts"** (Low balance, low purchases, low credit limit).
4. **Silhouette Sample Distribution Plot**:
   - Authentic silhouette plot grouping samples by cluster horizontally, showing individual $s_i$ values from -0.1 to +1.0, with a dashed red line for the global average silhouette score.
5. **Feature Distribution Explorer**:
   - Side-by-side histogram/boxplots comparing any selected feature across clusters.

---

### 3.3 View 3: Autoresearch Studio (`/autoresearch`)
**Objective**: Real-time cockpit for the autonomous hill-climbing optimization engine that iteratively explores preprocessing spaces, feature combinations, and algorithm hyperparameters to maximize cluster separation and stability.

```
+---------------------------------------------------------------------------------------------------+
| Controls: [ ▶ Start Run ] [ ⏸ Pause ] [ ⏭ Step Next ] [ ⚡ Perturb / Restart ] [ ⏹ Reset ]        |
| Status: [ 🟢 SEARCHING ]   Step: 28 / 100   Best Silhouette: 0.5842 (+0.1421 vs baseline)         |
+-----------------------------------------------------------------+---------------------------------+
|  Optimization Trajectory Curves                                 |  Live Telemetry & Best Config   |
|  - Green curve: Silhouette Score progression                    |  - Algorithm: K-Means++         |
|  - Blue curve: Davies-Bouldin Index (Minimization)              |  - Scaling: PowerTransformer    |
|  - Orange markers: Restarts / Perturbations                     |  - Features: 12 Selected PCA    |
|  - Step time & convergence delta indicator                      |  - k: 4 | n_init: 20            |
+-----------------------------------------------------------------+---------------------------------+
|  Parameter Delta History & Mutation Log                         |  Experiment Leaderboard         |
|  [Step 28] Mutated k=4 -> k=5, Score: 0.5842 (ACCEPTED - BEST)  |  Rank 1: Trial #28 (Score: 0.58)|
|  [Step 27] Mutated eps=0.45, Score: 0.5120 (REJECTED)           |  Rank 2: Trial #19 (Score: 0.56)|
|  [Step 26] Perturbation applied: Random feature mask            |  Rank 3: Trial #12 (Score: 0.54)|
+-----------------------------------------------------------------+---------------------------------+
```

#### Detailed Features
1. **Interactive Execution Controls**:
   - `Start Run`: Launches live hill-climbing loop (via SSE `/api/autoresearch/stream` or interactive local TypeScript simulator).
   - `Pause`: Halts iteration loop to inspect current state.
   - `Step Next`: Executes exactly one optimization step.
   - `Perturb / Restart`: Applies randomized neighborhood perturbation (temperature boost) to escape local optima.
   - `Reset`: Restores default baseline configuration.
2. **Dual-Axis Optimization Trajectory Curve (Recharts)**:
   - Primary Y-axis: Silhouette Score (Maximizing: 0.40 -> 0.58+).
   - Secondary Y-axis: Davies-Bouldin Index (Minimizing: 1.45 -> 0.82).
   - Step X-axis: Iterations 1 to $N$, with visual flag annotations for accepted improvements, rejected mutations, and perturbation restarts.
3. **Parameter Delta & Mutation Log**:
   - Visual diff table indicating which hyperparameter was modified at each step, the evaluated score delta $\Delta S$, and acceptance decision according to the hill-climbing acceptance criterion.
4. **Leaderboard & Pareto Frontier**:
   - Ranked trials with filterable columns: Trial ID, Algorithm, Preprocessing, Hyperparameters, Silhouette, Davies-Bouldin, Calinski-Harabasz, Runtime, and Pareto status.

---

### 3.4 View 4: Research Benchmark Matrix (`/benchmarks`)
**Objective**: Academic paper-style evaluation suite connecting empirical results to published clustering literature, with comparative tables, error bounds, ablation charts, and LaTeX/Markdown export.

```
+---------------------------------------------------------------------------------------------------+
| Research Benchmark Matrix: Kaggle Credit Card Clustering Benchmark               [ Export LaTeX ] |
+---------------------------------------------------------------------------------------------------+
|  Academic Comparative Analysis Table (Standard Deviation over 10 Folds)                           |
|  -----------------------------------------------------------------------------------------------  |
|  Algorithm                | Silhouette ↑   | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Time (ms)  |
|  -----------------------------------------------------------------------------------------------  |
|  K-Means (Baseline, k=4)  | 0.442 ± 0.012  | 1.341 ± 0.034    | 2,842.1 ± 45.2      | 18.4 ± 2.1 |
|  K-Means (Hill-Climbed)   | 0.584 ± 0.009* | 0.812 ± 0.021*   | 3,921.6 ± 38.4*     | 24.2 ± 3.0 |
|  DBSCAN (eps=0.5)         | 0.381 ± 0.025  | 1.624 ± 0.052    | 1,412.0 ± 82.1      | 42.1 ± 4.5 |
|  HDBSCAN (min_cluster=50) | 0.495 ± 0.015  | 1.052 ± 0.028    | 3,105.4 ± 51.0      | 88.6 ± 7.2 |
|  Agglomerative (Ward, k=4)| 0.468 ± 0.011  | 1.215 ± 0.030    | 2,980.2 ± 41.2      | 145.0 ± 12 |
|  GMM (Full Covariance)    | 0.451 ± 0.018  | 1.290 ± 0.040    | 2,750.8 ± 60.1      | 65.3 ± 5.8 |
|  GMM (Hill-Climbed)       | 0.562 ± 0.010* | 0.895 ± 0.025*   | 3,640.5 ± 44.0*     | 78.1 ± 6.2 |
+---------------------------------------------------------------------------------------------------+
|  Ablation Study Matrix                                  |  Literature Alignment & Citations       |
|  - Baseline (Raw Features + K-Means): 0.382             |  - Rousseeuw (1987): Silhouette Metric  |
|  - + Imputation & PowerTransform: 0.442 (+0.060)        |  - Arthur & Vassilvitskii (2007): k-means++|
|  - + UMAP / PCA Denoising: 0.518 (+0.076)               |  - McInnes et al. (2018): UMAP / HDBSCAN|
|  - + Autoresearch Hill-Climbing: 0.584 (+0.066)         |  - Ester et al. (1996): DBSCAN          |
+---------------------------------------------------------------------------------------------------+
```

#### Detailed Features
1. **Paper-Style Comparison Table**:
   - Bolded optimal metrics with asterisk indicators ($^*$).
   - Bootstrap standard error bounds ($\mu \pm \sigma$).
2. **Ablation Study Breakdown**:
   - Waterfall / stacked bar chart illustrating the metric gain contributed by each step of the pipeline.
3. **LaTeX & Markdown Modal Exporter**:
   - One-click copy for `\begin{table}...\end{table}` LaTeX snippet formatted for IEEE / ACM / NeurIPS paper insertion.
   - BibTeX citation blocks for key benchmark references.
4. **Literature Alignment Discussion Panel**:
   - Explanations linking empirical findings (e.g. why HDBSCAN outperforms standard DBSCAN on varying density credit card transactions; why PowerTransformer + K-Means improves silhouette over raw StandardScaler).

---

### 3.5 View 5: Inference & Profiler Playground (`/playground`)
**Objective**: Interactive sandbox where data scientists or business users can input customer behavioral records (via sliders or CSV upload) and receive instant cluster predictions, soft assignment probabilities, and persona interpretations.

```
+---------------------------------------------------------------------------------------------------+
| Quick Presets: [ 💎 VIP Spender ] [ 💳 Revolver / Cash Advance ] [ 🛍️ Transactor ] [ 💤 Inactive ] |
+-----------------------------------------------------------------+---------------------------------+
|  Customer Behavioral Inputs (18 Features)                       |  Prediction & Segment Profile   |
|  - Balance ($): [======|======] $4,250.00                       |                                 |
|  - Balance Frequency: [===========|] 0.95                       |  Assigned Segment:              |
|  - Purchases ($): [====|==========] $1,200.00                   |  🎯 Cluster 2: VIP High Spender |
|  - Cash Advance ($): [|=============] $0.00                     |  Confidence: 94.8%              |
|  - Credit Limit ($): [========|=====] $8,500.00                 |                                 |
|  - Payments ($): [=====|========] $1,850.00                     |  Soft Membership Probabilities: |
|  - Tenure (Months): [==============|] 12                        |  - Cluster 2 (VIP): 94.8%       |
|                                                                 |  - Cluster 0 (Transactor): 4.1% |
|  [ ⚡ Predict Segment ]  [ 🔄 Reset Defaults ]                  |  - Cluster 1 (Revolver): 0.8%   |
|                                                                 |  - Cluster 3 (Inactive): 0.3%   |
+-----------------------------------------------------------------+---------------------------------+
|  Radar Overlay (Input Sample vs. Cluster 2 Centroid)            |  Batch CSV Inference Tester     |
|  [Recharts Radar overlaying customer values against cluster avg]|  [Dropzone for .csv batch runs] |
+-----------------------------------------------------------------+---------------------------------+
```

#### Detailed Features
1. **1-Click Archetype Presets**:
   - Instantly pre-fills the 18 feature inputs with characteristic personas (e.g., High Spender, Cash Advance Heavy, Pure Transactor, Inactive).
2. **Interactive Feature Sliders & Numerical Controls**:
   - Bounded sliders with real-time validation for all 18 features (Balance, Purchases, Cash Advance, Credit Limit, Tenure, Frequencies).
3. **Soft Assignment & Probability Breakdown**:
   - Computes distances to all cluster centroids and derives normalized softmax probabilities.
4. **Radar Profile Overlay**:
   - Superimposes the user's custom input point (Cyan line) over the predicted cluster's mean centroid (Emerald polygon) across normalized attributes.
5. **Batch CSV Inference Dropzone**:
   - Allows uploading a `.csv` file with customer records, computes batch predictions with progress bar, and displays results in a sortable/downloadable table.

---

## 4. Visualization Technology & Charting Library Survey

### 4.1 Library Trade-off Evaluation

| Library | Strengths | Weaknesses | Recommended Role in Dashboard |
|---|---|---|---|
| **Recharts** (v2.12+) | Native React SVG, responsive, smooth animations, zero SSR issues, excellent for Radar, Line, Bar, Area charts. | SVG DOM can degrade above 3,000 individual scatter points. | **Primary for 2D Analytics**: Radar persona profiles, Autoresearch trajectory curves, silhouette distributions, feature histograms, probability bars. |
| **HTML5 Canvas 2D Scatter** | Ultra-fast (60 FPS for 10,000+ points), zero external bundle weight, custom hover tooltips and selection lasso. | Requires manual axis drawing if done from scratch. | **Primary for 2D Projection**: PCA / UMAP / t-SNE scatter with high frame rates and cluster color mapping. |
| **Three.js / Canvas 3D** | High-performance WebGL 3D orbital scatter, camera controls, point raycasting, rotation animation. | Slightly more complex setup. | **Primary for 3D Projection**: 3D PCA coordinate explorer with mouse drag orbit and glowing cluster centroids. |
| **Plotly.js (`react-plotly.js`)** | Out-of-the-box 2D/3D scatter with built-in lasso/zoom tools. | Heavy bundle size (~3MB) unless loaded dynamically (`next/dynamic`). | **Optional Dynamic Fallback**: Can be dynamically loaded without blocking initial page load. |

### 4.2 Visualization Stack Decision
- **Recharts** for all structured charts (Radar, Line, Bar, Area, Composite).
- **Custom Canvas 2D & WebGL 3D Canvas** for ultra-responsive projection scatters (lightweight, zero hydration bugs, zero SSR bundle penalties).
- All visual components wrapped with React Client Component boundary (`"use client"`) and mounted lifecycle check (`isMounted`) to eliminate any Next.js hydration mismatch.

---

## 5. API Client Integration & Offline Mock Resilience

### 5.1 Dual-Mode Architecture

```
                                  +-----------------------+
                                  |  Next.js UI Component |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |   Unified API Client  |
                                  |     (src/lib/api.ts)  |
                                  +-----------+-----------+
                                              |
                     +------------------------+------------------------+
                     | (FastAPI Online)                                | (FastAPI Offline / Demo)
                     v                                                 v
         +-----------------------+                         +-----------------------+
         |  FastAPI Backend REST |                         | High-Fidelity Mock    |
         |  http://localhost:8000|                         | Simulation Engine     |
         +-----------------------+                         +-----------------------+
```

### 5.2 API Route Specifications & TypeScript Contracts

```typescript
// src/types/api.ts

export interface DatasetHealth {
  totalRows: number;
  totalFeatures: number;
  missingValuesCount: number;
  missingValuesRate: number;
  outliersDetected: number;
  outlierRate: number;
  memoryUsageMb: number;
}

export interface FeatureSummary {
  name: string;
  dataType: string;
  mean: number;
  std: number;
  min: number;
  median: number;
  max: number;
  missingCount: number;
  description: string;
}

export interface ClusterPoint2D {
  id: string;
  x: number;
  y: number;
  cluster: number;
  balance: number;
  purchases: number;
  creditLimit: number;
}

export interface ClusterPoint3D extends ClusterPoint2D {
  z: number;
}

export interface ClusterPersona {
  clusterId: number;
  name: string;
  size: number;
  percentage: number;
  silhouetteScore: number;
  radarMetrics: {
    feature: string;
    value: number; // Normalized 0-1
    rawValue: number;
  }[];
  description: string;
  recommendations: string[];
}

export interface AutoresearchStep {
  step: number;
  timestamp: string;
  algorithm: string;
  hyperparameters: Record<string, any>;
  preprocessing: Record<string, any>;
  silhouetteScore: number;
  daviesBouldinIndex: number;
  calinskiHarabaszIndex: number;
  stabilityScore: number;
  status: 'ACCEPTED' | 'REJECTED' | 'PERTURBED' | 'RESTART';
  mutationDescription: string;
  deltaSilhouette: number;
}

export interface BenchmarkRow {
  algorithm: string;
  isOptimized: boolean;
  kOrEps: string;
  silhouetteMean: number;
  silhouetteStd: number;
  daviesBouldinMean: number;
  daviesBouldinStd: number;
  calinskiHarabaszMean: number;
  calinskiHarabaszStd: number;
  stabilityScore: number;
  runtimeMs: number;
  outlierRatio: number;
}

export interface InferenceRequest {
  balance: number;
  balanceFrequency: number;
  purchases: number;
  oneOffPurchases: number;
  installmentsPurchases: number;
  cashAdvance: number;
  purchasesFrequency: number;
  oneOffPurchasesFrequency: number;
  purchasesInstallmentsFrequency: number;
  cashAdvanceFrequency: number;
  cashAdvanceTrx: number;
  purchasesTrx: number;
  creditLimit: number;
  payments: number;
  minimumPayments: number;
  prcFullPayment: number;
  tenure: number;
}

export interface InferenceResponse {
  predictedCluster: number;
  personaName: string;
  confidence: number;
  probabilities: { clusterId: number; personaName: string; probability: number }[];
  radarComparison: { feature: string; sampleValue: number; clusterCentroid: number }[];
  topContributingFeatures: { feature: string; deviation: number }[];
}
```

### 5.3 Deterministic Mock Engine Design (`src/lib/mockData.ts`)
1. **Pre-computed Projections**: Includes 500 representative sample points with exact 2D PCA/UMAP and 3D coordinates, cluster labels 0-3, and realistic customer attributes.
2. **Interactive Autoresearch Simulation**: An internal step generator produces real-time trajectory curves that demonstrate hill climbing from Silhouette 0.442 to 0.584 with temperature perturbations and parameter diffs.
3. **Interactive Inference**: A client-side cluster distance calculator using Euclidean centroid norms and softmax probabilities so user slider movements immediately produce live predictions with full persona breakdowns.

---

## 6. Build Verification, Static Packaging & Verification Strategy

### 6.1 Package Configuration (`package.json`)

```json
{
  "name": "crisp-dm-clustering-dashboard",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev -p 3000",
    "build": "next build",
    "start": "next start -p 3000",
    "lint": "next lint",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "clsx": "^2.1.1",
    "lucide-react": "^0.395.0",
    "next": "^14.2.5",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "recharts": "^2.12.7",
    "tailwind-merge": "^2.3.0"
  },
  "devDependencies": {
    "@types/node": "^20.12.0",
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.4",
    "typescript": "^5.4.5"
  }
}
```

### 6.2 Verification Checklist for Implementation Milestone
- [ ] `npm install` executes cleanly with zero dependency conflicts.
- [ ] `npm run typecheck` passes with zero TypeScript errors under strict mode (`"strict": true`).
- [ ] `npm run lint` passes with zero warnings/errors.
- [ ] `npm run build` generates production build artifacts without any SSR hydration errors.
- [ ] All 5 views render correctly in both Connected Mode (FastAPI online) and Demo Fallback Mode (FastAPI offline).
- [ ] Sliders, algorithm selectors, 2D/3D projections, and autoresearch controls are fully interactive.

---

## 7. Recommended Implementation Milestones for Orchestrator

1. **Step 1: Frontend Scaffold & UI Primitives**:
   - Initialize Next.js 14 App Router, Tailwind CSS dark theme configuration, typography, and core UI components (Navbar, SystemStatusBar, Cards, Buttons, Badges, Tabs, Sliders, Modals).
2. **Step 2: API Client & Resilient Mock Engine**:
   - Implement `src/lib/api.ts` with auto-ping health detection and `src/lib/mockData.ts` with complete pre-calculated projections, personas, benchmark matrices, and client-side inference calculator.
3. **Step 3: CRISP-DM Overview & Cluster Explorer Views**:
   - Build View 1 (`/overview` with 6-phase tracker, health metrics, correlation matrix) and View 2 (`/clusters` with 2D/3D projection scatter, persona radar chart, silhouette plot, feature distribution histograms).
4. **Step 4: Autoresearch Studio & Research Benchmarks**:
   - Build View 3 (`/autoresearch` with live trajectory curves, parameter deltas, leaderboard, controls) and View 4 (`/benchmarks` with paper comparison table, LaTeX exporter, ablation matrix).
5. **Step 5: Inference & Profiler Playground**:
   - Build View 5 (`/playground` with interactive sliders, presets, real-time probability charts, radar overlay, and CSV batch upload).
6. **Step 6: Build Verification & Dual-Track E2E Integration**:
   - Execute `npm run build` and `npm run typecheck`, verify end-to-end integration with FastAPI backend endpoints.
