# Comprehensive Skills Catalog: Agent ML & Data Analytics Skills

This reference catalog details all **46 production skills** integrated into the demonstration platform, including **15 skills from `param087/agent-ml-skills`** and **31 skills from `nimrodfisher/data-analytics-skills`** mapped into the 6 CRISP-DM phases.

---

## 1. Agent ML Skills (`param087/agent-ml-skills`) — 15 Skills

| # | Skill Identifier | CRISP-DM Phase | Description | Inputs | Key Outputs / Artifacts |
|---|------------------|----------------|-------------|--------|--------------------------|
| 1 | `solution-design` | Phase 1: Business Understanding | Formulates ML objective function, primary/secondary metrics, and baseline feasibility. | Business goals, dataset shape | Metric hierarchy, target definition, constraints |
| 2 | `exploratory-data-analysis` | Phase 2: Data Understanding | Systematic profiling, distribution skew, multicollinearity, target leakage detection. | Raw DataFrame | Missingness matrix, correlation table, distribution figures |
| 3 | `data-cleaning` | Phase 3: Data Preparation | Leakage-free train-only imputation, numerical coercion, and zero-leakage transforms. | Train/test split | Imputed splits, median values, imputation ledger |
| 4 | `feature-engineering` | Phase 3: Data Preparation | Domain feature engineering: tenure cohorts, service bundles, bill velocity ratios. | Cleaned DataFrames | 26 engineered attributes, feature definitions |
| 5 | `pandas-patterns` | Phase 2: Data Understanding | Vectorized, high-efficiency pandas patterns avoiding `SettingWithCopyWarning` and memory bloat. | Raw DataFrame | Optimized DataFrame, memory reduction % |
| 6 | `imbalanced-data` | Phase 3: Data Preparation | Balanced class weights, SMOTE oversampling, and PR-AUC optimization for rare classes. | Imbalanced training set | Balanced SMOTE split, class weight dictionary |
| 7 | `sklearn-pipelines` | Phase 4: Modeling | Leakage-free `Pipeline` and `ColumnTransformer` bundling scaling, one-hot encoding, and estimators. | Feature schema, models | Composed pipelines for LR, RF, HistGBM |
| 8 | `model-training` | Phase 4: Modeling | Training supervised classification models across multiple model paradigms. | Preprocessed splits | Fitted pipelines, training diagnostics |
| 9 | `hyperparameter-tuning` | Phase 4: Modeling | Stratified cross-validation grid search over tree depths, estimators, and leaf bounds. | Parameter spaces, training set | Best parameter configurations, tuning ledger |
| 10 | `model-evaluation` | Phase 5: Evaluation | Holdout test evaluation: ROC-AUC, PR-AUC, Brier score calibration, confusion matrix. | Champion pipeline, test split | ROC & PR curves figure, optimal threshold, F1 |
| 11 | `cross-validation` | Phase 4: Modeling | Stratified 5-fold cross-validation with out-of-fold scoring and stability variance. | Training set, pipelines | Mean/std fold scores, overfitting delta |
| 12 | `ml-monitoring-observability` | Phase 6: Deployment & Monitoring | Production monitoring tracking Kolmogorov-Smirnov test, PSI drift, and latency SLAs. | Reference & current data | Feature drift tables, prediction PSI, drift figure |
| 13 | `lora-finetuning` | Phase 5: Evaluation | Parameter-efficient fine-tuning configuration (LoRA rank, alpha, target modules) for retention LLMs. | Base LLM specs (Llama-3-8B) | Trainable parameters count, VRAM reduction factor |
| 14 | `ragas-evaluation` | Phase 5: Evaluation | Synthetic RAG retrieval evaluation measuring Faithfulness, Answer Relevancy, Precision, and Recall. | FAQ retrieval pairs | RAGAS composite score, dimension breakdown |
| 15 | `llm-as-judge` | Phase 5: Evaluation | Multi-criteria rubric evaluation of AI retention interventions across empathy, clarity, and persuasiveness. | Retention candidate messages | Composite rubric scores, approval verdict |

---

## 2. Data Analytics Skills (`nimrodfisher/data-analytics-skills`) — 31 Skills

### Category 01: Data Quality & Validation (5 Skills)
| # | Skill Identifier | CRISP-DM Phase | Description | Output / Artifact |
|---|------------------|----------------|-------------|-------------------|
| 16 | `programmatic-eda` | Phase 2: Data Understanding | Automated data profiling, type inference, and distribution sanity checks. | Automated EDA profile JSON |
| 17 | `data-quality-audit` | Phase 2: Data Understanding | Comprehensive quality audit evaluating schema completeness, whitespace blanks, duplicates. | Health score (74/100), audit issue list |
| 18 | `query-validation` | Phase 2: Data Understanding | SQL query review for correctness, join logic, and zero-division protection. | Validated SQL query specifications |
| 19 | `schema-mapper` | Phase 2: Data Understanding | Entity-relationship and table schema documentation across customer and billing. | ER entities and column dictionary |
| 20 | `metric-reconciliation` | Phase 2: Data Understanding | Reconciles calculated lifetime billing against reported TotalCharges. | Reconciliation variance report |

### Category 02: Documentation & Knowledge (5 Skills)
| # | Skill Identifier | CRISP-DM Phase | Description | Output / Artifact |
|---|------------------|----------------|-------------|-------------------|
| 21 | `semantic-model-builder` | Phase 1: Business Understanding | Standardized metrics and dimensions semantic definitions. | Shared semantic layer specification |
| 22 | `analysis-documentation` | Phase 5: Evaluation | Reproducible methodology, findings, and evidence logging. | Analysis documentation framework |
| 23 | `data-catalog-entry` | Phase 2: Data Understanding | Metadata, technical lineage, and data asset descriptions. | Data catalog entry JSON |
| 24 | `sql-to-business-logic` | Phase 2: Data Understanding | Translates SQL aggregation queries into plain business explanations. | Business logic translation dictionary |
| 25 | `analysis-assumptions-log` | Phase 1: Business Understanding | Tracks critical assumptions and sensitivity impacts. | Assumptions log table |

### Category 03: Data Analysis & Investigation (7 Skills)
| # | Skill Identifier | CRISP-DM Phase | Description | Output / Artifact |
|---|------------------|----------------|-------------|-------------------|
| 26 | `cohort-analysis` | Phase 5: Evaluation | Tenure-based customer cohort retention tracking and decay curves. | Cohort retention heatmap & matrix |
| 27 | `segmentation-analysis` | Phase 5: Evaluation | Multi-dimensional behavioral persona segmentation (VIP Spenders, At-Risk). | Customer persona definitions |
| 28 | `funnel-analysis` | Phase 5: Evaluation | Customer journey drop-off analysis from contract start to renewal. | 4-stage lifecycle conversion funnel |
| 29 | `time-series-analysis` | Phase 5: Evaluation | Tenure-dependent hazard estimation and contract renewal seasonality. | Hazard rate decay curve analysis |
| 30 | `root-cause-investigation` | Phase 5: Evaluation | 5-Whys diagnostic and feature importance driver analysis. | 5-Whys root cause diagnostic |
| 31 | `ab-test-analysis` | Phase 5: Evaluation | Two-sample proportion hypothesis test with p-value and confidence intervals. | A/B test statistical report (p < 0.001) |
| 32 | `business-metrics-calculator` | Phase 1: Business Understanding | Recurring revenue metrics: MRR, Churn Rate, ARPU, and CLV. | Recurring revenue summary |

### Category 04: Data Storytelling & Visualization (5 Skills)
| # | Skill Identifier | CRISP-DM Phase | Description | Output / Artifact |
|---|------------------|----------------|-------------|-------------------|
| 33 | `insight-synthesis` | Phase 6: Deployment | Converts empirical model findings into prioritized business insights. | Top 3 executive insight pillars |
| 34 | `visualization-builder` | Phase 6: Deployment | Publication-grade chart design specifications and aesthetics. | 6 Matplotlib/Seaborn figures |
| 35 | `executive-summary-generator` | Phase 6: Deployment | High-level 1-page executive summary for leadership. | Executive summary report |
| 36 | `dashboard-specification` | Phase 6: Deployment | Comprehensive UI and KPI layout specification. | Dashboard hierarchy wireframe |
| 37 | `data-narrative-builder` | Phase 6: Deployment | 4-part data narrative arc: Context, Tension, Discovery, Resolution. | Executive narrative script |

### Category 05: Stakeholder Communication (5 Skills)
| # | Skill Identifier | CRISP-DM Phase | Description | Output / Artifact |
|---|------------------|----------------|-------------|-------------------|
| 38 | `technical-to-business-translator` | Phase 6: Deployment | Translates ROC-AUC, PR-AUC, and log-loss to business revenue terms. | Metric translation dictionary |
| 39 | `stakeholder-requirements-gathering` | Phase 1: Business Understanding | Structured interview guide and acceptance criteria for retention leads. | Requirement discovery ledger |
| 40 | `analysis-qa-checklist` | Phase 5: Evaluation | Pre-flight analytical QA gate before executive presentation. | 6-item QA verification checklist |
| 41 | `methodology-explainer` | Phase 6: Deployment | Plain-English explanation of machine learning algorithms for non-engineers. | Algorithm explainer guides |
| 42 | `impact-quantification` | Phase 6: Deployment | Quantifies projected dollar return on investment and cost-benefit ratios. | Financial ROI model ($259K Net Gain) |

### Category 06: Workflow Optimization (4 Skills)
| # | Skill Identifier | CRISP-DM Phase | Description | Output / Artifact |
|---|------------------|----------------|-------------|-------------------|
| 43 | `analysis-planning` | Phase 1: Business Understanding | Upfront work breakdown structure and milestone tracking. | 6-phase analytical work plan |
| 44 | `context-packager` | Phase 3: Data Preparation | Packages minimum viable context for AI agent handoff. | Feature dictionary & split summary |
| 45 | `peer-review-template` | Phase 6: Deployment | Peer review checklist assessing leakage, rigor, and validity. | MLOps peer review sign-off |
| 46 | `analysis-retrospective` | Phase 6: Deployment | Post-project retrospective and operational lessons learned. | Retrospective summary |
