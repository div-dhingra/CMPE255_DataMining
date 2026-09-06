# Project 5: Agent ML & Data Analytics Skills CRISP-DM Demonstration Platform

[![Skills](https://img.shields.io/badge/Total_Skills-46_Skills-4a9eed?style=for-the-badge)](#skills-overview)
[![Categories](https://img.shields.io/badge/Categories-7_Categories-8b5cf6?style=for-the-badge)](#skills-overview)
[![CRISP-DM](https://img.shields.io/badge/CRISP--DM-6_Phases-22c55e?style=for-the-badge)](#crisp-dm-6-phase-architecture)
[![Dataset](https://img.shields.io/badge/Kaggle_Benchmark-IBM_Telco_Churn-f59e0b?style=for-the-badge)](#dataset-benchmark)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end data mining and machine learning demonstration platform integrating **15 production-grade skills from `param087/agent-ml-skills`** and **31 analytical skills across 6 categories from `nimrodfisher/data-analytics-skills`** (46 skills total). The platform is anchored on the benchmark **Kaggle / IBM Telco Customer Churn** dataset (7,043 customer accounts, 21 attributes), fully structured across the **6 CRISP-DM phases**, and served via an interactive **FastAPI Analytical Backend + Modern Web Dashboard** and an automated sequential **CLI Master Runner**.

---

## Key Highlights

- **Complete 46-Skill Integration**: Combines AI coding agent ML engineering workflows (`agent-ml-skills`) with human-in-the-loop data analyst methodologies (`data-analytics-skills`).
- **Canonical Skill Definitions**: All 46 skills include official Markdown specifications (`SKILL.md`) with YAML frontmatter, workflow protocols, code patterns, and pitfall prevention.
- **Strict CRISP-DM 6-Phase Alignment**: Complete traceability from Phase 1 Business Understanding through Phase 6 Deployment & Monitoring.
- **Zero-Leakage ML Pipeline**: Preprocessing transforms (imputation, scaling, encoding) fitted strictly on the training partition with Stratified 5-Fold Cross-Validation.
- **Financial Impact Quantification**: Directly translates machine learning metrics (0.85 ROC-AUC, 0.67 PR-AUC) into **$259,734.24 in net annual profit gain** and **+114.2% ROI**.
- **Interactive Single-Page Application Dashboard**: Modern dark-themed dashboard featuring a live CRISP-DM phase explorer, searchable skills catalog, real-time customer churn profiler, and ML drift observability console.
- **Master CLI Runner**: Automated script `python run_all_skills.py` executing all 46 skills sequentially in under 12 seconds.
- **Comprehensive Automated Test Suite**: 61 automated tests verifying every skill module, pipeline phase, and REST endpoint (`./run_tests.sh`).

---

## Architecture Diagram

```
+--------------------------------------------------------------------------------------------------+
|                            AGENT SKILLS DEMONSTRATION PLATFORM                                   |
+--------------------------------------------------------------------------------------------------+
                                                 |
         +---------------------------------------+---------------------------------------+
         |                                                                               |
         v                                                                               v
+-----------------------------------+                           +-----------------------------------+
|     param087/agent-ml-skills      |                           |   nimrodfisher/data-analytics     |
|            (15 Skills)            |                           |            (31 Skills)            |
|-----------------------------------|                           |-----------------------------------|
| • Solution Design                 |                           | • 01 Data Quality & Validation (5)|
| • Exploratory Data Analysis       |                           | • 02 Documentation & Knowledge (5)|
| • Data Cleaning                   |                           | • 03 Data Analysis & Invest.   (7)|
| • Feature Engineering             |                           | • 04 Storytelling & Viz        (5)|
| • Pandas Patterns                 |                           | • 05 Stakeholder Comm.         (5)|
| • Imbalanced Data (SMOTE)         |                           | • 06 Workflow Optimization     (4)|
| • Scikit-Learn Pipelines          |                           +-----------------------------------+
| • Model Training (LR, RF, HistGBM)|                                             |
| • Hyperparameter Tuning           |                                             v
| • Model Evaluation (ROC/PR/Brier) |                     +---------------------------------------+
| • 5-Fold Cross-Validation         |                     |     Kaggle IBM Telco Churn Dataset    |
| • ML Monitoring & Observability   |                     |     (7,043 Accounts, 21 Features)     |
| • LoRA Fine-Tuning Setup          |                     +---------------------------------------+
| • RAGAS Retrieval Evaluation      |                                             |
| • LLM-as-Judge Rubric Evaluation  |                                             |
+-----------------------------------+                                             |
         |                                                                        |
         +---------------------------------------+--------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                  THE 6 CRISP-DM PHASES                                           |
|--------------------------------------------------------------------------------------------------|
| Phase 1: Business Understanding -> Metric Tree ($1.67M at stake), Semantic Layer, Assumptions   |
| Phase 2: Data Understanding     -> Programmatic EDA, Quality Audit (74/100), Schema, Correl.     |
| Phase 3: Data Preparation       -> Train-Only Imputation, 26 Features, SMOTE Class Balance      |
| Phase 4: Modeling               -> Scikit-Learn Pipelines, 5-Fold CV (0.8504 AUC), RF Tuning     |
| Phase 5: Evaluation             -> ROC/PR Curves, Cohort Heatmap, 5-Whys, A/B Test (p < 0.001)   |
| Phase 6: Deployment & Monitoring-> $259K Net ROI, Insight-to-Action, Observability (KS/PSI)     |
+--------------------------------------------------------------------------------------------------+
                                                 |
         +---------------------------------------+---------------------------------------+
         |                                                                               |
         v                                                                               v
+-----------------------------------+                           +-----------------------------------+
|   Interactive Web Dashboard       |                           |      Master CLI Sequential        |
|   (FastAPI + Modern HTML5/JS)     |                           |          Runner Engine            |
|-----------------------------------|                           |-----------------------------------|
| • CRISP-DM Flow Explorer          |                           | • python run_all_skills.py        |
| • 46-Skill Catalog & Spec Viewer  |                           | • Executes all 46 skills in 10s   |
| • Real-Time Churn Profiler Form   |                           | • Generates JSON reports in /art  |
| • Observability & Drift Studio    |                           | • Outputs 6 publication figures   |
+-----------------------------------+                           +-----------------------------------+
```

---

## Skills Overview

See the complete catalog of all 46 skills in [SKILLS_CATALOG.md](SKILLS_CATALOG.md).

### 15 Agent ML Skills (`param087/agent-ml-skills`)
1. `solution-design` (Phase 1)
2. `exploratory-data-analysis` (Phase 2)
3. `pandas-patterns` (Phase 2)
4. `data-cleaning` (Phase 3)
5. `feature-engineering` (Phase 3)
6. `imbalanced-data` (Phase 3)
7. `sklearn-pipelines` (Phase 4)
8. `model-training` (Phase 4)
9. `hyperparameter-tuning` (Phase 4)
10. `cross-validation` (Phase 4)
11. `model-evaluation` (Phase 5)
12. `lora-finetuning` (Phase 5)
13. `ragas-evaluation` (Phase 5)
14. `llm-as-judge` (Phase 5)
15. `ml-monitoring-observability` (Phase 6)

### 31 Data Analytics Skills (`nimrodfisher/data-analytics-skills`)
- **01 Data Quality & Validation (5 skills)**: `programmatic-eda`, `data-quality-audit`, `query-validation`, `schema-mapper`, `metric-reconciliation`
- **02 Documentation & Knowledge (5 skills)**: `semantic-model-builder`, `analysis-documentation`, `data-catalog-entry`, `sql-to-business-logic`, `analysis-assumptions-log`
- **03 Data Analysis & Investigation (7 skills)**: `cohort-analysis`, `segmentation-analysis`, `funnel-analysis`, `time-series-analysis`, `root-cause-investigation`, `ab-test-analysis`, `business-metrics-calculator`
- **04 Data Storytelling & Visualization (5 skills)**: `insight-synthesis`, `visualization-builder`, `executive-summary-generator`, `dashboard-specification`, `data-narrative-builder`
- **05 Stakeholder Communication (5 skills)**: `technical-to-business-translator`, `stakeholder-requirements-gathering`, `analysis-qa-checklist`, `methodology-explainer`, `impact-quantification`
- **06 Workflow Optimization (4 skills)**: `analysis-planning`, `context-packager`, `peer-review-template`, `analysis-retrospective`

---

## Dataset Benchmark

The platform runs on the **Kaggle / IBM Telco Customer Churn** dataset (`data/Telco-Customer-Churn.csv`):
- **Size**: 7,043 customer accounts, 21 attributes.
- **Target**: `Churn` ('Yes' = 1,869 accounts / 26.54%, 'No' = 5,174 accounts / 73.46%).
- **Financial Scale**: Baseline Monthly Recurring Revenue (MRR) = $456,116.60; Annualized Churn Revenue Risk = $1,669,570.20.

---

## CRISP-DM 6-Phase Architecture

For a comprehensive report detailing methodology, equations, and results, read [CRISP_DM.md](CRISP_DM.md).

| Phase | Core Objective | Primary Deliverables | Champion Result |
| :--- | :--- | :--- | :--- |
| **Phase 1: Business Understanding** | Formulate retention goals, revenue impact, and metric trees | Semantic layer, Metric tree, Assumptions log | $1.67M ARR at stake |
| **Phase 2: Data Understanding** | Profile distributions, audit quality, reconcile charges | Quality audit report, correlation heatmap | Health Score: 74/100 |
| **Phase 3: Data Preparation** | Zero-leakage cleaning, feature engineering, SMOTE balance | Train-only median imputer, 26 domain features | 8,278 balanced samples |
| **Phase 4: Modeling** | Train & tune Scikit-Learn pipelines via 5-fold CV | Model leaderboard, CV reports | **0.8504 CV ROC-AUC** |
| **Phase 5: Evaluation** | Evaluate holdout ROC/PR curves, cohorts, root cause, A/B test | ROC/PR figure, Cohort heatmap, 5-Whys | **0.8477 Test ROC-AUC**, p < 0.001 |
| **Phase 6: Deployment & Monitoring** | Quantify dollar ROI, draft action matrix, monitor drift | Executive summary, drift plots, latency SLAs | **+$259K Net ROI (+114.2%)** |

---

## Quickstart Guide

### 1. Requirements & Setup
The project requires Python 3.11+. Dependencies are listed in `requirements.txt`.
```bash
# If using the course environment:
source /Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/activate

# Or install dependencies locally:
pip install -r requirements.txt
```

### 2. Execute All 46 Skills via Master CLI Runner
Run the automated sequential runner:
```bash
python run_all_skills.py
```
This executes all 46 skills across all 6 CRISP-DM phases sequentially, generating all JSON artifacts, Markdown reports, and publication figures in `artifacts/` in ~10 seconds.

### 3. Launch Interactive Web Dashboard
Start the FastAPI server:
```bash
uvicorn src.server.app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
- **Interactive Web Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger REST API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Run Automated Test Suite
Execute the 61-test automated suite:
```bash
./run_tests.sh
```
Or run specific subsets:
```bash
./run_tests.sh --agent-ml        # Run 15 Agent-ML skill tests
./run_tests.sh --data-analytics # Run 31 Data-Analytics skill tests
./run_tests.sh --crisp          # Run 6 CRISP-DM phase pipeline tests
./run_tests.sh --api            # Run FastAPI REST endpoint tests
```

---

## REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health status and engine metadata |
| `GET` | `/api/skills` | List all 46 skills with metadata |
| `GET` | `/api/skills/{skill_id}` | Retrieve skill details and raw Markdown specification |
| `POST` | `/api/skills/{skill_id}/run` | Execute a single skill on demand and return its output |
| `GET` | `/api/crisp/phases` | List all 6 CRISP-DM phases and their metadata |
| `GET` | `/api/crisp/{phase_id}` | Retrieve generated phase artifacts and report |
| `POST` | `/api/predict` | Real-time customer churn prediction & retention intervention |
| `GET` | `/api/monitoring/drift` | Live feature and prediction drift metrics (KS & PSI) |

---

## Project Structure

```
agent-skills-demonstration/
├── data/
│   └── Telco-Customer-Churn.csv         # Canonical benchmark dataset (7,043 rows)
├── skills/                              # 46 Canonical Skill Markdown Specifications
│   ├── agent-ml/                        # 15 param087 agent-ml-skills (SKILL.md)
│   └── data-analytics/                  # 31 nimrodfisher data-analytics-skills (SKILL.md)
├── src/
│   ├── config.py                        # Central configuration, schemas, paths, thresholds
│   ├── skills_registry.py               # Registry of all 46 skills with execution handlers
│   ├── crisp_dm/                        # 6 CRISP-DM Phase Orchestrators
│   ├── modules/                         # Core Machine Learning & Analytical Modules
│   └── server/                          # FastAPI Analytical Backend & Web Dashboard
├── artifacts/                           # Generated Reports, Metrics & Figures
│   └── figures/                         # Publication-grade analytical figures
├── tests/                               # 61 Automated Pytest Unit & Integration Tests
├── run_all_skills.py                    # Master CLI sequential runner
├── run_tests.sh                         # Test suite runner script
├── requirements.txt                     # Package dependencies
├── README.md                            # Comprehensive platform overview
├── CRISP_DM.md                          # Detailed 6-Phase CRISP-DM methodology report
└── SKILLS_CATALOG.md                    # Reference catalog of all 46 skills
```
