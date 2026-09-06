# Retroactive Implementation Plan: Associative Pattern Mining

This document outlines the retroactive implementation plan detailing the architecture, technical stack, key components, and testing strategy for the Associative Pattern Mining system.

## 1. Built Architecture

The system is built as a monolithic web application leveraging a decoupled client-server architecture:
- **Backend API:** A FastAPI-based RESTful backend responsible for parsing datasets, executing frequent itemset mining algorithms, computing advanced interestingness metrics, and running an automated hyperparameter search engine.
- **Frontend SPA:** A lightweight, dependency-free Single Page Application (SPA) dashboard using vanilla HTML, CSS, and JavaScript. It communicates asynchronously with the backend API via HTTP.
- **Data Persistence:** In-memory state management (handled in `backend/src/core/state.py` and `dataset.py`) storing current CRISP-DM stages, loaded transactional datasets, generated itemsets, and discovered rules for the duration of the server lifecycle.

## 2. Technical Stack

- **Language:** Python 3.9+
- **Backend Framework:** FastAPI with Uvicorn (ASGI server)
- **Data Processing:** pandas, numpy, scipy
- **Algorithms:** Custom native implementations of Apriori, FP-Growth, and ECLAT (without external heavy ML libraries), using Python data structures (Dict, Set, List) for efficiency.
- **Testing:** pytest with httpx (for API testing)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API, Chart.js, Vis.js)

## 3. Key Components

### 3.1. API Layer (`backend/src/api/`)
Defines the REST endpoints organized by domain:
- `routes_data.py`: Handles CSV dataset uploads and transformation into a transactional format.
- `routes_mining.py`: Endpoints for triggering algorithms (Apriori, FP-Growth, ECLAT) and generating association rules based on customizable `min_support` and `min_confidence`.
- `routes_crisp_dm.py`: Tracks and advances the phases of the CRISP-DM lifecycle.
- `routes_autoresearch.py`: Exposes endpoints for the automated hyperparameter search loop (hill-climbing optimization).
- `routes_cart.py`: Manages the real-time shopping cart simulator used to demonstrate rule-based recommendations.

### 3.2. Core Logic (`backend/src/core/`)
- **`dataset.py`**: Parses raw dataset inputs into transactional and one-hot encoded matrix formats.
- **`metrics.py`**: Computes 7 specific interestingness metrics (Support, Confidence, Lift, Leverage, Conviction, Zhang's Metric, Kulczynski) for generated rules.
- **`rules.py`**: Extracts strong association rules from mined frequent itemsets.
- **`recommendations.py`**: Matches a user's current simulated "cart" against discovered rules to provide real-time cross-sell recommendations.
- **`crisp_dm.py`**: System state manager tracking the 6 standard phases of CRISP-DM.

### 3.3. Mining Algorithms (`backend/src/algorithms/`)
- **`apriori.py`**: Standard Apriori algorithm with breadth-first search and candidate pruning.
- **`fpgrowth.py`**: High-performance FP-Growth algorithm utilizing an FP-Tree structure for efficient frequent pattern discovery without candidate generation.
- **`eclat.py`**: ECLAT algorithm using vertical data formatting and depth-first search for fast intersection-based frequent itemset mining.

### 3.4. Autoresearch Engine (`backend/src/autoresearch/`)
An optimization layer designed to autonomously discover the best mining configurations (`min_support`, `min_confidence`) across algorithms.
- **`optimizer.py` & `fitness.py`**: Uses local search heuristics (Hill Climbing) to evaluate the quality of rule sets based on aggregated metric scores (e.g., average Lift, maximum Confidence, rule count balance).

### 3.5. Frontend (`backend/src/frontend/`)
- **`index.html` & static assets**: A 5-view dashboard that provides panels for CRISP-DM lifecycle tracking, network graph rule visualization, rules matrix scatter plots, autoresearch metrics, and an interactive shopping cart simulator.

## 4. Testing Strategy

The system enforces correctness and reliability through a comprehensive `pytest` suite located in the `tests/` directory:
- **`test_algorithms.py`**: Unit tests verifying the exactness of Apriori, FP-Growth, and ECLAT outputs using known, deterministic transactional datasets.
- **`test_metrics.py`**: Validates the mathematical precision for all 7 interestingness metric formulas.
- **`test_api.py`**: Integration tests targeting the FastAPI endpoints (data ingestion, executing mining jobs, checking health).
- **`test_crisp_dm.py`**: Ensures the backend state machine transitions correctly between CRISP-DM stages.
- **`test_dataset.py`**: Tests CSV ingestion and transactional parsing rules.
- **`test_recommendations.py`**: Validates that generated rules correctly trigger actionable outputs in a simulated cart payload.
- **`test_autoresearch.py`**: Verifies the logic behind the multi-objective fitness calculation and iteration state within the hyperparameter optimization loop.
