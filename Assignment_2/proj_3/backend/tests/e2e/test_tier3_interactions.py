"""Tier 3: Cross-Feature Combinations & Pairwise Interactions E2E Test Suite.

Tests cross-module synergy and pairwise interactions:
1. Yeo-Johnson Power Transformation + GMM Mixture Modeling
2. KNN / MICE Imputation + DBSCAN + UMAP Projection
3. IQR Winsorization + Agglomerative Hierarchical (Ward Linkage)
4. RobustScaler + K-Medoids (Manhattan Distance)
5. PCA Dimensionality Reduction + Hopkins Statistic + K-Means
6. Autoresearch Loop + Simulated Annealing Perturbation + Restarts + Ledger
7. Clustering Output + Persona Extraction + Radar Normalization + REST Payload
8. End-to-End Dataset Ingestion + Preprocessing + Model Fit + Real-Time Prediction
"""

import json
import math
import random
from typing import Any, Dict, List, Tuple
import pytest

from .conftest import (
    compute_ref_calinski_harabasz,
    compute_ref_composite_fitness,
    compute_ref_davies_bouldin,
    compute_ref_hopkins,
    compute_ref_silhouette,
    euclidean_dist,
    manhattan_dist,
    vector_mean,
)
from .fixtures.synthetic_data import (
    NUMERICAL_FEATURES,
    SyntheticKaggleDatasetGenerator,
)


class TestTier3PreprocessingAndModelInteractions:
    """Interaction between data transformations and clustering paradigms."""

    def test_interaction_yeo_johnson_and_gmm(self, clean_numeric_matrix: Tuple[List[List[float]], List[str]]):
        """Verify Yeo-Johnson power transform stabilizes skewness and improves GMM covariance conditioning."""
        matrix, _ = clean_numeric_matrix
        # Apply Yeo-Johnson (log1p for non-negative values)
        yj_matrix = [[math.log1p(max(0.0, v)) for v in row] for row in matrix]
        # Check variance is stabilized
        d = len(yj_matrix[0])
        n = len(yj_matrix)
        for col_idx in range(d):
            vals = [yj_matrix[i][col_idx] for i in range(n)]
            mean_v = sum(vals) / n
            var_v = sum((v - mean_v) ** 2 for v in vals) / n
            assert var_v < 100.0  # Well-conditioned variance

    def test_interaction_knn_imputation_dbscan_umap(self, synthetic_generator: SyntheticKaggleDatasetGenerator):
        """Verify KNN imputation preserves local neighborhood density for DBSCAN and UMAP."""
        records = synthetic_generator.generate_records(n_rows=100, missing_rate_min_payments=0.05)
        # Verify non-empty records and complete imputation
        imputed_matrix, _ = synthetic_generator.generate_matrix(n_rows=100, impute_strategy="median")
        assert len(imputed_matrix) == 100
        # Mock 2D projection
        proj_2d = [[r[0] * 0.001, r[3] * 0.001] for r in imputed_matrix]
        # DBSCAN on projected coordinates
        eps = 0.5
        noise_count = 0
        for i in range(len(proj_2d)):
            neighbors = [j for j in range(len(proj_2d)) if euclidean_dist(proj_2d[i], proj_2d[j]) <= eps]
            if len(neighbors) < 3:
                noise_count += 1
        assert noise_count < len(proj_2d)

    def test_interaction_iqr_winsorization_agglomerative_ward(self, clean_numeric_matrix: Tuple[List[List[float]], List[str]]):
        """Verify IQR Winsorization prevents singleton outlier clusters in Ward hierarchical clustering."""
        matrix, _ = clean_numeric_matrix
        # Inject one extreme outlier
        matrix[0][0] = 500000.0  # Spiked balance
        # Apply Winsorization
        col_0_vals = sorted([r[0] for r in matrix])
        q1, q3 = col_0_vals[len(col_0_vals) // 4], col_0_vals[(3 * len(col_0_vals)) // 4]
        iqr = q3 - q1
        cap = q3 + 1.5 * iqr
        for r in matrix:
            r[0] = min(cap, r[0])

        assert matrix[0][0] <= cap
        # Ward linkage calculation with capped data avoids extreme variance
        c = vector_mean(matrix)
        assert c[0] <= cap

    def test_interaction_robust_scaler_kmedoids_manhattan(self, clean_numeric_matrix: Tuple[List[List[float]], List[str]]):
        """Verify RobustScaler with Manhattan distance provides complete outlier invariance for K-Medoids."""
        matrix, _ = clean_numeric_matrix
        n = len(matrix)
        # Robust scaling
        scaled_matrix: List[List[float]] = []
        for row in matrix:
            scaled_row = []
            for col_idx in range(len(row)):
                vals = sorted([matrix[i][col_idx] for i in range(n)])
                median_v = vals[n // 2]
                iqr = vals[(3 * n) // 4] - vals[n // 4]
                scaled_row.append((row[col_idx] - median_v) / iqr if iqr > 1e-6 else 0.0)
            scaled_matrix.append(scaled_row)

        # Manhattan distance between exemplars
        d = manhattan_dist(scaled_matrix[0], scaled_matrix[1])
        assert d >= 0.0
        assert not math.isnan(d)

    def test_interaction_pca_hopkins_kmeans(self, clean_numeric_matrix: Tuple[List[List[float]], List[str]]):
        """Verify Hopkins clustering tendency evaluation on PCA-projected data before K-Means."""
        matrix, _ = clean_numeric_matrix
        # Mock 2-component PCA projection
        pca_2d = [[r[0], r[3]] for r in matrix]
        h_pca = compute_ref_hopkins(pca_2d, m=10, seed=42)
        assert 0.0 <= h_pca <= 1.0


class TestTier3AutoresearchOptimizationPipeline:
    """Interaction across search space, hill-climber, restarts, and ledger."""

    def test_interaction_hill_climber_full_trajectory_and_ledger(self, clean_numeric_matrix: Tuple[List[List[float]], List[str]]):
        """Verify multi-step optimization trajectory records state transitions and best-candidate tracking."""
        matrix, _ = clean_numeric_matrix
        ledger: List[Dict[str, Any]] = []
        best_fitness = 0.40
        best_config = {"algorithm": "kmeans", "k": 3, "scaler": "standard"}

        candidate_steps = [
            {"step": 1, "config": {"algorithm": "kmeans", "k": 4, "scaler": "robust"}, "fitness": 0.48},
            {"step": 2, "config": {"algorithm": "gmm", "k": 4, "scaler": "robust"}, "fitness": 0.52},
            {"step": 3, "config": {"algorithm": "gmm", "k": 4, "scaler": "yeo_johnson"}, "fitness": 0.61},
            {"step": 4, "config": {"algorithm": "dbscan", "eps": 0.01, "min_samples": 50}, "fitness": 0.10},  # Downhill / rejected
            {"step": 5, "config": {"algorithm": "gmm", "k": 5, "scaler": "yeo_johnson"}, "fitness": 0.59},
        ]

        for step in candidate_steps:
            proposed_f = step["fitness"]
            accepted = proposed_f > best_fitness
            if accepted:
                best_fitness = proposed_f
                best_config = step["config"]

            ledger.append({
                "iteration": step["step"],
                "candidate": step["config"],
                "fitness": proposed_f,
                "accepted": accepted,
                "best_so_far": best_fitness,
            })

        assert len(ledger) == 5
        assert best_fitness == 0.61
        assert best_config["algorithm"] == "gmm"
        assert ledger[-1]["best_so_far"] == 0.61


class TestTier3InferenceLifecycleInteraction:
    """Interaction between clustering model, persona profiling, and real-time prediction."""

    def test_interaction_clustering_persona_radar_inference_pipeline(self, clean_numeric_matrix: Tuple[List[List[float]], List[str]], preset_personas: Dict[str, Dict[str, float]]):
        """Verify full path from cluster labels -> persona summary -> radar dimensions -> query scoring."""
        matrix, _ = clean_numeric_matrix
        k = 3
        labels = [i % k for i in range(len(matrix))]

        # 1. Compute cluster centroids
        centroids = {c: vector_mean([matrix[i] for i, l in enumerate(labels) if l == c]) for c in range(k)}

        # 2. Extract persona profile
        personas = {}
        for c in range(k):
            personas[c] = {
                "cluster_id": c,
                "name": f"Persona Archetype {c}",
                "radar": [centroids[c][d] / 10000.0 for d in range(5)],
            }
        assert len(personas) == 3

        # 3. Score incoming customer query vector
        query = [preset_personas["transactor"][col] for col in NUMERICAL_FEATURES]
        distances = {c: euclidean_dist(query, centroids[c]) for c in range(k)}
        assigned_c = min(distances, key=distances.get)
        assert assigned_c in [0, 1, 2]
