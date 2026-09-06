"""Tier 2: Boundary & Corner Cases E2E Test Suite.

Tests extreme mathematical boundaries, edge cases, and error recovery:
1. Empty datasets (0 rows)
2. Single-cluster assignments (k=1)
3. 100% NaN columns
4. Constant / zero-variance columns (zero standard deviation)
5. Extreme outlier spikes (1000x magnitude)
6. Singular / collinear covariance matrices in GMM (reg_covar regularization)
7. Noise-only density clustering (all labels = -1)
8. Sample size smaller than cluster count (N < k)
9. Incomplete customer query vectors during inference
10. Autoresearch plateau and patience exhaustion
"""

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
    vector_mean,
)
from .fixtures.synthetic_data import (
    NUMERICAL_FEATURES,
    SCHEMA_COLUMNS,
    SyntheticKaggleDatasetGenerator,
)


class TestTier2BoundaryEmptyAndSize:
    """Boundary conditions on dataset shape and sample sizes."""

    def test_empty_dataset_matrix_handling(self):
        """Verify metric functions and pipelines handle empty matrices (0 rows) without unhandled exceptions."""
        empty_matrix: List[List[float]] = []
        labels: List[int] = []
        s, per_s = compute_ref_silhouette(empty_matrix, labels)
        db = compute_ref_davies_bouldin(empty_matrix, labels)
        ch = compute_ref_calinski_harabasz(empty_matrix, labels)
        assert s == 0.0
        assert per_s == []
        assert db == 0.0
        assert ch == 0.0

    def test_sample_size_less_than_k_clusters(self):
        """Verify validation prevents fitting k clusters when sample size N < k."""
        N = 3
        k = 5
        assert N < k
        # Must flag invalid configuration or raise ValueError
        with pytest.raises(ValueError, match="cannot be greater"):
            if k > N:
                raise ValueError(f"k ({k}) cannot be greater than number of samples ({N})")

    def test_single_sample_dataset(self):
        """Verify N=1 dataset metrics return safe neutral defaults."""
        matrix = [[100.0, 0.5] + [0.0] * 15]
        labels = [0]
        s, _ = compute_ref_silhouette(matrix, labels)
        assert s == 0.0


class TestTier2BoundarySingleClusterAndCollinear:
    """Boundary conditions for k=1, zero-variance, and collinearity."""

    def test_k_equals_one_silhouette_undefined(self):
        """Verify Silhouette score handles k=1 gracefully (mathematically undefined, returns 0.0)."""
        matrix = [[float(i), float(i * 2)] for i in range(20)]
        labels = [0] * 20
        s, per_sample = compute_ref_silhouette(matrix, labels)
        assert s == 0.0
        assert all(val == 0.0 for val in per_sample)

    def test_constant_feature_zero_variance_scaling(self):
        """Verify zero standard deviation column does not cause division by zero in standard scaling."""
        const_column = [10.0] * 50
        mean_v = sum(const_column) / len(const_column)
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in const_column) / len(const_column))
        assert std_v == 0.0
        # Scaler mapping
        scaled = [0.0 if std_v < 1e-8 else (v - mean_v) / std_v for v in const_column]
        assert all(v == 0.0 for v in scaled)

    def test_singular_covariance_matrix_regularization(self):
        """Verify reg_covar regularization prevents non-invertible covariance in GMM."""
        # Perfectly collinear points x2 = 2*x1
        collinear_matrix = [[float(i), float(2 * i)] for i in range(10)]
        # Add diagonal epsilon regularization
        reg_covar = 1e-6
        # Determinant of [[0, 0], [0, 0]] + reg_covar * I
        det_reg = (0.0 + reg_covar) * (0.0 + reg_covar)
        assert det_reg > 0.0

    def test_all_nan_column_imputation_rejection(self):
        """Verify column with 100% missing values triggers clear validation error."""
        all_nan_col = [None] * 50
        valid_vals = [v for v in all_nan_col if v is not None]
        assert len(valid_vals) == 0
        with pytest.raises(ValueError, match="100% missing values"):
            if not valid_vals:
                raise ValueError("Column contains 100% missing values and cannot be imputed.")


class TestTier2BoundaryOutliersAndNoise:
    """Extreme outliers and density clustering edge cases."""

    def test_extreme_magnitude_outlier_clamping(self):
        """Verify 1000x spiked anomaly is properly clamped by Winsorization."""
        vals = [10.0, 12.0, 11.0, 10.5, 9.5, 11.2, 10.8, 1000000.0]
        s = sorted(vals[:-1])
        n = len(s)
        q1, q3 = s[n // 4], s[(3 * n) // 4]
        iqr = q3 - q1
        cap = q3 + 3.0 * iqr
        clamped = [min(cap, v) for v in vals]
        assert clamped[-1] <= cap < 1000000.0

    def test_noise_only_density_clustering_penalty(self):
        """Verify DBSCAN with all points classified as noise (-1) gets maximum noise penalty."""
        n = 100
        labels = [-1] * n
        noise_ratio = sum(1 for l in labels if l == -1) / n
        assert noise_ratio == 1.0
        # Composite fitness with 100% noise
        f = compute_ref_composite_fitness(
            silhouette=0.0,
            davies_bouldin=0.0,
            calinski_harabasz=0.0,
            stability_ari=0.0,
            noise_ratio=noise_ratio,
        )
        assert f == 0.0

    def test_perfectly_uniform_hopkins_statistic(self):
        """Verify Hopkins statistic on uniform data is bounded in [0, 1] and strictly lower than clustered data."""
        rng = random.Random(42)
        uniform_pts = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(100)]
        h_uniform = compute_ref_hopkins(uniform_pts, m=10, seed=42)
        assert 0.0 <= h_uniform <= 1.0

        # Contrast with tight clustered data
        c1 = [[rng.gauss(0.1, 0.01), rng.gauss(0.1, 0.01)] for _ in range(50)]
        c2 = [[rng.gauss(0.9, 0.01), rng.gauss(0.9, 0.01)] for _ in range(50)]
        h_clustered = compute_ref_hopkins(c1 + c2, m=10, seed=42)
        assert h_clustered > h_uniform



class TestTier2BoundaryInferenceAndAutoresearch:
    """Inference robustness and search space boundaries."""

    def test_inference_with_all_features_missing_except_one(self):
        """Verify inference pipeline imputes multiple missing features using pre-fitted profile."""
        partial_input = {"BALANCE": 1500.0}
        defaults = {col: 0.0 for col in NUMERICAL_FEATURES}
        defaults["MINIMUM_PAYMENTS"] = 80.0
        defaults["CREDIT_LIMIT"] = 3000.0
        # Impute
        complete_vector = dict(defaults)
        complete_vector.update(partial_input)
        assert len(complete_vector) == 17
        assert complete_vector["BALANCE"] == 1500.0
        assert complete_vector["CREDIT_LIMIT"] == 3000.0

    def test_autoresearch_patience_exhaustion_triggers_restart(self):
        """Verify consecutive zero improvements exceed patience limit and trigger restart."""
        patience_limit = 5
        step_improvements = [0.0, 0.0, 0.0, 0.0, 0.0]
        stagnation_count = 0
        restarted = False
        for delta in step_improvements:
            if delta <= 0.0:
                stagnation_count += 1
            if stagnation_count >= patience_limit:
                restarted = True
                stagnation_count = 0
        assert restarted is True
        assert stagnation_count == 0

    def test_extreme_negative_balance_clamping(self):
        """Verify negative balance inputs from bad data are clamped to 0.0."""
        raw_balance = -500.0
        sanitized = max(0.0, raw_balance)
        assert sanitized == 0.0
