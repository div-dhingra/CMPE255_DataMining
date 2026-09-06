"""E2E Test Configuration and Authoritative Fixtures.

Provides:
- Standard synthetic Kaggle dataset fixtures
- Reference mathematical oracles for internal cluster metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, Hopkins, Composite Fitness)
- Edge case data generators (empty, 1-cluster, extreme outliers, all-NaN, collinear)
- Preset customer profile vectors
- Pipeline contract verification helpers
"""

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pytest

from .fixtures.synthetic_data import (
    NUMERICAL_FEATURES,
    SCHEMA_COLUMNS,
    SyntheticKaggleDatasetGenerator,
    create_synthetic_dataset,
)


# ============================================================================
# Authoritative Reference Oracles (Pure Python Mathematical Verifiers)
# ============================================================================

def euclidean_dist(p1: List[float], p2: List[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def manhattan_dist(p1: List[float], p2: List[float]) -> float:
    """Manhattan distance between two vectors."""
    return sum(abs(a - b) for a, b in zip(p1, p2))


def vector_mean(vectors: List[List[float]]) -> List[float]:
    """Compute centroid / mean vector."""
    if not vectors:
        return []
    dim = len(vectors[0])
    n = len(vectors)
    return [sum(v[d] for v in vectors) / n for d in range(dim)]


def compute_ref_silhouette(X: List[List[float]], labels: List[int]) -> Tuple[float, List[float]]:
    """Compute Silhouette coefficient globally and per sample.
    
    If k < 2 or all points are noise (-1), returns (0.0, [0.0]*N).
    """
    n = len(X)
    unique_labels = sorted(list(set(l for l in labels if l != -1)))
    if len(unique_labels) < 2 or n < 2:
        return 0.0, [0.0] * n

    # Group sample indices by cluster
    cluster_indices: Dict[int, List[int]] = {l: [] for l in unique_labels}
    for idx, l in enumerate(labels):
        if l in cluster_indices:
            cluster_indices[l].append(idx)

    sample_silhouettes: List[float] = [0.0] * n

    for i in range(n):
        c_i = labels[i]
        if c_i == -1 or len(cluster_indices[c_i]) <= 1:
            sample_silhouettes[i] = 0.0
            continue

        # a(i): mean intra-cluster distance
        same_cluster = cluster_indices[c_i]
        a_i = sum(euclidean_dist(X[i], X[j]) for j in same_cluster if j != i) / (len(same_cluster) - 1)

        # b(i): min mean distance to other clusters
        b_i = float("inf")
        for other_c, other_indices in cluster_indices.items():
            if other_c == c_i or not other_indices:
                continue
            dist_to_c = sum(euclidean_dist(X[i], X[j]) for j in other_indices) / len(other_indices)
            if dist_to_c < b_i:
                b_i = dist_to_c

        if math.isinf(b_i):
            s_i = 0.0
        else:
            denom = max(a_i, b_i)
            s_i = (b_i - a_i) / denom if denom > 1e-12 else 0.0
        sample_silhouettes[i] = s_i

    # Valid non-noise samples
    valid_s = [s for idx, s in enumerate(sample_silhouettes) if labels[idx] != -1]
    global_s = sum(valid_s) / len(valid_s) if valid_s else 0.0
    return global_s, sample_silhouettes


def compute_ref_davies_bouldin(X: List[List[float]], labels: List[int]) -> float:
    """Compute Davies-Bouldin Index."""
    unique_labels = sorted(list(set(l for l in labels if l != -1)))
    k = len(unique_labels)
    if k < 2:
        return 0.0

    clusters: Dict[int, List[List[float]]] = {l: [] for l in unique_labels}
    for x, l in zip(X, labels):
        if l in clusters:
            clusters[l].append(x)

    centroids: Dict[int, List[float]] = {}
    dispersions: Dict[int, float] = {}

    for l, pts in clusters.items():
        if not pts:
            centroids[l] = [0.0] * len(X[0])
            dispersions[l] = 0.0
            continue
        c = vector_mean(pts)
        centroids[l] = c
        # s_i: average distance of points in cluster to centroid
        dispersions[l] = sum(euclidean_dist(p, c) for p in pts) / len(pts)

    r_values: List[float] = []
    for i in unique_labels:
        max_r_ij = 0.0
        for j in unique_labels:
            if i == j:
                continue
            c_dist = euclidean_dist(centroids[i], centroids[j])
            if c_dist > 1e-12:
                r_ij = (dispersions[i] + dispersions[j]) / c_dist
                if r_ij > max_r_ij:
                    max_r_ij = r_ij
        r_values.append(max_r_ij)

    return sum(r_values) / k if k > 0 else 0.0


def compute_ref_calinski_harabasz(X: List[List[float]], labels: List[int]) -> float:
    """Compute Calinski-Harabasz Index (Variance Ratio Criterion)."""
    n = len(X)
    unique_labels = sorted(list(set(l for l in labels if l != -1)))
    k = len(unique_labels)
    if k < 2 or n <= k:
        return 0.0

    dim = len(X[0])
    global_mean = vector_mean(X)

    clusters: Dict[int, List[List[float]]] = {l: [] for l in unique_labels}
    for x, l in zip(X, labels):
        if l in clusters:
            clusters[l].append(x)

    # Between group sum of squares (trace(B_k))
    trace_b = 0.0
    # Within group sum of squares (trace(W_k))
    trace_w = 0.0

    for l, pts in clusters.items():
        n_k = len(pts)
        if n_k == 0:
            continue
        c_k = vector_mean(pts)
        trace_b += n_k * (euclidean_dist(c_k, global_mean) ** 2)
        for p in pts:
            trace_w += euclidean_dist(p, c_k) ** 2

    if trace_w < 1e-12:
        return 0.0

    ch = (trace_b / (k - 1)) / (trace_w / (n - k))
    return max(0.0, ch)


def compute_ref_hopkins(X: List[List[float]], m: int = 15, seed: int = 42) -> float:
    """Compute Hopkins Statistic for spatial clustering tendency."""
    n = len(X)
    dim = len(X[0])
    if n <= m:
        return 0.5

    rng = random.Random(seed)
    # Min/max bounds for synthetic uniform sampling
    mins = [min(row[d] for row in X) for d in range(dim)]
    maxs = [max(row[d] for row in X) for d in range(dim)]

    # Sample m synthetic uniform points U
    U = [[rng.uniform(mins[d], maxs[d]) for d in range(dim)] for _ in range(m)]

    # Sample m points from X
    indices = rng.sample(range(n), m)
    W = [X[i] for i in indices]

    # Sum of min distances from U to X
    sum_u = 0.0
    for u in U:
        min_d = min(euclidean_dist(u, x) for x in X)
        sum_u += min_d

    # Sum of min distances from W to X (excluding self)
    sum_w = 0.0
    for w_idx in indices:
        w_pt = X[w_idx]
        min_d = min(euclidean_dist(w_pt, X[j]) for j in range(n) if j != w_idx)
        sum_w += min_d

    denom = sum_u + sum_w
    return sum_u / denom if denom > 1e-12 else 0.5


def compute_ref_composite_fitness(
    silhouette: float,
    davies_bouldin: float,
    calinski_harabasz: float,
    stability_ari: float,
    noise_ratio: float = 0.0,
    cluster_sizes: Optional[List[int]] = None,
    weights: Tuple[float, float, float, float] = (0.40, 0.25, 0.15, 0.20),
) -> float:
    """Compute normalized composite fitness F(theta) in [0, 1]."""
    w1, w2, w3, w4 = weights

    # Normalized metrics
    s_norm = max(0.0, min(1.0, (silhouette + 1.0) / 2.0))
    db_norm = max(0.0, min(1.0, davies_bouldin / 5.0))
    inv_db = 1.0 - db_norm
    ch_norm = min(1.0, math.log1p(max(0.0, calinski_harabasz)) / math.log1p(10000.0))
    stab_norm = max(0.0, min(1.0, stability_ari))

    base_fitness = w1 * s_norm + w2 * inv_db + w3 * ch_norm + w4 * stab_norm

    # Penalties
    p_noise = 0.5 * max(0.0, noise_ratio - 0.10)
    p_imbalance = 0.0
    if cluster_sizes and len(cluster_sizes) > 1:
        total_pts = sum(cluster_sizes)
        if total_pts > 0:
            k = len(cluster_sizes)
            entropy = 0.0
            for sz in cluster_sizes:
                if sz > 0:
                    p_k = sz / total_pts
                    entropy -= p_k * math.log(p_k)
            max_entropy = math.log(k)
            if max_entropy > 1e-12:
                p_imbalance = 0.2 * max(0.0, 1.0 - (entropy / max_entropy))

    fitness = base_fitness - p_noise - p_imbalance
    return max(0.0, min(1.0, fitness))


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def synthetic_generator() -> SyntheticKaggleDatasetGenerator:
    """Instantiate a reproducible synthetic dataset generator."""
    return SyntheticKaggleDatasetGenerator(seed=42)


@pytest.fixture
def sample_dataset_records(synthetic_generator: SyntheticKaggleDatasetGenerator) -> List[Dict[str, Any]]:
    """500 realistic Kaggle credit card records."""
    return synthetic_generator.generate_records(n_rows=500)


@pytest.fixture
def sample_csv_string() -> str:
    """CSV text representation for ingestion and API upload."""
    return create_synthetic_dataset(n_rows=200, seed=42)


@pytest.fixture
def clean_numeric_matrix(synthetic_generator: SyntheticKaggleDatasetGenerator) -> Tuple[List[List[float]], List[str]]:
    """Clean (N x 17) numerical matrix and list of customer IDs."""
    return synthetic_generator.generate_matrix(n_rows=200, impute_strategy="median")


@pytest.fixture
def preset_personas() -> Dict[str, Dict[str, float]]:
    """Authoritative preset persona vectors for inference & profiler tests."""
    return {
        "transactor": {
            "BALANCE": 800.0,
            "BALANCE_FREQUENCY": 0.85,
            "PURCHASES": 2500.0,
            "ONEOFF_PURCHASES": 1800.0,
            "INSTALLMENTS_PURCHASES": 700.0,
            "CASH_ADVANCE": 0.0,
            "PURCHASES_FREQUENCY": 0.85,
            "ONEOFF_PURCHASES_FREQUENCY": 0.70,
            "PURCHASES_INSTALLMENTS_FREQUENCY": 0.50,
            "CASH_ADVANCE_FREQUENCY": 0.0,
            "CASH_ADVANCE_TRX": 0,
            "PURCHASES_TRX": 30,
            "CREDIT_LIMIT": 6000.0,
            "PAYMENTS": 2400.0,
            "MINIMUM_PAYMENTS": 120.0,
            "PRC_FULL_PAYMENT": 0.80,
            "TENURE": 12,
        },
        "revolver": {
            "BALANCE": 3500.0,
            "BALANCE_FREQUENCY": 0.95,
            "PURCHASES": 400.0,
            "ONEOFF_PURCHASES": 150.0,
            "INSTALLMENTS_PURCHASES": 250.0,
            "CASH_ADVANCE": 600.0,
            "PURCHASES_FREQUENCY": 0.25,
            "ONEOFF_PURCHASES_FREQUENCY": 0.10,
            "PURCHASES_INSTALLMENTS_FREQUENCY": 0.15,
            "CASH_ADVANCE_FREQUENCY": 0.30,
            "CASH_ADVANCE_TRX": 4,
            "PURCHASES_TRX": 6,
            "CREDIT_LIMIT": 4500.0,
            "PAYMENTS": 300.0,
            "MINIMUM_PAYMENTS": 110.0,
            "PRC_FULL_PAYMENT": 0.02,
            "TENURE": 12,
        },
        "cash_advance": {
            "BALANCE": 4500.0,
            "BALANCE_FREQUENCY": 0.98,
            "PURCHASES": 150.0,
            "ONEOFF_PURCHASES": 50.0,
            "INSTALLMENTS_PURCHASES": 100.0,
            "CASH_ADVANCE": 4000.0,
            "PURCHASES_FREQUENCY": 0.10,
            "ONEOFF_PURCHASES_FREQUENCY": 0.03,
            "PURCHASES_INSTALLMENTS_FREQUENCY": 0.07,
            "CASH_ADVANCE_FREQUENCY": 0.65,
            "CASH_ADVANCE_TRX": 12,
            "PURCHASES_TRX": 2,
            "CREDIT_LIMIT": 5000.0,
            "PAYMENTS": 1800.0,
            "MINIMUM_PAYMENTS": 180.0,
            "PRC_FULL_PAYMENT": 0.01,
            "TENURE": 12,
        },
        "inactive": {
            "BALANCE": 150.0,
            "BALANCE_FREQUENCY": 0.30,
            "PURCHASES": 30.0,
            "ONEOFF_PURCHASES": 30.0,
            "INSTALLMENTS_PURCHASES": 0.0,
            "CASH_ADVANCE": 0.0,
            "PURCHASES_FREQUENCY": 0.05,
            "ONEOFF_PURCHASES_FREQUENCY": 0.05,
            "PURCHASES_INSTALLMENTS_FREQUENCY": 0.0,
            "CASH_ADVANCE_FREQUENCY": 0.0,
            "CASH_ADVANCE_TRX": 0,
            "PURCHASES_TRX": 1,
            "CREDIT_LIMIT": 2000.0,
            "PAYMENTS": 40.0,
            "MINIMUM_PAYMENTS": 10.0,
            "PRC_FULL_PAYMENT": 1.0,
            "TENURE": 12,
        },
        "vip_spender": {
            "BALANCE": 5000.0,
            "BALANCE_FREQUENCY": 1.0,
            "PURCHASES": 9000.0,
            "ONEOFF_PURCHASES": 6000.0,
            "INSTALLMENTS_PURCHASES": 3000.0,
            "CASH_ADVANCE": 500.0,
            "PURCHASES_FREQUENCY": 0.95,
            "ONEOFF_PURCHASES_FREQUENCY": 0.85,
            "PURCHASES_INSTALLMENTS_FREQUENCY": 0.75,
            "CASH_ADVANCE_FREQUENCY": 0.10,
            "CASH_ADVANCE_TRX": 2,
            "PURCHASES_TRX": 75,
            "CREDIT_LIMIT": 16000.0,
            "PAYMENTS": 9500.0,
            "MINIMUM_PAYMENTS": 300.0,
            "PRC_FULL_PAYMENT": 0.60,
            "TENURE": 12,
        },
    }


@pytest.fixture
def metric_oracles() -> Dict[str, Any]:
    """Dictionary of reference oracle functions."""
    return {
        "silhouette": compute_ref_silhouette,
        "davies_bouldin": compute_ref_davies_bouldin,
        "calinski_harabasz": compute_ref_calinski_harabasz,
        "hopkins": compute_ref_hopkins,
        "composite_fitness": compute_ref_composite_fitness,
        "euclidean": euclidean_dist,
        "manhattan": manhattan_dist,
    }
