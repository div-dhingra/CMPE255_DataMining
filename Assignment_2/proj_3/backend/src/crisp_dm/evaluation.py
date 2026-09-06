"""
CRISP-DM Phase 5: Evaluation Metrics Module
Provides internal and external validation metrics:
- Silhouette Coefficient (Rousseeuw 1987) with per-sample silhouette ribbons
- Davies-Bouldin Index (Davies & Bouldin 1979)
- Calinski-Harabasz Variance Ratio Criterion (Calinski & Harabasz 1974)
- Inertia WCSS curve with automated Kneedle Elbow Detection (Satopaa et al. 2011)
- Subsampling Bootstrap Stability with Adjusted Rand Index (ARI)
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import comb


def compute_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[float, np.ndarray]:
    """
    Computes Rousseeuw (1987) Silhouette score:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) is mean intra-cluster distance and b(i) is mean nearest-cluster distance.
    Ignores noise points (-1) in calculation.
    
    Returns:
        (global_mean_silhouette, sample_silhouettes_array)
    """
    X_arr = np.asarray(X, dtype=float)
    n_samples = len(X_arr)
    sample_silhouettes = np.zeros(n_samples, dtype=float)

    # Filter non-noise samples
    non_noise_mask = labels >= 0
    unique_labels = np.unique(labels[non_noise_mask])

    if len(unique_labels) < 2:
        return 0.0, sample_silhouettes

    # Subsample if requested for large datasets
    eval_indices = np.where(non_noise_mask)[0]
    if sample_size is not None and len(eval_indices) > sample_size:
        rng = np.random.default_rng(random_state)
        eval_indices = rng.choice(eval_indices, size=sample_size, replace=False)

    # Distance matrix between evaluated samples and all non-noise samples
    X_eval = X_arr[eval_indices]
    eval_labels = labels[eval_indices]
    dist_matrix = cdist(X_eval, X_arr[non_noise_mask], metric="euclidean")
    target_labels = labels[non_noise_mask]

    for row_idx, i in enumerate(eval_indices):
        c_i = eval_labels[row_idx]
        same_cluster_mask = target_labels == c_i

        # Intra-cluster distance a(i)
        n_same = np.sum(same_cluster_mask)
        if n_same > 1:
            a_i = np.sum(dist_matrix[row_idx, same_cluster_mask]) / (n_same - 1)
        else:
            a_i = 0.0

        # Nearest-cluster distance b(i)
        b_i = np.inf
        for c_other in unique_labels:
            if c_other != c_i:
                other_cluster_mask = target_labels == c_other
                n_other = np.sum(other_cluster_mask)
                if n_other > 0:
                    d_other = float(np.mean(dist_matrix[row_idx, other_cluster_mask]))
                    if d_other < b_i:
                        b_i = d_other

        if max(a_i, b_i) > 0 and b_i != np.inf:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0

        sample_silhouettes[i] = float(np.clip(s_i, -1.0, 1.0))

    valid_silhouettes = sample_silhouettes[eval_indices]
    global_score = float(np.mean(valid_silhouettes)) if len(valid_silhouettes) > 0 else 0.0
    return float(np.clip(global_score, -1.0, 1.0)), sample_silhouettes


def compute_davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes Davies-Bouldin Index (Davies & Bouldin 1979).
    DB = (1 / k) * sum_{i=1}^k max_{j != i} ( (s_i + s_j) / d(mu_i, mu_j) )
    Lower is better.
    """
    X_arr = np.asarray(X, dtype=float)
    non_noise_mask = labels >= 0
    unique_labels = np.unique(labels[non_noise_mask])
    k = len(unique_labels)

    if k < 2:
        return 0.0

    centroids = np.empty((k, X_arr.shape[1]))
    cluster_dispersions = np.empty(k)

    for idx, c in enumerate(unique_labels):
        pts = X_arr[labels == c]
        mu = np.mean(pts, axis=0)
        centroids[idx] = mu
        dists = np.sqrt(np.sum((pts - mu) ** 2, axis=1))
        cluster_dispersions[idx] = float(np.mean(dists)) if len(pts) > 0 else 0.0

    centroid_dists = cdist(centroids, centroids, metric="euclidean")
    # Avoid zero division on diagonal
    np.fill_diagonal(centroid_dists, np.inf)

    r_max = np.zeros(k)
    for i in range(k):
        r_ij = (cluster_dispersions[i] + cluster_dispersions) / centroid_dists[i]
        r_max[i] = np.max(r_ij)

    db_score = float(np.mean(r_max))
    return float(max(0.0, db_score))


def compute_calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes Calinski-Harabasz Index (Variance Ratio Criterion, Calinski & Harabasz 1974).
    CH = (Tr(B_k) / (k - 1)) / (Tr(W_k) / (n - k))
    Higher is better.
    """
    X_arr = np.asarray(X, dtype=float)
    non_noise_mask = labels >= 0
    X_valid = X_arr[non_noise_mask]
    labels_valid = labels[non_noise_mask]
    unique_labels = np.unique(labels_valid)
    k = len(unique_labels)
    n = len(X_valid)

    if k < 2 or n <= k:
        return 0.0

    global_mean = np.mean(X_valid, axis=0)

    # Between-group dispersion Tr(B_k)
    # Within-group dispersion Tr(W_k)
    tr_b = 0.0
    tr_w = 0.0

    for c in unique_labels:
        pts = X_valid[labels_valid == c]
        n_c = len(pts)
        if n_c > 0:
            mu_c = np.mean(pts, axis=0)
            tr_b += n_c * np.sum((mu_c - global_mean) ** 2)
            tr_w += np.sum((pts - mu_c) ** 2)

    if tr_w <= 1e-12:
        return 0.0

    ch_score = (tr_b / (k - 1)) / (tr_w / (n - k))
    return float(max(0.0, ch_score))


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Computes the Adjusted Rand Index (ARI) between two clusterings.
    ARI in [-1, 1], with 1.0 indicating identical clusterings.
    """
    labels_a = np.asarray(labels_true)
    labels_b = np.asarray(labels_pred)
    n = len(labels_a)

    if n <= 1:
        return 1.0

    # Build contingency table
    classes_a, idx_a = np.unique(labels_a, return_inverse=True)
    classes_b, idx_b = np.unique(labels_b, return_inverse=True)
    n_a = len(classes_a)
    n_b = len(classes_b)

    contingency = np.zeros((n_a, n_b), dtype=np.int64)
    for i in range(n):
        contingency[idx_a[i], idx_b[i]] += 1

    # Sum of n_ij choose 2
    sum_nij_comb = np.sum(contingency * (contingency - 1) // 2)

    # Sum of a_i choose 2
    a_sums = np.sum(contingency, axis=1)
    sum_a_comb = np.sum(a_sums * (a_sums - 1) // 2)

    # Sum of b_j choose 2
    b_sums = np.sum(contingency, axis=0)
    sum_b_comb = np.sum(b_sums * (b_sums - 1) // 2)

    total_pairs = n * (n - 1) // 2
    expected_index = (sum_a_comb * sum_b_comb) / total_pairs
    max_index = 0.5 * (sum_a_comb + sum_b_comb)

    denom = max_index - expected_index
    if denom == 0:
        return 1.0 if sum_nij_comb == expected_index else 0.0

    ari = (sum_nij_comb - expected_index) / denom
    return float(np.clip(ari, -1.0, 1.0))


def compute_inertia_elbow_curve(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes K-Means inertia across k in [k_min, k_max] and applies
    the Kneedle algorithm (Satopaa et al. 2011) to automatically locate the elbow k*.
    """
    from .models.partitioning import KMeansModel

    X_arr = np.asarray(X, dtype=float)
    k_values = list(range(k_min, k_max + 1))
    inertias = []

    for k in k_values:
        model = KMeansModel(n_clusters=k, n_init=5, max_iter=200, random_state=random_state)
        model.fit_predict(X_arr)
        inertias.append(float(model.inertia_))

    # Kneedle Algorithm for elbow detection:
    # 1. Normalize k and inertia to [0, 1]
    k_arr = np.array(k_values, dtype=float)
    in_arr = np.array(inertias, dtype=float)

    x_norm = (k_arr - k_arr[0]) / (k_arr[-1] - k_arr[0])
    in_range = in_arr[0] - in_arr[-1]
    y_norm = (in_arr - in_arr[-1]) / in_range if in_range > 0 else np.zeros_like(in_arr)

    # 2. Difference curve from the diagonal line connecting (0, 1) to (1, 0)
    # Line connecting (x_norm[0], y_norm[0]) to (x_norm[-1], y_norm[-1]): y = 1 - x
    # Distance to diagonal: diff = y_norm - (1 - x_norm) = y_norm + x_norm - 1
    # For a convex decreasing curve, difference is (1 - x_norm) - y_norm or perpendicular distance
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    distances = []
    for i in range(len(k_values)):
        p = np.array([x_norm[i], y_norm[i]])
        # 2D cross product formula: |v_x * (p1_y - p_y) - v_y * (p1_x - p_x)| / ||v||
        cross_2d = np.abs(line_vec[0] * (p1[1] - p[1]) - line_vec[1] * (p1[0] - p[0]))
        dist = cross_2d / (line_len + 1e-12)
        distances.append(float(dist))

    elbow_idx = int(np.argmax(distances))
    elbow_k = k_values[elbow_idx]

    return {
        "k_values": k_values,
        "inertias": [round(val, 4) for val in inertias],
        "elbow_k": int(elbow_k),
        "normalized_differences": [round(d, 4) for d in distances],
    }


def compute_subsampling_stability(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    n_bootstraps: int = 10,
    subsample_ratio: float = 0.8,
    random_state: int = 42,
) -> float:
    """
    Evaluates cluster stability via bootstrap subsampling and Adjusted Rand Index (ARI).
    Subsamples the dataset B times, clusters each subsample, and compares cluster labels
    on the overlapping samples with the full-dataset clustering.
    """
    X_arr = np.asarray(X, dtype=float)
    n_samples = len(X_arr)
    rng = np.random.default_rng(random_state)

    # Full dataset baseline model
    base_model = model_factory()
    base_labels = base_model.fit_predict(X_arr)

    subsample_size = int(max(10, np.round(n_samples * subsample_ratio)))
    ari_scores = []

    for b in range(n_bootstraps):
        sub_indices = rng.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X_arr[sub_indices]
        
        boot_model = model_factory()
        boot_labels = boot_model.fit_predict(X_sub)

        # Compare labels on the subsampled subset
        base_sub_labels = base_labels[sub_indices]
        ari = adjusted_rand_index(base_sub_labels, boot_labels)
        ari_scores.append(ari)

    return float(np.mean(ari_scores)) if len(ari_scores) > 0 else 1.0


def evaluate_clustering_solution(
    X: np.ndarray,
    labels: np.ndarray,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Computes a comprehensive evaluation summary for a clustering solution:
    Silhouette score, Davies-Bouldin index, Calinski-Harabasz score, cluster counts,
    noise counts, inertia, BIC, AIC.
    """
    X_arr = np.asarray(X, dtype=float)
    sil_score, sample_sils = compute_silhouette_score(X_arr, labels)
    db_index = compute_davies_bouldin_index(X_arr, labels)
    ch_score = compute_calinski_harabasz_score(X_arr, labels)

    unique_labels = np.unique(labels)
    n_clusters = int(len(unique_labels[unique_labels >= 0]))
    n_noise = int(np.sum(labels == -1))
    noise_ratio = float(n_noise / len(labels)) if len(labels) > 0 else 0.0

    cluster_sizes = {int(c): int(np.sum(labels == c)) for c in unique_labels}

    res: Dict[str, Any] = {
        "silhouette_score": round(sil_score, 4),
        "davies_bouldin_index": round(db_index, 4),
        "calinski_harabasz_score": round(ch_score, 4),
        "n_clusters": n_clusters,
        "noise_points_count": n_noise,
        "noise_ratio": round(noise_ratio, 4),
        "cluster_sizes": cluster_sizes,
        "sample_silhouettes": [round(float(s), 4) for s in sample_sils],
    }

    if model is not None:
        if hasattr(model, "inertia_") and model.inertia_ is not None:
            res["inertia"] = round(float(model.inertia_), 4)
        if hasattr(model, "bic_") and model.bic_ is not None:
            res["bic"] = round(float(model.bic_), 4)
        if hasattr(model, "aic_") and model.aic_ is not None:
            res["aic"] = round(float(model.aic_), 4)

    return res
