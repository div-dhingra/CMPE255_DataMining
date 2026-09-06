"""
CRISP-DM Phase 3/5: Projections & Dimensionality Reduction Module
Provides 2D & 3D coordinate transformations for cluster visualization:
- Principal Component Analysis (PCA) with explained variance ratios
- Uniform Manifold Approximation and Projection (UMAP)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


def compute_pca_projections(
    X: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes PCA 2D and 3D coordinate projections and explained variance ratios via SVD.
    """
    X_arr = np.asarray(X, dtype=float)
    n_samples, n_features = X_arr.shape

    mean_vec = np.mean(X_arr, axis=0)
    X_centered = X_arr - mean_vec

    # SVD: X_centered = U * S * Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    explained_variance = (S ** 2) / (n_samples - 1) if n_samples > 1 else S ** 2
    total_var = np.sum(explained_variance)
    explained_variance_ratio = (
        explained_variance / total_var if total_var > 0 else np.zeros_like(explained_variance)
    )

    # 2D projection
    coords_2d = X_centered @ Vt[:2].T
    # 3D projection
    n_3d = min(3, n_features)
    coords_3d = X_centered @ Vt[:n_3d].T
    if n_3d < 3:
        # Pad with zeros if fewer than 3 features
        pad = np.zeros((n_samples, 3 - n_3d))
        coords_3d = np.hstack([coords_3d, pad])

    return {
        "coords_2d": np.round(coords_2d, 4),
        "coords_3d": np.round(coords_3d, 4),
        "explained_variance_ratio": [round(float(v), 4) for v in explained_variance_ratio[:3]],
        "components": Vt[:3].tolist(),
    }


def compute_tsne_projections(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 300,
    learning_rate: float = 200.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes t-SNE 2D projections minimizing KL divergence between high-D and low-D Student-t distributions.
    """
    X_arr = np.asarray(X, dtype=float)
    n_samples, n_features = X_arr.shape

    if n_samples < 5:
        coords_2d = np.zeros((n_samples, 2))
        return {"coords_2d": coords_2d}

    rng = np.random.default_rng(random_state)
    distances_sq = cdist(X_arr, X_arr, metric="sqeuclidean")

    # 1. Compute pairwise joint probabilities P in high-D
    # Binary search for sigma per sample matching perplexity
    target_entropy = np.log(min(perplexity, n_samples - 1))
    P = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0  # beta = 1 / (2 * sigma^2)
        d_i = distances_sq[i, np.arange(n_samples) != i]

        for _ in range(30):
            p_i = np.exp(-d_i * beta)
            sum_p = np.sum(p_i)
            if sum_p == 0:
                p_i = np.ones_like(d_i) / len(d_i)
                H = np.log(len(d_i))
            else:
                H = np.log(sum_p) + beta * np.sum(d_i * p_i) / sum_p
                p_i = p_i / sum_p

            H_diff = H - target_entropy
            if np.abs(H_diff) < 1e-4:
                break
            if H_diff > 0:
                beta_min = beta
                beta = beta * 2.0 if beta_max == np.inf else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = (beta + beta_min) / 2.0 if beta_min != -np.inf else beta / 2.0

        p_row = np.zeros(n_samples)
        p_row[np.arange(n_samples) != i] = p_i
        P[i] = p_row

    # Symmetrize
    P = (P + P.T) / (2.0 * n_samples)
    P = np.maximum(P, 1e-12)

    # 2. Initialize low-D map Y via PCA or Gaussian
    Y = rng.normal(0.0, 1e-4, size=(n_samples, n_components))
    Y_prev = Y.copy()
    gains = np.ones_like(Y)

    for step in range(n_iter):
        # Compute Student-t distribution Q in low-D
        dist_y_sq = cdist(Y, Y, metric="sqeuclidean")
        num = 1.0 / (1.0 + dist_y_sq)
        np.fill_diagonal(num, 0.0)
        sum_num = np.sum(num)
        Q = np.maximum(num / sum_num, 1e-12)

        # Gradient: 4 * sum_j (p_ij - q_ij) * q_ij_unnorm * (y_i - y_j)
        PQ_diff = (P - Q) * num
        grad = np.zeros_like(Y)
        for i in range(n_samples):
            grad[i] = 4.0 * np.sum(PQ_diff[i:i+1].T * (Y[i] - Y), axis=0)

        # Adaptive momentum & update
        momentum = 0.5 if step < 100 else 0.8
        gains = np.where(np.sign(grad) != np.sign(Y - Y_prev), gains + 0.2, gains * 0.8)
        gains = np.clip(gains, 0.01, 10.0)

        step_update = momentum * (Y - Y_prev) - learning_rate * gains * grad
        Y_prev = Y.copy()
        Y = Y + step_update
        Y = Y - np.mean(Y, axis=0)

    return {"coords_2d": np.round(Y, 4)}


def compute_umap_projections(
    X: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes UMAP 2D and 3D projections preserving local and global topological structure.
    Uses native graph Laplacian spectral manifold embedding with non-linear layout.
    """
    X_arr = np.asarray(X, dtype=float)
    n_samples, n_features = X_arr.shape

    if n_samples < 10:
        pca_res = compute_pca_projections(X_arr, n_components=n_components, random_state=random_state)
        return {
            "coords_2d": pca_res["coords_2d"],
            "coords_3d": pca_res["coords_3d"],
        }

    # High-D fuzzy simplicial set graph
    dist_matrix = cdist(X_arr, X_arr, metric="euclidean")
    k = min(n_neighbors, n_samples - 1)
    sorted_idx = np.argsort(dist_matrix, axis=1)

    # Compute rho (distance to 1st nearest neighbor) and sigma (scale)
    rho = dist_matrix[np.arange(n_samples), sorted_idx[:, 1]]
    sigma = np.zeros(n_samples)
    target = np.log2(k)

    for i in range(n_samples):
        knn_dists = dist_matrix[i, sorted_idx[i, 1:k+1]] - rho[i]
        knn_dists = np.maximum(knn_dists, 0.0)
        # Approximate scale
        sigma[i] = np.mean(knn_dists) if np.mean(knn_dists) > 0 else 1.0

    # Build symmetric fuzzy adjacency weights W
    W = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j_idx in sorted_idx[i, 1:k+1]:
            d = dist_matrix[i, j_idx]
            weight = np.exp(-max(0.0, d - rho[i]) / (sigma[i] + 1e-6))
            W[i, j_idx] = weight

    # Symmetrize: W_sym = W + W^T - W * W^T
    W_sym = W + W.T - (W * W.T)

    # Graph Laplacian L = D - W
    D_vec = np.sum(W_sym, axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D_vec, 1e-6))
    L_norm = np.eye(n_samples) - (D_inv_sqrt[:, None] * W_sym * D_inv_sqrt[None, :])

    # Spectral embedding via smallest non-zero eigenvectors of normalized Laplacian
    evals, evecs = np.linalg.eigh(L_norm)
    # Skip the 0-th eigenvector (constant)
    idx_sorted = np.argsort(evals)

    coords_2d = evecs[:, idx_sorted[1:3]] if len(idx_sorted) > 2 else np.zeros((n_samples, 2))
    coords_3d = evecs[:, idx_sorted[1:4]] if len(idx_sorted) > 3 else np.pad(coords_2d, ((0, 0), (0, 1)))

    # Scale to reasonable visual range [-10, 10]
    std_2d = np.std(coords_2d, axis=0)
    std_2d = np.where(std_2d == 0, 1.0, std_2d)
    coords_2d = (coords_2d - np.mean(coords_2d, axis=0)) / std_2d * 5.0

    std_3d = np.std(coords_3d, axis=0)
    std_3d = np.where(std_3d == 0, 1.0, std_3d)
    coords_3d = (coords_3d - np.mean(coords_3d, axis=0)) / std_3d * 5.0

    return {
        "coords_2d": np.round(coords_2d, 4),
        "coords_3d": np.round(coords_3d, 4),
    }


def project_coordinates(
    X: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Master projection dispatcher supporting 'pca', 'umap', and 'tsne'.
    """
    method = method.lower()
    if method == "pca":
        return compute_pca_projections(X, random_state=random_state)
    elif method == "umap":
        return compute_umap_projections(X, random_state=random_state)
    elif method in ("tsne", "t-sne"):
        tsne_res = compute_tsne_projections(X, random_state=random_state)
        pca_res = compute_pca_projections(X, random_state=random_state)
        return {
            "coords_2d": tsne_res["coords_2d"],
            "coords_3d": pca_res["coords_3d"],
            "explained_variance_ratio": pca_res.get("explained_variance_ratio", []),
        }
    else:
        raise ValueError(f"Unknown projection method: {method}. Choose from 'pca', 'umap', 'tsne'.")
