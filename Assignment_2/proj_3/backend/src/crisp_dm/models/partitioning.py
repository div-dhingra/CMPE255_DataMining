"""
CRISP-DM Phase 4: Partitioning Clustering Models
Implements K-Means (with k-means++ seeding and Lloyd updates) and K-Medoids (PAM / FasterPAM).
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy.spatial.distance import cdist
from .base import ClusteringModelBase


class KMeansModel(ClusteringModelBase):
    """
    K-Means clustering algorithm with k-means++ initialization and convergence tracking.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        super().__init__(model_name="kmeans")
        self.n_clusters = int(n_clusters)
        self.init = init.lower()
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _kmeans_plus_plus_init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=float)

        # Pick first center randomly
        first_idx = rng.integers(0, n_samples)
        centers[0] = X[first_idx]

        # Closest distance squared to any center
        closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

        for c_idx in range(1, self.n_clusters):
            total_dist_sq = np.sum(closest_dist_sq)
            if total_dist_sq == 0:
                probs = np.ones(n_samples) / n_samples
            else:
                probs = closest_dist_sq / total_dist_sq

            next_idx = rng.choice(n_samples, p=probs)
            centers[c_idx] = X[next_idx]

            # Update closest distances
            new_dist_sq = np.sum((X - centers[c_idx]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centers

    def _fit_single_run(self, X: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
        n_samples, n_features = X.shape

        if self.init == "k-means++":
            centers = self._kmeans_plus_plus_init(X, rng)
        else:
            init_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            centers = X[init_indices].copy()

        labels = np.zeros(n_samples, dtype=int)

        for _ in range(self.max_iter):
            # Compute squared Euclidean distances
            dist_matrix_sq = cdist(X, centers, metric="sqeuclidean")
            new_labels = np.argmin(dist_matrix_sq, axis=1)

            # Update centers
            new_centers = np.empty_like(centers)
            for k in range(self.n_clusters):
                cluster_points = X[new_labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = np.mean(cluster_points, axis=0)
                else:
                    # Reseed empty cluster with farthest point
                    farthest_idx = int(np.argmax(np.min(dist_matrix_sq, axis=1)))
                    new_centers[k] = X[farthest_idx]

            center_shift = np.max(np.sqrt(np.sum((new_centers - centers) ** 2, axis=1)))
            centers = new_centers
            labels = new_labels

            if center_shift < self.tol:
                break

        # Compute final inertia (WCSS)
        final_dists = cdist(X, centers, metric="sqeuclidean")
        inertia = float(np.sum(np.min(final_dists, axis=1)))
        return centers, labels, inertia

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples = len(X)
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} must be >= n_clusters={self.n_clusters}")

        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        best_centers = None
        best_labels = None

        for _ in range(self.n_init):
            centers, labels, inertia = self._fit_single_run(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        self.centroids_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_clusters_ = self.n_clusters
        self.is_fitted_ = True
        return self.labels_

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.centroids_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        X_new = np.asarray(X_new, dtype=float)
        dists = cdist(X_new, self.centroids_, metric="euclidean")
        return np.argmin(dists, axis=1)

    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.centroids_ is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_new = np.asarray(X_new, dtype=float)
        dists = cdist(X_new, self.centroids_, metric="euclidean")
        # Softmax over negative distances with temperature scaling
        tau = np.mean(dists) if np.mean(dists) > 0 else 1.0
        neg_dists = -dists / tau
        exp_vals = np.exp(neg_dists - np.max(neg_dists, axis=1, keepdims=True))
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return probs

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }


class KMedoidsModel(ClusteringModelBase):
    """
    K-Medoids clustering algorithm with FasterPAM / PAM optimization.
    Medoids are actual exemplar data points, providing robustness against extreme outliers.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        metric: str = "manhattan",
        method: str = "fasterpam",
        max_iter: int = 300,
        random_state: int = 42,
    ):
        super().__init__(model_name="kmedoids")
        self.n_clusters = int(n_clusters)
        self.metric = metric.lower()
        self.method = method.lower()
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoid_indices_: Optional[np.ndarray] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples = len(X)
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} must be >= n_clusters={self.n_clusters}")

        rng = np.random.default_rng(self.random_state)
        # Normalize metric name for scipy cdist
        scipy_metric = "cityblock" if self.metric == "manhattan" else self.metric
        # Compute pairwise distance matrix
        dist_matrix = cdist(X, X, metric=scipy_metric)

        # 1. Initialize medoids via k-medoids++ seeding
        first_idx = int(rng.integers(0, n_samples))
        medoids = [first_idx]
        closest_dists = dist_matrix[:, first_idx].copy()

        for _ in range(1, self.n_clusters):
            tot = np.sum(closest_dists)
            probs = closest_dists / tot if tot > 0 else np.ones(n_samples) / n_samples
            next_idx = int(rng.choice(n_samples, p=probs))
            medoids.append(next_idx)
            closest_dists = np.minimum(closest_dists, dist_matrix[:, next_idx])

        medoids = np.array(medoids, dtype=int)

        # 2. FasterPAM Eager Swap Loop
        improved = True
        iteration = 0

        while improved and iteration < self.max_iter:
            improved = False
            iteration += 1

            # Distances to current medoids: shape (n_samples, k)
            d_to_medoids = dist_matrix[:, medoids]
            # Nearest and second nearest medoid indices & distances
            sorted_idx = np.argsort(d_to_medoids, axis=1)
            nearest = sorted_idx[:, 0]
            d_first = d_to_medoids[np.arange(n_samples), nearest]
            d_second = d_to_medoids[np.arange(n_samples), sorted_idx[:, 1]]

            current_loss = np.sum(d_first)

            # Test candidate non-medoids
            non_medoids = np.setdiff1d(np.arange(n_samples), medoids)
            rng.shuffle(non_medoids)

            for cand in non_medoids[:min(len(non_medoids), 100)]:
                cand_dists = dist_matrix[:, cand]

                # Evaluate swap with each current medoid m_idx
                for m_idx in range(self.n_clusters):
                    # Change in loss if medoids[m_idx] is replaced by cand
                    loss_diff = 0.0
                    for i in range(n_samples):
                        d_cand = cand_dists[i]
                        if nearest[i] == m_idx:
                            # Object lost its nearest medoid
                            new_d = min(d_second[i], d_cand)
                            loss_diff += (new_d - d_first[i])
                        else:
                            # Object kept its nearest medoid unless candidate is closer
                            if d_cand < d_first[i]:
                                loss_diff += (d_cand - d_first[i])

                    if loss_diff < -1e-7:
                        medoids[m_idx] = cand
                        improved = True
                        break  # Eager swap (FasterPAM)
                if improved:
                    break

        # Final assignments
        d_to_final_medoids = dist_matrix[:, medoids]
        labels = np.argmin(d_to_final_medoids, axis=1)

        self.medoid_indices_ = medoids
        self.centroids_ = X[medoids].copy()
        self.labels_ = labels
        self.inertia_ = float(np.sum(np.min(d_to_final_medoids, axis=1)))
        self.n_clusters_ = self.n_clusters
        self.is_fitted_ = True
        return self.labels_

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.centroids_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        X_new = np.asarray(X_new, dtype=float)
        scipy_metric = "cityblock" if self.metric == "manhattan" else self.metric
        dists = cdist(X_new, self.centroids_, metric=scipy_metric)
        return np.argmin(dists, axis=1)

    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.centroids_ is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_new = np.asarray(X_new, dtype=float)
        scipy_metric = "cityblock" if self.metric == "manhattan" else self.metric
        dists = cdist(X_new, self.centroids_, metric=scipy_metric)
        tau = np.mean(dists) if np.mean(dists) > 0 else 1.0
        neg_dists = -dists / tau
        exp_vals = np.exp(neg_dists - np.max(neg_dists, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "metric": self.metric,
            "method": self.method,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
