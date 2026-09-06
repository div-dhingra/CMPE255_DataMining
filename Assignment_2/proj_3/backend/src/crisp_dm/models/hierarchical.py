"""
CRISP-DM Phase 4: Hierarchical Clustering Models
Implements Agglomerative Clustering supporting Ward, Complete, Average, and Single linkages.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from scipy.spatial.distance import cdist
from .base import ClusteringModelBase


class AgglomerativeModel(ClusteringModelBase):
    """
    Agglomerative Hierarchical Clustering.
    Builds a bottom-up cluster tree iteratively merging cluster pairs according to linkage criteria.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        linkage: str = "ward",
        metric: str = "euclidean",
    ):
        super().__init__(model_name="agglomerative")
        self.n_clusters = int(n_clusters)
        self.linkage = linkage.lower()
        self.metric = metric.lower()
        if self.linkage == "ward" and self.metric != "euclidean":
            raise ValueError("Ward linkage requires metric='euclidean'.")
        self.linkage_matrix_: Optional[np.ndarray] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} must be >= n_clusters={self.n_clusters}")

        scipy_metric = "cityblock" if self.metric == "manhattan" else self.metric

        # Active cluster representations: map cluster_id -> list of sample indices
        active_clusters = {i: [i] for i in range(n_samples)}
        cluster_sizes = {i: 1 for i in range(n_samples)}
        cluster_centroids = {i: X[i].copy() for i in range(n_samples)}

        # Initial pairwise distances
        dist_matrix = cdist(X, X, metric=scipy_metric)
        np.fill_diagonal(dist_matrix, np.inf)

        # Distance cache between active cluster IDs
        cluster_ids = list(range(n_samples))
        current_cluster_dists = dist_matrix.copy()

        next_cluster_id = n_samples
        merge_history = []  # [c1, c2, dist, size]

        while len(active_clusters) > self.n_clusters:
            # Find closest pair of active clusters
            min_dist = np.inf
            best_pair = (-1, -1)
            active_list = list(active_clusters.keys())
            n_act = len(active_list)

            for i_idx in range(n_act):
                c_i = active_list[i_idx]
                for j_idx in range(i_idx + 1, n_act):
                    c_j = active_list[j_idx]

                    if self.linkage == "ward":
                        n_i = cluster_sizes[c_i]
                        n_j = cluster_sizes[c_j]
                        diff = cluster_centroids[c_i] - cluster_centroids[c_j]
                        d = float((n_i * n_j) / (n_i + n_j) * np.sum(diff ** 2))
                    elif self.linkage == "complete":
                        pts_i = active_clusters[c_i]
                        pts_j = active_clusters[c_j]
                        d = float(np.max(dist_matrix[np.ix_(pts_i, pts_j)]))
                    elif self.linkage == "average":
                        pts_i = active_clusters[c_i]
                        pts_j = active_clusters[c_j]
                        d = float(np.mean(dist_matrix[np.ix_(pts_i, pts_j)]))
                    else:  # single
                        pts_i = active_clusters[c_i]
                        pts_j = active_clusters[c_j]
                        d = float(np.min(dist_matrix[np.ix_(pts_i, pts_j)]))

                    if d < min_dist:
                        min_dist = d
                        best_pair = (c_i, c_j)

            c_a, c_b = best_pair
            # Merge c_a and c_b into next_cluster_id
            merged_members = active_clusters[c_a] + active_clusters[c_b]
            merged_size = cluster_sizes[c_a] + cluster_sizes[c_b]
            merged_centroid = (
                cluster_centroids[c_a] * cluster_sizes[c_a] + cluster_centroids[c_b] * cluster_sizes[c_b]
            ) / merged_size

            merge_history.append([c_a, c_b, min_dist, merged_size])

            del active_clusters[c_a]
            del active_clusters[c_b]
            del cluster_sizes[c_a]
            del cluster_sizes[c_b]
            del cluster_centroids[c_a]
            del cluster_centroids[c_b]

            active_clusters[next_cluster_id] = merged_members
            cluster_sizes[next_cluster_id] = merged_size
            cluster_centroids[next_cluster_id] = merged_centroid

            next_cluster_id += 1

        # Assign final cluster labels [0, ..., n_clusters-1]
        labels = np.zeros(n_samples, dtype=int)
        final_centroids = []
        for new_label, c_id in enumerate(active_clusters.keys()):
            for member_idx in active_clusters[c_id]:
                labels[member_idx] = new_label
            final_centroids.append(cluster_centroids[c_id])

        self.labels_ = labels
        self.centroids_ = np.array(final_centroids)
        self.n_clusters_ = self.n_clusters
        self.linkage_matrix_ = np.array(merge_history)
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
            "linkage": self.linkage,
            "metric": self.metric,
        }
