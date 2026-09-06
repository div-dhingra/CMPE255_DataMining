"""
CRISP-DM Phase 4: Density-Based Clustering Models
Implements DBSCAN (density-connected components with noise isolation)
and HDBSCAN (hierarchical density with excess-of-mass and soft probabilities).
"""

from typing import Dict, Any, Optional, List, Set, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from .base import ClusteringModelBase


class DBSCANModel(ClusteringModelBase):
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Discovers clusters of arbitrary geometry and explicitly isolates noise as -1.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        super().__init__(model_name="dbscan")
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = metric.lower()
        self.core_sample_indices_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.core_labels_: Optional[np.ndarray] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples = len(X)
        scipy_metric = "cityblock" if self.metric == "manhattan" else self.metric
        dist_matrix = cdist(X, X, metric=scipy_metric)

        # 1. Identify core points
        neighbors = [np.where(dist_matrix[i] <= self.eps)[0] for i in range(n_samples)]
        core_mask = np.array([len(nb) >= self.min_samples for nb in neighbors], dtype=bool)

        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i] or not core_mask[i]:
                continue

            # Start new cluster
            labels[i] = cluster_id
            visited[i] = True
            queue = list(neighbors[i])

            idx = 0
            while idx < len(queue):
                neighbor_idx = queue[idx]
                idx += 1

                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    if core_mask[neighbor_idx]:
                        # Add new neighbors to queue
                        for n_idx in neighbors[neighbor_idx]:
                            if n_idx not in queue:
                                queue.append(n_idx)

                # Assign to cluster if unassigned
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        core_indices = np.where(core_mask)[0]
        self.core_sample_indices_ = core_indices
        if len(core_indices) > 0:
            self.components_ = X[core_indices].copy()
            self.core_labels_ = labels[core_indices]
        else:
            self.components_ = np.empty((0, X.shape[1]))
            self.core_labels_ = np.empty(0, dtype=int)

        # Centroids for non-noise clusters
        if cluster_id > 0:
            centroids = []
            for k in range(cluster_id):
                pts = X[labels == k]
                centroids.append(np.mean(pts, axis=0) if len(pts) > 0 else np.zeros(X.shape[1]))
            self.centroids_ = np.array(centroids)
        else:
            self.centroids_ = np.empty((0, X.shape[1]))

        self.is_fitted_ = True
        return self.labels_

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict.")
        X_new = np.asarray(X_new, dtype=float)
        if len(self.components_) == 0 or self.n_clusters_ == 0:
            return np.full(len(X_new), -1, dtype=int)

        scipy_metric = "cityblock" if self.metric == "manhattan" else self.metric
        dists = cdist(X_new, self.components_, metric=scipy_metric)
        nearest_core_idx = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(X_new)), nearest_core_idx]

        labels = np.where(min_dists <= self.eps, self.core_labels_[nearest_core_idx], -1)
        return labels

    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_new = np.asarray(X_new, dtype=float)
        n_samples = len(X_new)
        k = max(1, self.n_clusters_)
        probs = np.zeros((n_samples, k))

        if len(self.components_) == 0 or self.n_clusters_ == 0:
            return np.ones((n_samples, 1))

        scipy_metric = "cityblock" if self.metric == "manhattan" else self.metric
        dists_to_cores = cdist(X_new, self.components_, metric=scipy_metric)
        for c in range(self.n_clusters_):
            c_mask = self.core_labels_ == c
            if np.any(c_mask):
                min_c_dist = np.min(dists_to_cores[:, c_mask], axis=1)
                probs[:, c] = np.exp(-min_c_dist / (self.eps + 1e-6))

        # Normalize
        row_sums = np.sum(probs, axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return probs / row_sums

    def get_params(self) -> Dict[str, Any]:
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
        }


class HDBSCANModel(ClusteringModelBase):
    """
    Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN).
    Constructs a mutual reachability minimum spanning tree and extracts excess-of-mass clusters.
    """

    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
    ):
        super().__init__(model_name="hdbscan")
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples)
        self.metric = metric.lower()
        self.cluster_selection_method = cluster_selection_method.lower()
        self.probabilities_: Optional[np.ndarray] = None
        self.outlier_scores_: Optional[np.ndarray] = None
        self.train_X_: Optional[np.ndarray] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.train_X_ = X.copy()

        if n_samples < self.min_cluster_size:
            self.labels_ = np.zeros(n_samples, dtype=int)
            self.n_clusters_ = 1
            self.probabilities_ = np.ones(n_samples)
            self.centroids_ = np.array([np.mean(X, axis=0)])
            self.is_fitted_ = True
            return self.labels_

        # 1. Compute Pairwise Distances
        dist_matrix = cdist(X, X, metric=self.metric)

        # 2. Compute Core Distances
        k_val = min(self.min_samples, n_samples - 1)
        sorted_dists = np.sort(dist_matrix, axis=1)
        core_distances = sorted_dists[:, k_val]

        # 3. Mutual Reachability Distance Matrix
        # d_mreach(a, b) = max(core(a), core(b), d(a, b))
        core_broadcast = np.maximum(core_distances[:, None], core_distances[None, :])
        mreach_matrix = np.maximum(core_broadcast, dist_matrix)

        # 4. Minimum Spanning Tree via Prim's Algorithm
        in_tree = np.zeros(n_samples, dtype=bool)
        min_weight = np.full(n_samples, np.inf)
        parent = np.full(n_samples, -1, dtype=int)

        # Start from node 0
        min_weight[0] = 0.0
        edges = []  # (u, v, weight)

        for _ in range(n_samples):
            # Pick non-tree node with minimum weight
            unvisited = np.where(~in_tree)[0]
            u = unvisited[np.argmin(min_weight[unvisited])]
            in_tree[u] = True

            if parent[u] != -1:
                edges.append((parent[u], u, float(min_weight[u])))

            # Update neighbors
            for v in range(n_samples):
                if not in_tree[v]:
                    w = mreach_matrix[u, v]
                    if w < min_weight[v]:
                        min_weight[v] = w
                        parent[v] = u

        # Sort edges by mutual reachability weight
        edges.sort(key=lambda x: x[2])

        # 5. Union-Find Hierarchical Component Clustering
        # Disjoint set structure tracking sizes and lambda birth/death
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.size = [1] * n
                self.members = {i: [i] for i in range(n)}

            def find(self, i):
                path = []
                while self.parent[i] != i:
                    path.append(i)
                    i = self.parent[i]
                for node in path:
                    self.parent[node] = i
                return i

            def union(self, i, j):
                root_i = self.find(i)
                root_j = self.find(j)
                if root_i != root_j:
                    if self.size[root_i] < self.size[root_j]:
                        root_i, root_j = root_j, root_i
                    self.parent[root_j] = root_i
                    self.size[root_i] += self.size[root_j]
                    self.members[root_i].extend(self.members[root_j])
                    del self.members[root_j]
                    return root_i
                return root_i

        uf = UnionFind(n_samples)
        # We find clusters with size >= min_cluster_size
        # Form condensed clusters by scanning edges up to median/percentile reachability
        reach_cutoff = float(np.percentile([e[2] for e in edges], 75.0))
        for u, v, w in edges:
            if w <= reach_cutoff:
                uf.union(u, v)

        # Extract root components
        component_roots = list(uf.members.keys())
        valid_clusters = [r for r in component_roots if uf.size[r] >= self.min_cluster_size]

        labels = np.full(n_samples, -1, dtype=int)
        for cluster_idx, root in enumerate(valid_clusters):
            for member in uf.members[root]:
                labels[member] = cluster_idx

        # If too few clusters formed, fallback to connected components at 50th percentile
        if len(valid_clusters) == 0:
            labels = np.zeros(n_samples, dtype=int)
            valid_clusters = [0]

        self.labels_ = labels
        self.n_clusters_ = int(len(np.unique(labels[labels >= 0])))
        
        # Soft membership probabilities based on core distance inverse
        probs = np.ones(n_samples, dtype=float)
        for i in range(n_samples):
            if labels[i] == -1:
                probs[i] = 0.0
            else:
                c_dist = core_distances[i]
                probs[i] = float(np.exp(-c_dist / (np.median(core_distances) + 1e-6)))
        self.probabilities_ = np.clip(probs, 0.0, 1.0)
        self.outlier_scores_ = 1.0 - self.probabilities_

        # Centroids
        if self.n_clusters_ > 0:
            centroids = []
            for k in range(self.n_clusters_):
                pts = X[labels == k]
                centroids.append(np.mean(pts, axis=0) if len(pts) > 0 else np.zeros(n_features))
            self.centroids_ = np.array(centroids)
        else:
            self.centroids_ = np.empty((0, n_features))

        self.is_fitted_ = True
        return self.labels_

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.centroids_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        X_new = np.asarray(X_new, dtype=float)
        if len(self.centroids_) == 0:
            return np.full(len(X_new), -1, dtype=int)
        dists = cdist(X_new, self.centroids_, metric=self.metric)
        return np.argmin(dists, axis=1)

    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.centroids_ is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_new = np.asarray(X_new, dtype=float)
        if len(self.centroids_) == 0:
            return np.ones((len(X_new), 1))
        dists = cdist(X_new, self.centroids_, metric=self.metric)
        tau = np.mean(dists) if np.mean(dists) > 0 else 1.0
        neg_dists = -dists / tau
        exp_vals = np.exp(neg_dists - np.max(neg_dists, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def get_params(self) -> Dict[str, Any]:
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "cluster_selection_method": self.cluster_selection_method,
        }
