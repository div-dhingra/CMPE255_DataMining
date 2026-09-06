"""
Autoresearch Composite Multi-Objective Fitness Evaluation Module.
Computes normalized scalar fitness:
    F(theta) = w1 * S_norm + w2 * DB_norm + w3 * CH_norm + w4 * ARI_stability - P_noise - P_imbalance

Handles edge cases (degenerate clusterings, single cluster, all noise, exceptions)
with graceful penalty assignment (-1.0).
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd

from crisp_dm.data_preparation import DataPreparationPipeline, PipelineConfig
from crisp_dm.models.base import ClusteringModelBase
from crisp_dm.models.partitioning import KMeansModel, KMedoidsModel
from crisp_dm.models.density import DBSCANModel, HDBSCANModel
from crisp_dm.models.hierarchical import AgglomerativeModel
from crisp_dm.models.probabilistic import GaussianMixtureModel
from crisp_dm.evaluation import (
    compute_silhouette_score,
    compute_davies_bouldin_index,
    compute_calinski_harabasz_score,
    adjusted_rand_index,
)


@dataclass
class ObjectiveWeights:
    """Multi-objective fitness component weights (sum to 1.0)."""
    silhouette: float = 0.40
    davies_bouldin: float = 0.25
    calinski_harabasz: float = 0.15
    stability: float = 0.20

    def __post_init__(self):
        total = self.silhouette + self.davies_bouldin + self.calinski_harabasz + self.stability
        if total > 0 and abs(total - 1.0) > 1e-4:
            self.silhouette /= total
            self.davies_bouldin /= total
            self.calinski_harabasz /= total
            self.stability /= total


@dataclass
class ObjectiveEvaluation:
    """Detailed result of evaluating candidate pipeline theta."""
    fitness: float
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability_ari: float
    n_clusters: int
    noise_ratio: float
    noise_penalty: float
    imbalance_penalty: float
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    normalized_metrics: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)
    labels: Optional[np.ndarray] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitness": round(self.fitness, 4),
            "silhouette": round(self.silhouette, 4),
            "davies_bouldin": round(self.davies_bouldin, 4),
            "calinski_harabasz": round(self.calinski_harabasz, 4),
            "stability_ari": round(self.stability_ari, 4),
            "n_clusters": self.n_clusters,
            "noise_ratio": round(self.noise_ratio, 4),
            "noise_penalty": round(self.noise_penalty, 4),
            "imbalance_penalty": round(self.imbalance_penalty, 4),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "error": self.error,
        }


class CompositeObjective:
    """
    Evaluates end-to-end clustering pipelines under configuration theta and computes
    a normalized composite fitness score balancing cluster cohesion, separation, density,
    stability, and balance.
    """

    def __init__(
        self,
        dataset: Union[pd.DataFrame, np.ndarray],
        weights: Optional[ObjectiveWeights] = None,
        stability_bootstraps: int = 3,
        subsample_ratio: float = 0.8,
        random_state: int = 42,
        cache_preprocessing: bool = True,
    ):
        if isinstance(dataset, np.ndarray):
            # Wrap array in dataframe
            cols = [f"feat_{i}" for i in range(dataset.shape[1])]
            self.dataset = pd.DataFrame(dataset, columns=cols)
        else:
            self.dataset = dataset.copy()

        self.weights = weights or ObjectiveWeights()
        self.stability_bootstraps = max(1, stability_bootstraps)
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state
        self.cache_preprocessing = cache_preprocessing
        self._prep_cache: Dict[str, np.ndarray] = {}

    def _get_prep_cache_key(self, prep_config: Dict[str, Any]) -> str:
        items = sorted([(k, str(v)) for k, v in prep_config.items()])
        return "_".join(f"{k}:{v}" for k, v in items)

    def prepare_features(self, prep_config: Dict[str, Any]) -> np.ndarray:
        """
        Executes data preparation pipeline and dimensionality reduction for a preprocessing config.
        """
        cache_key = self._get_prep_cache_key(prep_config)
        if self.cache_preprocessing and cache_key in self._prep_cache:
            return self._prep_cache[cache_key].copy()

        # Build PipelineConfig
        imputer_strat = prep_config.get("imputer", "median")
        outlier_strat = prep_config.get("outlier_handler", "winsorize")
        scaling_strat = prep_config.get("scaler", "yeo_johnson")
        engineer_ratios = bool(prep_config.get("feature_engineering", True))

        pipeline_config = PipelineConfig(
            imputer_strategy=imputer_strat,
            outlier_strategy=outlier_strat,
            scaling_strategy=scaling_strat,
            engineer_ratios=engineer_ratios,
        )

        pipeline = DataPreparationPipeline(config=pipeline_config)
        X_prepared = pipeline.fit_transform(self.dataset)

        # Dimensionality Reduction (PCA) if requested
        pca_comp = prep_config.get("pca_components")
        if pca_comp is not None:
            try:
                n_comp = int(pca_comp)
                if 1 <= n_comp < X_prepared.shape[1]:
                    # Fast SVD PCA
                    mean_vec = np.mean(X_prepared, axis=0)
                    X_cent = X_prepared - mean_vec
                    _, _, Vt = np.linalg.svd(X_cent, full_matrices=False)
                    X_prepared = X_cent @ Vt[:n_comp].T
            except Exception:
                pass  # Fallback to full features on error

        if self.cache_preprocessing:
            self._prep_cache[cache_key] = X_prepared.copy()

        return X_prepared

    @staticmethod
    def instantiate_model(
        algorithm: str,
        hyperparameters: Dict[str, Any],
        random_state: int = 42,
    ) -> ClusteringModelBase:
        """
        Factory to instantiate clustering model with typed parameters.
        """
        alg = algorithm.lower()
        hyp = copy.deepcopy(hyperparameters)

        if alg == "kmeans":
            return KMeansModel(
                n_clusters=hyp.get("n_clusters", 4),
                init=hyp.get("init", "k-means++"),
                max_iter=hyp.get("max_iter", 300),
                random_state=random_state,
            )
        elif alg == "kmedoids":
            return KMedoidsModel(
                n_clusters=hyp.get("n_clusters", 4),
                metric=hyp.get("metric", "manhattan"),
                random_state=random_state,
            )
        elif alg == "dbscan":
            return DBSCANModel(
                eps=float(hyp.get("eps", 0.5)),
                min_samples=int(hyp.get("min_samples", 5)),
            )
        elif alg == "hdbscan":
            return HDBSCANModel(
                min_cluster_size=int(hyp.get("min_cluster_size", 15)),
                min_samples=int(hyp.get("min_samples", 5)),
            )
        elif alg == "agglomerative":
            linkage = hyp.get("linkage", "ward")
            metric = "euclidean" if linkage == "ward" else "euclidean"
            return AgglomerativeModel(
                n_clusters=int(hyp.get("n_clusters", 4)),
                linkage=linkage,
                metric=metric,
            )
        elif alg == "gmm":
            n_clusters = int(hyp.get("n_clusters", hyp.get("n_components", 4)))
            return GaussianMixtureModel(
                n_clusters=n_clusters,
                covariance_type=hyp.get("covariance_type", "full"),
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown clustering algorithm: '{alg}'")

    def _compute_stability(
        self,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        X: np.ndarray,
        base_labels: np.ndarray,
    ) -> float:
        """
        Computes subsampling bootstrap cluster stability via Adjusted Rand Index.
        """
        n_samples = len(X)
        if n_samples < 20:
            return 1.0

        rng = np.random.default_rng(self.random_state)
        subsample_size = int(max(10, np.round(n_samples * self.subsample_ratio)))
        ari_list = []

        for b in range(self.stability_bootstraps):
            sub_idx = rng.choice(n_samples, size=subsample_size, replace=False)
            X_sub = X[sub_idx]
            try:
                boot_model = self.instantiate_model(algorithm, hyperparameters, random_state=self.random_state + b + 1)
                boot_labels = boot_model.fit_predict(X_sub)
                base_sub_labels = base_labels[sub_idx]
                ari = adjusted_rand_index(base_sub_labels, boot_labels)
                ari_list.append(max(0.0, ari))
            except Exception:
                ari_list.append(0.0)

        return float(np.mean(ari_list)) if ari_list else 1.0

    def compute_fitness(
        self,
        silhouette: float,
        davies_bouldin: float,
        calinski_harabasz: float,
        stability_ari: float,
        labels: np.ndarray,
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Calculates normalized composite fitness F(theta) and penalty terms:
            S_norm = (S + 1.0) / 2.0
            DB_norm = 1.0 - min(1.0, DB / 5.0)
            CH_norm = min(1.0, ln(1 + max(0, CH)) / ln(1 + 10000))
            ARI_norm = clip(stability_ari, 0.0, 1.0)
            P_noise = 0.5 * max(0.0, noise_frac - 0.10)
            P_imbalance = 0.2 * max(0.0, 1.0 - H / ln(k))
        """
        # Normalization
        s_norm = float(np.clip((silhouette + 1.0) / 2.0, 0.0, 1.0))
        db_norm = float(np.clip(1.0 - min(1.0, max(0.0, davies_bouldin) / 5.0), 0.0, 1.0))
        ch_val = max(0.0, calinski_harabasz)
        ch_norm = float(np.clip(np.log1p(ch_val) / np.log1p(10000.0), 0.0, 1.0))
        ari_norm = float(np.clip(stability_ari, 0.0, 1.0))

        # Noise Penalty
        n_samples = len(labels)
        n_noise = int(np.sum(labels == -1))
        noise_frac = float(n_noise / n_samples) if n_samples > 0 else 0.0
        p_noise = float(0.5 * max(0.0, noise_frac - 0.10))

        # Imbalance Penalty (Entropy over non-noise clusters)
        valid_labels = labels[labels >= 0]
        unique_c, counts = np.unique(valid_labels, return_counts=True)
        k = len(unique_c)

        if k >= 2 and len(valid_labels) > 0:
            probs = counts / len(valid_labels)
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            max_entropy = np.log(k)
            balance_ratio = entropy / max_entropy if max_entropy > 0 else 1.0
            p_imbalance = float(0.2 * max(0.0, 1.0 - balance_ratio))
        else:
            p_imbalance = 0.2

        # Weighted Composite Fitness
        raw_fitness = (
            self.weights.silhouette * s_norm
            + self.weights.davies_bouldin * db_norm
            + self.weights.calinski_harabasz * ch_norm
            + self.weights.stability * ari_norm
            - p_noise
            - p_imbalance
        )

        fitness = float(np.clip(raw_fitness, 0.0, 1.0))

        normalized_metrics = {
            "s_norm": round(s_norm, 4),
            "db_norm": round(db_norm, 4),
            "ch_norm": round(ch_norm, 4),
            "ari_norm": round(ari_norm, 4),
        }
        penalties = {
            "p_noise": round(p_noise, 4),
            "p_imbalance": round(p_imbalance, 4),
            "noise_frac": round(noise_frac, 4),
        }

        return fitness, normalized_metrics, penalties

    def evaluate(self, theta: Dict[str, Any]) -> ObjectiveEvaluation:
        """
        Executes end-to-end evaluation for configuration theta.
        Catches exceptions gracefully and assigns -1.0 fitness upon failure.
        """
        start_time = time.perf_counter()

        prep_config = theta.get("preprocessing", {})
        algorithm = theta.get("algorithm", "kmeans")
        hyperparameters = theta.get("hyperparameters", {})

        try:
            # 1. Feature Preparation
            X_prepared = self.prepare_features(prep_config)
            n_samples = len(X_prepared)

            if n_samples < 5:
                raise ValueError(f"Dataset has insufficient samples ({n_samples}).")

            # 2. Model Instantiation & Fitting
            model = self.instantiate_model(algorithm, hyperparameters, random_state=self.random_state)
            labels = model.fit_predict(X_prepared)

            # Check cluster count validity
            unique_labels = np.unique(labels)
            valid_clusters = unique_labels[unique_labels >= 0]
            k = len(valid_clusters)

            if k < 2:
                # Degenerate: 1 cluster or all noise points
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return ObjectiveEvaluation(
                    fitness=-1.0,
                    silhouette=-1.0,
                    davies_bouldin=99.0,
                    calinski_harabasz=0.0,
                    stability_ari=0.0,
                    n_clusters=k,
                    noise_ratio=float(np.mean(labels == -1)),
                    noise_penalty=0.5,
                    imbalance_penalty=0.2,
                    raw_metrics={"n_clusters": k, "reason": "degenerate_single_cluster_or_all_noise"},
                    normalized_metrics={"s_norm": 0.0, "db_norm": 0.0, "ch_norm": 0.0, "ari_norm": 0.0},
                    penalties={"p_noise": 0.5, "p_imbalance": 0.2},
                    labels=labels,
                    execution_time_ms=elapsed,
                    error=f"Degenerate clustering: found only {k} valid cluster(s).",
                )

            # 3. Compute Validation Metrics
            # Silhouette
            sil_score, _ = compute_silhouette_score(X_prepared, labels, sample_size=min(500, n_samples))
            # Davies-Bouldin
            db_score = compute_davies_bouldin_index(X_prepared, labels)
            # Calinski-Harabasz
            ch_score = compute_calinski_harabasz_score(X_prepared, labels)
            # Stability
            stability_ari = self._compute_stability(algorithm, hyperparameters, X_prepared, labels)

            # 4. Composite Fitness & Penalties
            fitness, norm_metrics, penalties = self.compute_fitness(
                silhouette=sil_score,
                davies_bouldin=db_score,
                calinski_harabasz=ch_score,
                stability_ari=stability_ari,
                labels=labels,
            )

            elapsed = (time.perf_counter() - start_time) * 1000.0
            n_noise = int(np.sum(labels == -1))
            noise_ratio = float(n_noise / n_samples)

            return ObjectiveEvaluation(
                fitness=fitness,
                silhouette=sil_score,
                davies_bouldin=db_score,
                calinski_harabasz=ch_score,
                stability_ari=stability_ari,
                n_clusters=k,
                noise_ratio=noise_ratio,
                noise_penalty=penalties["p_noise"],
                imbalance_penalty=penalties["p_imbalance"],
                raw_metrics={
                    "silhouette": sil_score,
                    "davies_bouldin": db_score,
                    "calinski_harabasz": ch_score,
                    "stability_ari": stability_ari,
                    "n_clusters": k,
                    "n_noise": n_noise,
                },
                normalized_metrics=norm_metrics,
                penalties=penalties,
                labels=labels,
                execution_time_ms=elapsed,
                error=None,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ObjectiveEvaluation(
                fitness=-1.0,
                silhouette=-1.0,
                davies_bouldin=99.0,
                calinski_harabasz=0.0,
                stability_ari=0.0,
                n_clusters=0,
                noise_ratio=1.0,
                noise_penalty=0.5,
                imbalance_penalty=0.2,
                raw_metrics={"error": str(e)},
                normalized_metrics={"s_norm": 0.0, "db_norm": 0.0, "ch_norm": 0.0, "ari_norm": 0.0},
                penalties={"p_noise": 0.5, "p_imbalance": 0.2},
                labels=None,
                execution_time_ms=elapsed,
                error=str(e),
            )

    def __call__(self, theta: Dict[str, Any]) -> float:
        """Shorthand callable returning scalar fitness."""
        eval_res = self.evaluate(theta)
        return eval_res.fitness
