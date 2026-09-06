"""
Autoresearch Parameter Search Space Module.
Defines the formal configuration space Theta over:
- Preprocessing transformations (imputer, outlier handler, scaler, PCA components, feature engineering)
- 6 clustering algorithms across 4 paradigms (KMeans, KMedoids, DBSCAN, HDBSCAN, Agglomerative, GMM)
- Continuous and discrete hyperparameter bounds
- Stochastic neighborhood mutation operator mutate(theta) and random sample generator
- Deterministic configuration hashing and validation
"""

import copy
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


# Default boundaries and discrete sets
PREPROCESSING_SPACE = {
    "imputer": ["median", "mean", "knn", "mice"],
    "outlier_handler": ["winsorize", "iqr", "isolation_forest", "none"],
    "scaler": ["yeo_johnson", "standard", "robust", "minmax"],
    "pca_components": [None, 2, 3, 5, 10],
    "feature_engineering": [True, False],
}

ALGORITHM_CHOICES = ["kmeans", "kmedoids", "dbscan", "hdbscan", "agglomerative", "gmm"]

ALGORITHM_HYPERPARAMETERS = {
    "kmeans": {
        "n_clusters": {"type": "int", "low": 2, "high": 12, "default": 4, "step": 1},
        "init": {"type": "choice", "choices": ["k-means++", "random"], "default": "k-means++"},
        "max_iter": {"type": "int", "low": 100, "high": 500, "default": 300, "step": 50},
    },
    "kmedoids": {
        "n_clusters": {"type": "int", "low": 2, "high": 12, "default": 4, "step": 1},
        "metric": {"type": "choice", "choices": ["euclidean", "manhattan"], "default": "manhattan"},
    },
    "dbscan": {
        "eps": {"type": "float", "low": 0.1, "high": 2.5, "default": 0.5, "step": 0.1, "drift_std": 0.2},
        "min_samples": {"type": "int", "low": 3, "high": 20, "default": 5, "step": 1},
    },
    "hdbscan": {
        "min_cluster_size": {"type": "int", "low": 3, "high": 30, "default": 15, "step": 2},
        "min_samples": {"type": "int", "low": 2, "high": 15, "default": 5, "step": 1},
    },
    "agglomerative": {
        "n_clusters": {"type": "int", "low": 2, "high": 12, "default": 4, "step": 1},
        "linkage": {"type": "choice", "choices": ["ward", "complete", "average"], "default": "ward"},
    },
    "gmm": {
        "n_clusters": {"type": "int", "low": 2, "high": 12, "default": 4, "step": 1},
        "covariance_type": {"type": "choice", "choices": ["full", "tied", "diag", "spherical"], "default": "full"},
    },
}


class SearchSpace:
    """
    Formal representation of the clustering pipeline hyperparameter search space Theta.
    Encodes preprocessing choices, model selection, and algorithmic hyperparameter bounds.
    Provides random sampling, neighbor generation, and mutation operators.
    """

    def __init__(
        self,
        preprocessing_space: Optional[Dict[str, List[Any]]] = None,
        algorithm_hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
        algorithms: Optional[List[str]] = None,
    ):
        self.preprocessing_space = preprocessing_space or copy.deepcopy(PREPROCESSING_SPACE)
        self.algorithm_hyperparameters = algorithm_hyperparameters or copy.deepcopy(ALGORITHM_HYPERPARAMETERS)
        self.algorithms = algorithms or copy.deepcopy(ALGORITHM_CHOICES)

    def get_default_configuration(self, algorithm: str = "kmeans") -> Dict[str, Any]:
        """
        Returns the standard default baseline configuration for a given algorithm.
        """
        alg = algorithm.lower()
        if alg not in self.algorithms:
            raise ValueError(f"Algorithm '{alg}' not in search space: {self.algorithms}")

        hyperparams = {}
        for param, spec in self.algorithm_hyperparameters[alg].items():
            hyperparams[param] = spec["default"]

        return {
            "preprocessing": {
                "imputer": "median",
                "outlier_handler": "winsorize",
                "scaler": "yeo_johnson",
                "pca_components": None,
                "feature_engineering": True,
            },
            "algorithm": alg,
            "hyperparameters": hyperparams,
        }

    def sample_random_configuration(
        self,
        algorithm: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Uniformly samples a valid configuration theta from Theta.
        """
        rng = np.random.default_rng(random_state)

        # 1. Sample Preprocessing parameters
        prep_config = {}
        for param, choices in self.preprocessing_space.items():
            idx = int(rng.integers(0, len(choices)))
            prep_config[param] = choices[idx]

        # 2. Select Algorithm
        if algorithm is not None:
            alg = algorithm.lower()
            if alg not in self.algorithms:
                raise ValueError(f"Algorithm '{alg}' is not in search space algorithms.")
        else:
            alg = str(rng.choice(self.algorithms))

        # 3. Sample Hyperparameters for chosen Algorithm
        hyperparams = {}
        alg_spec = self.algorithm_hyperparameters[alg]

        for param, spec in alg_spec.items():
            p_type = spec["type"]
            if p_type == "choice":
                choices = spec["choices"]
                hyperparams[param] = choices[int(rng.integers(0, len(choices)))]
            elif p_type == "int":
                low = spec["low"]
                high = spec["high"]
                hyperparams[param] = int(rng.integers(low, high + 1))
            elif p_type == "float":
                low = spec["low"]
                high = spec["high"]
                val = float(rng.uniform(low, high))
                hyperparams[param] = round(val, 3)

        return self.validate_configuration({
            "preprocessing": prep_config,
            "algorithm": alg,
            "hyperparameters": hyperparams,
        })

    def mutate(
        self,
        theta: Dict[str, Any],
        mutation_rate: float = 0.3,
        allow_algorithm_mutation: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Applies stochastic perturbation / neighborhood mutation to configuration theta.
        Perturbs discrete choices or continuous/integer parameters with probability.
        Guarantees at least one parameter is mutated to produce a valid distinct neighbor theta'.
        """
        rng = np.random.default_rng(random_state)
        new_theta = copy.deepcopy(theta)
        prep = new_theta.setdefault("preprocessing", {})
        alg = new_theta.get("algorithm", "kmeans").lower()
        hyp = new_theta.setdefault("hyperparameters", {})

        # List all mutable targets: ('prep', param_name) or ('hyp', param_name) or ('alg', None)
        targets = []
        for p in self.preprocessing_space:
            targets.append(("prep", p))

        alg_spec = self.algorithm_hyperparameters.get(alg, {})
        for p in alg_spec:
            targets.append(("hyp", p))

        if allow_algorithm_mutation and len(self.algorithms) > 1:
            targets.append(("alg", "algorithm"))

        # Guarantee at least 1 mutation
        rng.shuffle(targets)
        mutated = False

        for target_type, param in targets:
            if not mutated or rng.random() < mutation_rate:
                if target_type == "alg":
                    other_algs = [a for a in self.algorithms if a != alg]
                    if other_algs:
                        new_alg = str(rng.choice(other_algs))
                        new_theta["algorithm"] = new_alg
                        # Resample or initialize default hyperparameters for new algorithm
                        new_hyp = {}
                        for hp_name, hp_spec in self.algorithm_hyperparameters[new_alg].items():
                            if hp_spec["type"] == "choice":
                                new_hyp[hp_name] = str(rng.choice(hp_spec["choices"]))
                            elif hp_spec["type"] == "int":
                                new_hyp[hp_name] = int(rng.integers(hp_spec["low"], hp_spec["high"] + 1))
                            elif hp_spec["type"] == "float":
                                new_hyp[hp_name] = round(float(rng.uniform(hp_spec["low"], hp_spec["high"])), 3)
                        new_theta["hyperparameters"] = new_hyp
                        mutated = True
                        break  # Algorithm switch is a major jump

                elif target_type == "prep":
                    choices = self.preprocessing_space[param]
                    curr_val = prep.get(param)
                    other_choices = [c for c in choices if c != curr_val]
                    if other_choices:
                        prep[param] = other_choices[int(rng.integers(0, len(other_choices)))]
                        mutated = True

                elif target_type == "hyp":
                    if param not in alg_spec:
                        continue
                    spec = alg_spec[param]
                    curr_val = hyp.get(param, spec.get("default"))
                    p_type = spec["type"]

                    if p_type == "choice":
                        choices = spec["choices"]
                        other_choices = [c for c in choices if c != curr_val]
                        if other_choices:
                            hyp[param] = other_choices[int(rng.integers(0, len(other_choices)))]
                            mutated = True
                    elif p_type == "int":
                        low = spec["low"]
                        high = spec["high"]
                        step = spec.get("step", 1)
                        # Drift by +/- step or +/- 2*step
                        delta = int(rng.choice([-2 * step, -step, step, 2 * step]))
                        new_val = int(np.clip(curr_val + delta, low, high))
                        if new_val == curr_val:
                            new_val = int(low if curr_val >= high else high)
                        hyp[param] = new_val
                        mutated = True
                    elif p_type == "float":
                        low = spec["low"]
                        high = spec["high"]
                        drift_std = spec.get("drift_std", 0.2)
                        drift = float(rng.normal(0.0, drift_std))
                        new_val = float(np.clip(curr_val + drift, low, high))
                        if abs(new_val - curr_val) < 1e-4:
                            new_val = low if curr_val >= (low + high) / 2 else high
                        hyp[param] = round(new_val, 3)
                        mutated = True

        return self.validate_configuration(new_theta)

    def get_neighbors(
        self,
        theta: Dict[str, Any],
        n_neighbors: int = 5,
        allow_algorithm_mutation: bool = False,
        random_state: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates a list of distinct neighboring configurations around theta.
        """
        rng = np.random.default_rng(random_state)
        neighbors: List[Dict[str, Any]] = []
        seen_hashes = {self.config_hash(theta)}

        for _ in range(n_neighbors * 4):
            seed = int(rng.integers(0, 1_000_000))
            cand = self.mutate(
                theta,
                mutation_rate=0.4,
                allow_algorithm_mutation=allow_algorithm_mutation,
                random_state=seed,
            )
            c_hash = self.config_hash(cand)
            if c_hash not in seen_hashes:
                seen_hashes.add(c_hash)
                neighbors.append(cand)
                if len(neighbors) >= n_neighbors:
                    break

        # If not enough distinct neighbors found through mutation, fallback to random sampling
        while len(neighbors) < n_neighbors:
            rand_seed = int(rng.integers(0, 1_000_000))
            cand = self.sample_random_configuration(
                algorithm=theta.get("algorithm") if not allow_algorithm_mutation else None,
                random_state=rand_seed,
            )
            c_hash = self.config_hash(cand)
            if c_hash not in seen_hashes:
                seen_hashes.add(c_hash)
                neighbors.append(cand)

        return neighbors

    def validate_configuration(self, theta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and coerces types, ensuring theta conforms strictly to valid ranges and bounds.
        """
        validated = copy.deepcopy(theta)
        prep = validated.setdefault("preprocessing", {})
        alg = validated.get("algorithm", "kmeans").lower()
        validated["algorithm"] = alg

        # Validate Preprocessing
        if prep.get("imputer") not in self.preprocessing_space["imputer"]:
            prep["imputer"] = "median"
        if prep.get("outlier_handler") not in self.preprocessing_space["outlier_handler"]:
            prep["outlier_handler"] = "winsorize"
        if prep.get("scaler") not in self.preprocessing_space["scaler"]:
            prep["scaler"] = "yeo_johnson"
        if prep.get("pca_components") not in self.preprocessing_space["pca_components"]:
            prep["pca_components"] = None
        prep["feature_engineering"] = bool(prep.get("feature_engineering", True))

        # Validate Algorithm & Hyperparameters
        if alg not in self.algorithm_hyperparameters:
            alg = "kmeans"
            validated["algorithm"] = alg

        alg_spec = self.algorithm_hyperparameters[alg]
        hyp = validated.setdefault("hyperparameters", {})

        # Ensure all required hyperparameters exist and are within bounds
        for param, spec in alg_spec.items():
            p_type = spec["type"]
            val = hyp.get(param, spec["default"])

            if p_type == "choice":
                if val not in spec["choices"]:
                    hyp[param] = spec["default"]
                else:
                    hyp[param] = val
            elif p_type == "int":
                try:
                    int_val = int(val)
                    hyp[param] = int(np.clip(int_val, spec["low"], spec["high"]))
                except (ValueError, TypeError):
                    hyp[param] = spec["default"]
            elif p_type == "float":
                try:
                    flt_val = float(val)
                    hyp[param] = round(float(np.clip(flt_val, spec["low"], spec["high"])), 3)
                except (ValueError, TypeError):
                    hyp[param] = spec["default"]

        return validated

    @staticmethod
    def config_hash(theta: Dict[str, Any]) -> str:
        """
        Computes a deterministic MD5 hash string for a configuration dictionary.
        Used for tabu memory and duplicate candidate detection.
        """
        # Canonical sort keys serialization
        def _canonicalize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _canonicalize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, (list, tuple)):
                return [_canonicalize(v) for v in obj]
            elif isinstance(obj, float):
                return round(obj, 4)
            return obj

        clean_dict = _canonicalize(theta)
        json_str = json.dumps(clean_dict, sort_keys=True)
        return hashlib.md5(json_str.encode("utf-8")).hexdigest()

    @staticmethod
    def flatten_config(theta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flattens nested config dict into a single level key-value dictionary for tables/CSV.
        """
        flat = {}
        prep = theta.get("preprocessing", {})
        for k, v in prep.items():
            flat[f"prep_{k}"] = v

        flat["algorithm"] = theta.get("algorithm", "kmeans")

        hyp = theta.get("hyperparameters", {})
        for k, v in hyp.items():
            flat[f"hyp_{k}"] = v

        return flat

    @staticmethod
    def unflatten_config(flat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstructs nested configuration structure from a flattened dictionary.
        """
        prep = {}
        hyp = {}
        algorithm = flat.get("algorithm", "kmeans")

        for k, v in flat.items():
            if k.startswith("prep_"):
                prep[k.replace("prep_", "")] = v
            elif k.startswith("hyp_"):
                hyp[k.replace("hyp_", "")] = v

        return {
            "preprocessing": prep,
            "algorithm": algorithm,
            "hyperparameters": hyp,
        }
