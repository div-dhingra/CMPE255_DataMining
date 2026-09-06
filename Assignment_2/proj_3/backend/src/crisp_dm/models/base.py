"""
CRISP-DM Phase 4: Base Clustering Model Interface
Defines the standard abstract base class ClusteringModelBase that all clustering models must adhere to.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class ClusteringModelBase(ABC):
    """
    Abstract base class for all clustering models across all 4 paradigms:
    Partitioning, Density-based, Hierarchical, and Probabilistic.
    """

    def __init__(self, model_name: str = "base_model"):
        self.model_name = model_name
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: int = 0
        self.centroids_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.is_fitted_: bool = False

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the model on training data X and returns cluster assignments.
        Returns:
            np.ndarray of shape (n_samples,) with cluster labels (integers, -1 for noise).
        """
        pass

    def fit(self, X: np.ndarray) -> "ClusteringModelBase":
        """Convenience method fitting model and returning self."""
        self.fit_predict(X)
        return self

    @abstractmethod
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Assigns new incoming feature vectors X_new to nearest cluster.
        Returns:
            np.ndarray of shape (n_samples,) with cluster labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        """
        Returns soft assignment / membership probabilities for each cluster.
        Returns:
            np.ndarray of shape (n_samples, n_clusters) with normalized probabilities summing to 1.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Returns dictionary of model hyperparameters.
        """
        pass
