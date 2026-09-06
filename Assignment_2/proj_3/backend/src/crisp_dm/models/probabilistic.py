"""
CRISP-DM Phase 4: Probabilistic Clustering Models
Implements Gaussian Mixture Models (GMM) with Expectation-Maximization (EM),
supporting Full, Tied, Diagonal, and Spherical covariance structures, along with BIC and AIC model selection.
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from .base import ClusteringModelBase


class GaussianMixtureModel(ClusteringModelBase):
    """
    Gaussian Mixture Model (GMM) trained via Expectation-Maximization.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        covariance_type: str = "full",
        n_init: int = 5,
        max_iter: int = 200,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        random_state: int = 42,
    ):
        super().__init__(model_name="gmm")
        self.n_clusters = int(n_clusters)
        self.covariance_type = covariance_type.lower()
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[Any] = None
        self.log_likelihood_: Optional[float] = None
        self.bic_: Optional[float] = None
        self.aic_: Optional[float] = None

    def _estimate_log_gaussian_prob(self, X: np.ndarray, means: np.ndarray, covars: Any) -> np.ndarray:
        """
        Computes log N(X | means[k], covars[k]) for all k.
        Returns array of shape (n_samples, n_clusters).
        """
        n_samples, n_features = X.shape
        k = self.n_clusters
        log_prob = np.empty((n_samples, k))

        for c in range(k):
            mu = means[c]
            diff = X - mu

            if self.covariance_type == "full":
                cov = covars[c] + self.reg_covar * np.eye(n_features)
                sign, log_det = np.linalg.slogdet(cov)
                if sign <= 0:
                    cov += 1e-4 * np.eye(n_features)
                    _, log_det = np.linalg.slogdet(cov)
                inv_cov = np.linalg.pinv(cov)
                # Mahalanobis distance squared: sum_j diff_ij * (diff @ inv_cov)_ij
                maha_sq = np.sum((diff @ inv_cov) * diff, axis=1)
                log_prob[:, c] = -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + maha_sq)

            elif self.covariance_type == "diag":
                cov_diag = covars[c] + self.reg_covar
                log_det = np.sum(np.log(cov_diag))
                maha_sq = np.sum((diff ** 2) / cov_diag, axis=1)
                log_prob[:, c] = -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + maha_sq)

            elif self.covariance_type == "tied":
                cov = covars + self.reg_covar * np.eye(n_features)
                sign, log_det = np.linalg.slogdet(cov)
                if sign <= 0:
                    cov += 1e-4 * np.eye(n_features)
                    _, log_det = np.linalg.slogdet(cov)
                inv_cov = np.linalg.pinv(cov)
                maha_sq = np.sum((diff @ inv_cov) * diff, axis=1)
                log_prob[:, c] = -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + maha_sq)

            elif self.covariance_type == "spherical":
                var = float(covars[c]) + self.reg_covar
                log_det = n_features * np.log(var)
                maha_sq = np.sum(diff ** 2, axis=1) / var
                log_prob[:, c] = -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + maha_sq)

        return log_prob

    def _e_step(self, X: np.ndarray, weights: np.ndarray, means: np.ndarray, covars: Any) -> tuple[np.ndarray, float]:
        """E-Step: returns posterior responsibilities and total log likelihood."""
        log_prob = self._estimate_log_gaussian_prob(X, means, covars)
        log_weights = np.log(weights + 1e-15)
        weighted_log_prob = log_prob + log_weights
        log_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_norm[:, None]
        responsibilities = np.exp(log_resp)
        total_log_likelihood = float(np.sum(log_norm))
        return responsibilities, total_log_likelihood

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
        """M-Step: updates weights, means, and covariances."""
        n_samples, n_features = X.shape
        k = self.n_clusters

        # Nk: effective number of points assigned to cluster k
        Nk = np.sum(resp, axis=0) + 10.0 * np.finfo(float).eps
        weights = Nk / n_samples
        means = (resp.T @ X) / Nk[:, None]

        if self.covariance_type == "full":
            covars = np.empty((k, n_features, n_features))
            for c in range(k):
                diff = X - means[c]
                covars[c] = (diff.T @ (diff * resp[:, c:c+1])) / Nk[c] + self.reg_covar * np.eye(n_features)

        elif self.covariance_type == "diag":
            covars = np.empty((k, n_features))
            for c in range(k):
                diff = X - means[c]
                covars[c] = np.sum(resp[:, c:c+1] * (diff ** 2), axis=0) / Nk[c] + self.reg_covar

        elif self.covariance_type == "tied":
            covars = np.zeros((n_features, n_features))
            for c in range(k):
                diff = X - means[c]
                covars += diff.T @ (diff * resp[:, c:c+1])
            covars = covars / n_samples + self.reg_covar * np.eye(n_features)

        elif self.covariance_type == "spherical":
            covars = np.empty(k)
            for c in range(k):
                diff = X - means[c]
                covars[c] = np.sum(resp[:, c] * np.sum(diff ** 2, axis=1)) / (n_features * Nk[c]) + self.reg_covar

        return weights, means, covars

    def _fit_single_run(self, X: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, Any, np.ndarray, float]:
        n_samples, n_features = X.shape
        k = self.n_clusters

        # Initialize means via k-means++
        first_idx = rng.integers(0, n_samples)
        means = np.empty((k, n_features))
        means[0] = X[first_idx]
        closest_dist_sq = np.sum((X - means[0]) ** 2, axis=1)

        for c_idx in range(1, k):
            tot = np.sum(closest_dist_sq)
            probs = closest_dist_sq / tot if tot > 0 else np.ones(n_samples) / n_samples
            means[c_idx] = X[rng.choice(n_samples, p=probs)]
            closest_dist_sq = np.minimum(closest_dist_sq, np.sum((X - means[c_idx]) ** 2, axis=1))

        weights = np.full(k, 1.0 / k)

        # Initial covariance
        empirical_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
        if self.covariance_type == "full":
            covars = np.array([empirical_cov.copy() for _ in range(k)])
        elif self.covariance_type == "diag":
            covars = np.array([np.diag(empirical_cov).copy() for _ in range(k)])
        elif self.covariance_type == "tied":
            covars = empirical_cov.copy()
        elif self.covariance_type == "spherical":
            covars = np.full(k, np.mean(np.diag(empirical_cov)))

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            resp, ll = self._e_step(X, weights, means, covars)
            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
            weights, means, covars = self._m_step(X, resp)

        resp, final_ll = self._e_step(X, weights, means, covars)
        return weights, means, covars, resp, final_ll

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} must be >= n_clusters={self.n_clusters}")

        rng = np.random.default_rng(self.random_state)
        best_ll = -np.inf
        best_weights = None
        best_means = None
        best_covars = None
        best_resp = None

        for _ in range(self.n_init):
            weights, means, covars, resp, ll = self._fit_single_run(X, rng)
            if ll > best_ll:
                best_ll = ll
                best_weights = weights
                best_means = means
                best_covars = covars
                best_resp = resp

        self.weights_ = best_weights
        self.means_ = best_means
        self.centroids_ = best_means
        self.covariances_ = best_covars
        self.log_likelihood_ = best_ll
        self.labels_ = np.argmax(best_resp, axis=1)
        self.n_clusters_ = self.n_clusters

        # Compute free parameters for BIC / AIC
        k = self.n_clusters
        d = n_features
        n_means = k * d
        n_weights = k - 1
        if self.covariance_type == "full":
            n_cov = k * d * (d + 1) // 2
        elif self.covariance_type == "diag":
            n_cov = k * d
        elif self.covariance_type == "tied":
            n_cov = d * (d + 1) // 2
        else:  # spherical
            n_cov = k

        p = n_means + n_weights + n_cov
        self.bic_ = float(-2.0 * best_ll + p * np.log(n_samples))
        self.aic_ = float(-2.0 * best_ll + 2.0 * p)

        self.is_fitted_ = True
        return self.labels_

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X_new), axis=1)

    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.weights_ is None or self.means_ is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_new = np.asarray(X_new, dtype=float)
        resp, _ = self._e_step(X_new, self.weights_, self.means_, self.covariances_)
        return resp

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "covariance_type": self.covariance_type,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "random_state": self.random_state,
        }
