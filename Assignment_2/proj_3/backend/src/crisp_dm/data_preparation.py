"""
CRISP-DM Phase 3: Data Preparation Module
Provides automated preprocessing pipelines:
- Missing Value Imputation: Median, Mean, KNN, and MICE (Iterative Ridge)
- Outlier Handling: Quantile Winsorization, IQR Clipping, and Isolation Forest
- Scaling & Power Transforms: Yeo-Johnson PowerTransformer, RobustScaler, StandardScaler, MinMaxScaler
- Domain Feature Engineering: Behavioral and financial ratios
- Complete DataPreparationPipeline with fit/transform lifecycle
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.spatial.distance import cdist


@dataclass
class PipelineConfig:
    """Configuration dataclass for DataPreparationPipeline."""
    imputer_strategy: str = "median"  # "median", "mean", "knn", "mice", "none"
    knn_neighbors: int = 5
    mice_max_iter: int = 10
    outlier_strategy: str = "winsorize"  # "winsorize", "iqr", "isolation_forest", "none"
    winsorize_limits: Tuple[float, float] = (0.01, 0.01)
    iqr_multiplier: float = 1.5
    isolation_forest_contamination: float = 0.02
    isolation_forest_n_trees: int = 50
    scaling_strategy: str = "yeo_johnson"  # "yeo_johnson", "standard", "robust", "minmax", "none"
    engineer_ratios: bool = True
    id_column: str = "CUST_ID"
    feature_columns: Optional[List[str]] = None


# --------------------------------------------------------------------------
# 1. Imputation Transformers
# --------------------------------------------------------------------------

class Imputer:
    """
    Imputes missing values using Median, Mean, KNN, or MICE (Multivariate Chained Equations).
    """

    def __init__(self, strategy: str = "median", knn_neighbors: int = 5, mice_max_iter: int = 10):
        self.strategy = strategy.lower()
        self.knn_neighbors = knn_neighbors
        self.mice_max_iter = mice_max_iter
        self.statistics_: Optional[np.ndarray] = None
        self.train_data_: Optional[np.ndarray] = None
        self.mice_models_: Optional[List[Any]] = None

    def fit(self, X: np.ndarray) -> "Imputer":
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
            # If an entire column is NaN, fill with 0.0
            self.statistics_ = np.nan_to_num(self.statistics_, nan=0.0)
        elif self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
            self.statistics_ = np.nan_to_num(self.statistics_, nan=0.0)
        elif self.strategy == "knn":
            # Store column medians as base fallback and non-NaN copy for neighbor distance
            self.statistics_ = np.nan_to_num(np.nanmedian(X, axis=0), nan=0.0)
            # Store clean training reference
            X_clean = X.copy()
            for j in range(n_features):
                mask = np.isnan(X_clean[:, j])
                X_clean[mask, j] = self.statistics_[j]
            self.train_data_ = X_clean
        elif self.strategy in ("mice", "iterative"):
            # Chained ridge regression imputation
            self.statistics_ = np.nan_to_num(np.nanmedian(X, axis=0), nan=0.0)
            X_filled = X.copy()
            for j in range(n_features):
                mask = np.isnan(X_filled[:, j])
                X_filled[mask, j] = self.statistics_[j]

            # Fit chained linear ridge models for columns that have missing values
            self.mice_models_ = []
            for j in range(n_features):
                if np.isnan(X[:, j]).any():
                    observed_idx = ~np.isnan(X[:, j])
                    if observed_idx.sum() > 5:
                        other_cols = [c for c in range(n_features) if c != j]
                        X_pred = X_filled[observed_idx][:, other_cols]
                        y_pred = X[observed_idx, j]
                        # Ridge regression: (X^T X + alpha*I)^-1 X^T y
                        X_pred_bias = np.hstack([np.ones((len(X_pred), 1)), X_pred])
                        alpha = 1.0
                        reg = alpha * np.eye(X_pred_bias.shape[1])
                        reg[0, 0] = 0.0
                        weights = np.linalg.pinv(X_pred_bias.T @ X_pred_bias + reg) @ X_pred_bias.T @ y_pred
                        self.mice_models_.append((j, other_cols, weights))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = np.asarray(X, dtype=float).copy()
        n_samples, n_features = X_out.shape

        if self.strategy in ("median", "mean", "none"):
            if self.strategy != "none" and self.statistics_ is not None:
                for j in range(n_features):
                    mask = np.isnan(X_out[:, j])
                    if mask.any():
                        X_out[mask, j] = self.statistics_[j]
            return X_out

        if self.strategy == "knn":
            if self.train_data_ is None or self.statistics_ is None:
                return self.fit(X_out).transform(X_out)

            for i in range(n_samples):
                nan_mask = np.isnan(X_out[i])
                if nan_mask.any():
                    valid_mask = ~nan_mask
                    if valid_mask.any():
                        # Compute distance to train points on valid features
                        diff = self.train_data_[:, valid_mask] - X_out[i, valid_mask]
                        dist = np.linalg.norm(diff, axis=1)
                        k = min(self.knn_neighbors, len(dist))
                        nn_idx = np.argpartition(dist, k)[:k]
                        weights = 1.0 / (dist[nn_idx] + 1e-6)
                        weights /= np.sum(weights)

                        for j in np.where(nan_mask)[0]:
                            imputed_val = float(np.sum(weights * self.train_data_[nn_idx, j]))
                            X_out[i, j] = imputed_val
                    else:
                        X_out[i] = self.statistics_
            return X_out

        if self.strategy in ("mice", "iterative"):
            # Initial median fill
            for j in range(n_features):
                mask = np.isnan(X_out[:, j])
                if mask.any() and self.statistics_ is not None:
                    X_out[mask, j] = self.statistics_[j]

            # Refine via fitted MICE models
            if self.mice_models_:
                for _ in range(self.mice_max_iter):
                    for j, other_cols, weights in self.mice_models_:
                        X_pred_bias = np.hstack([np.ones((n_samples, 1)), X_out[:, other_cols]])
                        y_imputed = X_pred_bias @ weights
                        # Apply to original NaNs
                        nan_orig = np.isnan(X[:, j])
                        X_out[nan_orig, j] = y_imputed[nan_orig]
            return X_out

        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------
# 2. Outlier Detection & Treatment Transformers
# --------------------------------------------------------------------------

class OutlierHandler:
    """
    Handles extreme values via Winsorization, IQR thresholding, or Isolation Forest.
    """

    def __init__(
        self,
        strategy: str = "winsorize",
        winsorize_limits: Tuple[float, float] = (0.01, 0.01),
        iqr_multiplier: float = 1.5,
        contamination: float = 0.02,
        n_trees: int = 50,
        random_state: int = 42,
    ):
        self.strategy = strategy.lower()
        self.winsorize_limits = winsorize_limits
        self.iqr_multiplier = iqr_multiplier
        self.contamination = contamination
        self.n_trees = n_trees
        self.random_state = random_state
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None
        self.isolation_forest_trees_: Optional[List[Dict[str, Any]]] = None
        self.anomaly_threshold_: Optional[float] = None

    def fit(self, X: np.ndarray) -> "OutlierHandler":
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if self.strategy == "winsorize":
            low_pct = self.winsorize_limits[0] * 100.0
            high_pct = (1.0 - self.winsorize_limits[1]) * 100.0
            self.lower_bounds_ = np.nanpercentile(X, low_pct, axis=0)
            self.upper_bounds_ = np.nanpercentile(X, high_pct, axis=0)

        elif self.strategy == "iqr":
            q25 = np.nanpercentile(X, 25.0, axis=0)
            q75 = np.nanpercentile(X, 75.0, axis=0)
            iqr = q75 - q25
            self.lower_bounds_ = q25 - self.iqr_multiplier * iqr
            self.upper_bounds_ = q75 + self.iqr_multiplier * iqr

        elif self.strategy == "isolation_forest":
            # Genuine Isolation Forest implementation
            rng = np.random.default_rng(self.random_state)
            trees = []
            max_depth = int(np.ceil(np.log2(max(n_samples, 2))))
            subsample_size = min(256, n_samples)

            for _ in range(self.n_trees):
                sub_idx = rng.choice(n_samples, size=subsample_size, replace=False)
                sub_X = X[sub_idx]
                tree = self._build_i_tree(sub_X, 0, max_depth, rng)
                trees.append(tree)

            self.isolation_forest_trees_ = trees
            # Compute path lengths on training data to set contamination threshold
            scores = self._compute_anomaly_scores(X)
            self.anomaly_threshold_ = float(np.percentile(scores, (1.0 - self.contamination) * 100.0))

            # Store percentile bounds as fallback for transformation
            self.lower_bounds_ = np.nanpercentile(X, 1.0, axis=0)
            self.upper_bounds_ = np.nanpercentile(X, 99.0, axis=0)

        return self

    def _build_i_tree(self, X: np.ndarray, current_depth: int, max_depth: int, rng: np.random.Generator) -> Dict[str, Any]:
        n_samples, n_features = X.shape
        if current_depth >= max_depth or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}

        # Pick random feature
        feature_idx = int(rng.integers(0, n_features))
        f_min = float(np.min(X[:, feature_idx]))
        f_max = float(np.max(X[:, feature_idx]))

        if f_min == f_max:
            return {"type": "leaf", "size": n_samples}

        split_val = float(rng.uniform(f_min, f_max))
        left_mask = X[:, feature_idx] < split_val
        right_mask = ~left_mask

        return {
            "type": "split",
            "feature": feature_idx,
            "value": split_val,
            "left": self._build_i_tree(X[left_mask], current_depth + 1, max_depth, rng),
            "right": self._build_i_tree(X[right_mask], current_depth + 1, max_depth, rng),
        }

    def _path_length(self, x: np.ndarray, node: Dict[str, Any], current_depth: int) -> float:
        if node["type"] == "leaf":
            size = node["size"]
            if size <= 1:
                return float(current_depth)
            # Average path length of unsuccessful search in BST
            c_factor = 2.0 * (np.log(size - 1) + 0.5772156649) - (2.0 * (size - 1) / size)
            return current_depth + c_factor

        feat = node["feature"]
        val = node["value"]
        if x[feat] < val:
            return self._path_length(x, node["left"], current_depth + 1)
        else:
            return self._path_length(x, node["right"], current_depth + 1)

    def _compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        if self.isolation_forest_trees_ is None:
            return np.zeros(len(X))
        n_samples = len(X)
        path_lengths = np.zeros((n_samples, len(self.isolation_forest_trees_)))
        for t_idx, tree in enumerate(self.isolation_forest_trees_):
            for i in range(n_samples):
                path_lengths[i, t_idx] = self._path_length(X[i], tree, 0)
        mean_paths = np.mean(path_lengths, axis=1)
        c_n = 2.0 * (np.log(max(256, 2) - 1) + 0.5772156649) - (2.0 * 255.0 / 256.0)
        scores = 2.0 ** (-mean_paths / c_n)
        return scores

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Crops extreme values to fitted lower and upper bounds."""
        X_out = np.asarray(X, dtype=float).copy()
        if self.strategy == "none" or self.lower_bounds_ is None or self.upper_bounds_ is None:
            return X_out

        for j in range(X_out.shape[1]):
            X_out[:, j] = np.clip(X_out[:, j], self.lower_bounds_[j], self.upper_bounds_[j])
        return X_out

    def filter(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (filtered_X, inlier_mask) where outlier points are removed.
        """
        X_arr = np.asarray(X, dtype=float)
        if self.strategy == "none":
            return X_arr, np.ones(len(X_arr), dtype=bool)

        if self.strategy == "isolation_forest" and self.anomaly_threshold_ is not None:
            scores = self._compute_anomaly_scores(X_arr)
            inlier_mask = scores <= self.anomaly_threshold_
            return X_arr[inlier_mask], inlier_mask

        # For winsorize or iqr bounds: check if any feature is outside bounds
        if self.lower_bounds_ is not None and self.upper_bounds_ is not None:
            inlier_mask = np.all((X_arr >= self.lower_bounds_) & (X_arr <= self.upper_bounds_), axis=1)
            return X_arr[inlier_mask], inlier_mask

        return X_arr, np.ones(len(X_arr), dtype=bool)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------
# 3. Scaling & Power Transformation Transformers
# --------------------------------------------------------------------------

class FeatureScaler:
    """
    Transforms and standardizes features via Yeo-Johnson PowerTransform, StandardScaler,
    RobustScaler, or MinMaxScaler.
    """

    def __init__(self, strategy: str = "yeo_johnson"):
        self.strategy = strategy.lower()
        self.lambdas_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.scales_: Optional[np.ndarray] = None
        self.medians_: Optional[np.ndarray] = None
        self.iqrs_: Optional[np.ndarray] = None
        self.mins_: Optional[np.ndarray] = None
        self.maxs_: Optional[np.ndarray] = None

    def _yeo_johnson_transform_col(self, y: np.ndarray, lmbda: float) -> np.ndarray:
        """Applies Yeo-Johnson transform for a single 1D array and scalar lambda."""
        out = np.zeros_like(y, dtype=float)
        pos_mask = y >= 0
        neg_mask = ~pos_mask

        # Case 1: y >= 0
        if np.isclose(lmbda, 0.0):
            out[pos_mask] = np.log1p(y[pos_mask])
        else:
            out[pos_mask] = ((y[pos_mask] + 1.0) ** lmbda - 1.0) / lmbda

        # Case 2: y < 0
        if np.isclose(lmbda, 2.0):
            out[neg_mask] = -np.log1p(-y[neg_mask])
        else:
            out[neg_mask] = -((-y[neg_mask] + 1.0) ** (2.0 - lmbda) - 1.0) / (2.0 - lmbda)

        return out

    def _yeo_johnson_log_likelihood(self, lmbda: float, y: np.ndarray) -> float:
        """Negative profile log-likelihood of transformed normal variable."""
        n = len(y)
        y_trans = self._yeo_johnson_transform_col(y, lmbda)
        variance = np.var(y_trans, ddof=0)
        if variance <= 1e-12:
            return 1e10

        # Jacobian term
        jacobian = np.sign(y) * np.log1p(np.abs(y))
        log_lik = -0.5 * n * np.log(variance) + (lmbda - 1.0) * np.sum(jacobian)
        return -log_lik  # Return negative for minimization

    def _estimate_optimal_lambda(self, y: np.ndarray) -> float:
        """Finds optimal Yeo-Johnson lambda using Brent's method on bounded interval [-2.5, 2.5]."""
        res = optimize.minimize_scalar(
            self._yeo_johnson_log_likelihood,
            bounds=(-2.5, 2.5),
            args=(y,),
            method="bounded",
        )
        return float(res.x)

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if self.strategy == "yeo_johnson":
            self.lambdas_ = np.zeros(n_features)
            X_trans = np.zeros_like(X)
            for j in range(n_features):
                lmb = self._estimate_optimal_lambda(X[:, j])
                self.lambdas_[j] = lmb
                X_trans[:, j] = self._yeo_johnson_transform_col(X[:, j], lmb)

            # Subsequent standard scaling
            self.means_ = np.mean(X_trans, axis=0)
            stds = np.std(X_trans, axis=0)
            self.scales_ = np.where(stds == 0, 1.0, stds)

        elif self.strategy == "standard":
            self.means_ = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            self.scales_ = np.where(stds == 0, 1.0, stds)

        elif self.strategy == "robust":
            self.medians_ = np.median(X, axis=0)
            q25 = np.percentile(X, 25.0, axis=0)
            q75 = np.percentile(X, 75.0, axis=0)
            iqr = q75 - q25
            self.iqrs_ = np.where(iqr == 0, 1.0, iqr)

        elif self.strategy == "minmax":
            self.mins_ = np.min(X, axis=0)
            self.maxs_ = np.max(X, axis=0)
            ranges = self.maxs_ - self.mins_
            self.scales_ = np.where(ranges == 0, 1.0, ranges)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = np.asarray(X, dtype=float).copy()
        n_samples, n_features = X_out.shape

        if self.strategy == "yeo_johnson":
            if self.lambdas_ is not None and self.means_ is not None and self.scales_ is not None:
                for j in range(n_features):
                    X_out[:, j] = self._yeo_johnson_transform_col(X_out[:, j], self.lambdas_[j])
                X_out = (X_out - self.means_) / self.scales_

        elif self.strategy == "standard":
            if self.means_ is not None and self.scales_ is not None:
                X_out = (X_out - self.means_) / self.scales_

        elif self.strategy == "robust":
            if self.medians_ is not None and self.iqrs_ is not None:
                X_out = (X_out - self.medians_) / self.iqrs_

        elif self.strategy == "minmax":
            if self.mins_ is not None and self.scales_ is not None:
                X_out = (X_out - self.mins_) / self.scales_

        return np.nan_to_num(X_out, nan=0.0, posinf=5.0, neginf=-5.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------
# 4. Feature Engineering
# --------------------------------------------------------------------------

class FeatureEngineer:
    """
    Derives domain-specific behavioral and financial ratios from Credit Card data.
    """

    @staticmethod
    def derive_ratios(df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        eps = 1e-5
        if "BALANCE" in df_out.columns and "CREDIT_LIMIT" in df_out.columns:
            df_out["UTILIZATION_RATIO"] = df_out["BALANCE"] / (df_out["CREDIT_LIMIT"] + eps)

        if "PAYMENTS" in df_out.columns and "MINIMUM_PAYMENTS" in df_out.columns:
            df_out["PAYMENT_MIN_PAYMENT_RATIO"] = df_out["PAYMENTS"] / (df_out["MINIMUM_PAYMENTS"] + eps)

        if "ONEOFF_PURCHASES" in df_out.columns and "PURCHASES" in df_out.columns:
            df_out["ONEOFF_PURCHASE_RATIO"] = df_out["ONEOFF_PURCHASES"] / (df_out["PURCHASES"] + eps)

        if "INSTALLMENTS_PURCHASES" in df_out.columns and "PURCHASES" in df_out.columns:
            df_out["INSTALLMENT_PURCHASE_RATIO"] = df_out["INSTALLMENTS_PURCHASES"] / (df_out["PURCHASES"] + eps)

        if "CASH_ADVANCE" in df_out.columns and "PURCHASES" in df_out.columns:
            df_out["CASH_ADVANCE_RATIO"] = df_out["CASH_ADVANCE"] / (df_out["PURCHASES"] + df_out["CASH_ADVANCE"] + eps)

        if "PURCHASES_TRX" in df_out.columns and "TENURE" in df_out.columns:
            df_out["PURCHASE_TRX_VELOCITY"] = df_out["PURCHASES_TRX"] / (df_out["TENURE"] + eps)

        return df_out


# --------------------------------------------------------------------------
# 5. Full Data Preparation Pipeline
# --------------------------------------------------------------------------

class DataPreparationPipeline:
    """
    End-to-end data preparation pipeline executing:
    1. ID extraction & feature selection
    2. Behavioral ratio engineering
    3. Missing value imputation
    4. Outlier treatment
    5. Feature scaling & power transformation
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.imputer = Imputer(
            strategy=self.config.imputer_strategy,
            knn_neighbors=self.config.knn_neighbors,
            mice_max_iter=self.config.mice_max_iter,
        )
        self.outlier_handler = OutlierHandler(
            strategy=self.config.outlier_strategy,
            winsorize_limits=self.config.winsorize_limits,
            iqr_multiplier=self.config.iqr_multiplier,
            contamination=self.config.isolation_forest_contamination,
            n_trees=self.config.isolation_forest_n_trees,
        )
        self.scaler = FeatureScaler(strategy=self.config.scaling_strategy)
        self.feature_names_: List[str] = []
        self.cust_ids_: Optional[pd.Series] = None
        self.is_fitted_: bool = False

    def _prepare_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df_work = df.copy()
        cust_ids = None
        if self.config.id_column in df_work.columns:
            cust_ids = df_work[self.config.id_column]
            df_work = df_work.drop(columns=[self.config.id_column])

        if self.config.engineer_ratios:
            df_work = FeatureEngineer.derive_ratios(df_work)

        if self.config.feature_columns is not None:
            cols = [c for c in self.config.feature_columns if c in df_work.columns]
            df_work = df_work[cols]
        else:
            numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
            df_work = df_work[numeric_cols]

        return df_work, cust_ids

    def fit(self, df: pd.DataFrame) -> "DataPreparationPipeline":
        df_work, cust_ids = self._prepare_df(df)
        self.cust_ids_ = cust_ids
        self.feature_names_ = list(df_work.columns)

        X = df_work.values.astype(float)
        # 1. Impute
        X_imp = self.imputer.fit_transform(X)
        # 2. Outliers
        X_outl = self.outlier_handler.fit_transform(X_imp)
        # 3. Scale
        self.scaler.fit(X_outl)

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        df_work, _ = self._prepare_df(df)
        # Ensure column alignment
        for col in self.feature_names_:
            if col not in df_work.columns:
                df_work[col] = 0.0
        df_work = df_work[self.feature_names_]

        X = df_work.values.astype(float)
        X_imp = self.imputer.transform(X)
        X_outl = self.outlier_handler.transform(X_imp)
        X_scaled = self.scaler.transform(X_outl)
        return X_scaled

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
