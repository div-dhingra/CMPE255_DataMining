"""Data Preparation, Cleaning, Feature Engineering, and Imbalanced Data Module.

Implements:
- data-cleaning (param087/agent-ml-skills)
- feature-engineering (param087/agent-ml-skills)
- imbalanced-data (param087/agent-ml-skills)
- context-packager (nimrodfisher/data-analytics-skills)
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def clean_dataset_leakage_safe(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Clean data using strict train-only statistics to avoid target/data leakage.

    - Computes TotalCharges median strictly on the training set
    - Imputes test set using the train-set median
    - Normalizes binary target column 'Churn' into {0, 1}
    """
    train_clean = train_df.copy()
    test_clean = test_df.copy()

    # Parse numeric TotalCharges
    train_clean["TotalCharges"] = pd.to_numeric(train_clean["TotalCharges"], errors="coerce")
    test_clean["TotalCharges"] = pd.to_numeric(test_clean["TotalCharges"], errors="coerce")

    # Fit median strictly on train
    train_total_median = float(train_clean["TotalCharges"].median())

    # Impute on both splits
    train_imputed_count = int(train_clean["TotalCharges"].isna().sum())
    test_imputed_count = int(test_clean["TotalCharges"].isna().sum())

    train_clean["TotalCharges"] = train_clean["TotalCharges"].fillna(train_total_median)
    test_clean["TotalCharges"] = test_clean["TotalCharges"].fillna(train_total_median)

    # Encode target column: 'Yes' -> 1, 'No' -> 0
    train_clean["target"] = (train_clean[TARGET_COLUMN] == "Yes").astype(int)
    test_clean["target"] = (test_clean[TARGET_COLUMN] == "Yes").astype(int)

    cleaning_metadata = {
        "train_imputed_missing_count": train_imputed_count,
        "test_imputed_missing_count": test_imputed_count,
        "imputation_strategy": "train_set_median",
        "imputed_value": round(train_total_median, 2),
        "target_mapping": {"No": 0, "Yes": 1},
        "leakage_safety": "GUARANTEED: Imputation statistic learned exclusively from train_df",
    }

    return train_clean, test_clean, cleaning_metadata


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-specific features for telecom customer churn."""
    df_feat = df.copy()

    # 1. Tenure Cohorts (Time-based groupings)
    df_feat["TenureCohort"] = pd.cut(
        df_feat["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    ).astype(str)

    # 2. Service Bundle Count (Sum of tech add-ons)
    bundle_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    service_sum = pd.Series(0, index=df_feat.index)
    for col in bundle_cols:
        if col in df_feat.columns:
            service_sum += (df_feat[col] == "Yes").astype(int)
    df_feat["ServiceBundleCount"] = service_sum

    # 3. High-Risk Indicators
    df_feat["HasFiberOptic"] = (df_feat["InternetService"] == "Fiber optic").astype(int)
    df_feat["IsMonthToMonth"] = (df_feat["Contract"] == "Month-to-month").astype(int)
    df_feat["IsElectronicCheck"] = (df_feat["PaymentMethod"] == "Electronic check").astype(int)

    # 4. Monthly to Total Ratio (Billing velocity)
    df_feat["MonthlyToTotalRatio"] = df_feat["MonthlyCharges"] / (df_feat["TotalCharges"] + 1.0)

    # 5. High-Spend Risk Flag (Month-to-month and high monthly charges)
    df_feat["HighSpendRisk"] = (
        (df_feat["IsMonthToMonth"] == 1) & (df_feat["MonthlyCharges"] > 70.0)
    ).astype(int)

    return df_feat


def handle_imbalanced_data(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Handle class imbalance through class weighting and SMOTE oversampling."""
    # Compute balanced class weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): round(float(w), 4) for c, w in zip(classes, weights)}

    # Minority vs Majority counts
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    imbalance_ratio = round(neg_count / max(1, pos_count), 2)

    # Synthetic Minority Oversampling (SMOTE-style interpolation on numeric attributes)
    # We implement a self-contained numpy-based oversampler for numeric features
    pos_indices = np.where(y_train == 1)[0]
    num_to_synthesize = neg_count - pos_count

    rng = np.random.default_rng(RANDOM_STATE)
    synthetic_samples_list = []

    if num_to_synthesize > 0 and len(pos_indices) > 1:
        sampled_base_idx = rng.choice(pos_indices, size=num_to_synthesize, replace=True)
        sampled_neighbor_idx = rng.choice(pos_indices, size=num_to_synthesize, replace=True)
        alphas = rng.uniform(0.1, 0.9, size=(num_to_synthesize, 1))

        # Copy base DataFrame rows
        base_rows = X_train.iloc[sampled_base_idx].copy().reset_index(drop=True)
        neigh_rows = X_train.iloc[sampled_neighbor_idx].copy().reset_index(drop=True)

        # Interpolate numeric features
        for num_col in ["tenure", "MonthlyCharges", "TotalCharges", "MonthlyToTotalRatio"]:
            if num_col in base_rows.columns:
                diff = neigh_rows[num_col].values - base_rows[num_col].values
                base_rows[num_col] = base_rows[num_col].values + (alphas.flatten() * diff)

        X_train_resampled = pd.concat([X_train, base_rows], ignore_index=True)
        y_train_resampled = pd.concat([y_train, pd.Series(1, index=range(num_to_synthesize))], ignore_index=True)
    else:
        X_train_resampled = X_train.copy()
        y_train_resampled = y_train.copy()

    imbalance_metadata = {
        "original_distribution": {"retained_0": neg_count, "churned_1": pos_count},
        "imbalance_ratio": f"{imbalance_ratio}:1",
        "balanced_class_weights": class_weight_dict,
        "smote_resampled_distribution": {
            "retained_0": int((y_train_resampled == 0).sum()),
            "churned_1": int((y_train_resampled == 1).sum()),
        },
        "synthetic_samples_generated": num_to_synthesize,
    }

    return X_train_resampled, y_train_resampled, imbalance_metadata


def package_preprocessing_context(
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
    cleaning_meta: Dict[str, Any],
    imbalance_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Package minimum viable context for downstream modeling and agent handoff."""
    all_features = [
        col for col in train_clean.columns if col not in [ID_COLUMN, TARGET_COLUMN, "target"]
    ]
    numerical = [c for c in all_features if c in NUMERICAL_FEATURES or c in ["MonthlyToTotalRatio", "ServiceBundleCount"]]
    categorical = [c for c in all_features if c not in numerical]

    context = {
        "train_samples": len(train_clean),
        "test_samples": len(test_clean),
        "total_features": len(all_features),
        "numerical_features": numerical,
        "categorical_features": categorical,
        "cleaning_metadata": cleaning_meta,
        "imbalance_metadata": imbalance_meta,
        "feature_list": all_features,
    }

    return context


def prepare_data_splits(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Execute complete Phase 3 Data Preparation pipeline and return prepared splits."""
    # 1. Train / Test Split (80/20 Stratified)
    train_raw, test_raw = train_test_split(
        raw_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=raw_df[TARGET_COLUMN],
    )

    # 2. Leakage-safe cleaning
    train_clean, test_clean, cleaning_meta = clean_dataset_leakage_safe(train_raw, test_raw)

    # 3. Feature engineering
    train_feat = apply_feature_engineering(train_clean)
    test_feat = apply_feature_engineering(test_clean)

    # 4. Separate X and y
    feature_cols = [c for c in train_feat.columns if c not in [ID_COLUMN, TARGET_COLUMN, "target"]]
    X_train = train_feat[feature_cols]
    y_train = train_feat["target"]
    X_test = test_feat[feature_cols]
    y_test = test_feat["target"]

    # 5. Imbalanced data handling
    X_train_smote, y_train_smote, imbalance_meta = handle_imbalanced_data(X_train, y_train)

    # 6. Context packaging
    context = package_preprocessing_context(train_feat, test_feat, cleaning_meta, imbalance_meta)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "cleaning_metadata": cleaning_meta,
        "imbalance_metadata": imbalance_meta,
        "context_package": context,
    }
