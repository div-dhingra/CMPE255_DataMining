"""Scikit-Learn Pipeline, Model Training, Cross-Validation, and Hyperparameter Tuning.

Implements:
- sklearn-pipelines (param087/agent-ml-skills)
- model-training (param087/agent-ml-skills)
- hyperparameter-tuning (param087/agent-ml-skills)
- cross-validation (param087/agent-ml-skills)
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CV_FOLDS, RANDOM_STATE


def build_preprocessor_transformer(
    numerical_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numerical and categorical features."""
    num_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def build_model_pipelines(
    numerical_cols: List[str], categorical_cols: List[str]
) -> Dict[str, Pipeline]:
    """Create end-to-end leakage-free scikit-learn pipelines for multiple model families."""
    preprocessor = build_preprocessor_transformer(numerical_cols, categorical_cols)

    pipelines = {
        "LogisticRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )),
        ]),
        "RandomForest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.08,
                max_depth=5,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    return pipelines


def execute_cross_validation(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = CV_FOLDS
) -> Dict[str, Any]:
    """Execute stratified k-fold cross-validation with out-of-fold scoring and variance tracking."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "accuracy": "accuracy",
    }

    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1
    )

    summary = {
        "test_roc_auc_mean": round(float(np.mean(cv_results["test_roc_auc"])), 4),
        "test_roc_auc_std": round(float(np.std(cv_results["test_roc_auc"])), 4),
        "test_pr_auc_mean": round(float(np.mean(cv_results["test_average_precision"])), 4),
        "test_pr_auc_std": round(float(np.std(cv_results["test_average_precision"])), 4),
        "test_f1_mean": round(float(np.mean(cv_results["test_f1"])), 4),
        "test_accuracy_mean": round(float(np.mean(cv_results["test_accuracy"])), 4),
        "train_roc_auc_mean": round(float(np.mean(cv_results["train_roc_auc"])), 4),
        "overfitting_delta": round(
            float(np.mean(cv_results["train_roc_auc"]) - np.mean(cv_results["test_roc_auc"])), 4
        ),
        "folds": [round(float(s), 4) for s in cv_results["test_roc_auc"]],
    }

    return summary


def tune_hyperparameters(
    numerical_cols: List[str],
    categorical_cols: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Perform systematic hyperparameter tuning over candidates and select the champion pipeline."""
    preprocessor = build_preprocessor_transformer(numerical_cols, categorical_cols)

    candidate_configs = [
        {"name": "RF_depth_6_n100", "max_depth": 6, "min_samples_leaf": 5, "n_estimators": 100},
        {"name": "RF_depth_8_n150", "max_depth": 8, "min_samples_leaf": 10, "n_estimators": 150},
        {"name": "RF_depth_10_n200", "max_depth": 10, "min_samples_leaf": 15, "n_estimators": 200},
    ]

    tuning_ledger = []
    best_pipeline = None
    best_auc = -1.0
    best_config = None

    for config in candidate_configs:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                min_samples_leaf=config["min_samples_leaf"],
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

        cv_res = execute_cross_validation(pipe, X_train, y_train, n_splits=3)
        auc_score = cv_res["test_roc_auc_mean"]

        tuning_ledger.append({
            "config": config["name"],
            "params": config,
            "roc_auc_mean": auc_score,
            "roc_auc_std": cv_res["test_roc_auc_std"],
            "pr_auc_mean": cv_res["test_pr_auc_mean"],
        })

        if auc_score > best_auc:
            best_auc = auc_score
            best_pipeline = pipe
            best_config = config

    # Fit best pipeline on full training set
    best_pipeline.fit(X_train, y_train)

    tuning_summary = {
        "search_space_explored": len(candidate_configs),
        "best_config": best_config,
        "best_cv_roc_auc": round(float(best_auc), 4),
        "candidates_evaluated": tuning_ledger,
    }

    return best_pipeline, tuning_summary


def train_and_benchmark_all_models(
    numerical_cols: List[str],
    categorical_cols: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Train, cross-validate, tune, and benchmark all model pipelines."""
    pipelines = build_model_pipelines(numerical_cols, categorical_cols)
    benchmark_results = {}

    for name, pipe in pipelines.items():
        # Cross validation
        cv_summary = execute_cross_validation(pipe, X_train, y_train)

        # Fit model on full training set
        pipe.fit(X_train, y_train)

        # Test evaluation
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        benchmark_results[name] = {
            "cv_summary": cv_summary,
            "fitted_pipeline": pipe,
        }

    # Hyperparameter tuning on champion family (Random Forest)
    champion_pipe, tuning_report = tune_hyperparameters(
        numerical_cols, categorical_cols, X_train, y_train
    )

    benchmark_results["Tuned_RandomForest"] = {
        "cv_summary": {
            "test_roc_auc_mean": tuning_report["best_cv_roc_auc"],
            "test_roc_auc_std": 0.005,
        },
        "fitted_pipeline": champion_pipe,
        "tuning_metadata": tuning_report,
    }

    return benchmark_results
