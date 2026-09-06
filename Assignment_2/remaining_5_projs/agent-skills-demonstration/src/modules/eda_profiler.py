"""Exploratory Data Analysis and Pandas Patterns Module.

Implements:
- exploratory-data-analysis (param087/agent-ml-skills)
- programmatic-eda (nimrodfisher/data-analytics-skills)
- pandas-patterns (param087/agent-ml-skills)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    FIGURES_DIR,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)


def load_raw_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


def execute_programmatic_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive programmatic exploratory data analysis.

    Returns detailed profiling including row/col shapes, missingness,
    data types, numerical summaries, categorical frequencies, and churn balance.
    """
    total_rows, total_cols = df.shape

    # Missing value detection (including blank spaces in string columns)
    missing_counts = {}
    for col in df.columns:
        if df[col].dtype == "object":
            # Check for empty string or whitespace
            blanks = (df[col].astype(str).str.strip() == "").sum()
            nans = df[col].isna().sum()
            missing_counts[col] = int(blanks + nans)
        else:
            missing_counts[col] = int(df[col].isna().sum())

    # Numerical feature profiling
    num_profile = {}
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            # Coerce to numeric if currently object (e.g. TotalCharges)
            series = pd.to_numeric(df[col], errors="coerce")
            num_profile[col] = {
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "q25": round(float(series.quantile(0.25)), 2),
                "median": round(float(series.median()), 2),
                "q75": round(float(series.quantile(0.75)), 2),
                "max": round(float(series.max()), 2),
                "skewness": round(float(series.skew()), 2),
            }

    # Categorical feature profiling
    cat_profile = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            top_vals = df[col].value_counts().head(5).to_dict()
            cat_profile[col] = {
                "unique_values": int(df[col].nunique()),
                "top_categories": {str(k): int(v) for k, v in top_vals.items()},
            }

    # Target class distribution
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    churn_rate = round(float(target_counts.get("Yes", 0) / total_rows * 100), 2)

    # Multicollinearity check among numericals
    numeric_df = pd.DataFrame()
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            numeric_df[col] = pd.to_numeric(df[col], errors="coerce")
    corr_matrix = numeric_df.corr().round(3).to_dict()

    summary = {
        "dataset_shape": {"rows": total_rows, "columns": total_cols},
        "target_distribution": {
            "Yes": int(target_counts.get("Yes", 0)),
            "No": int(target_counts.get("No", 0)),
            "churn_rate_pct": churn_rate,
        },
        "missing_values": missing_counts,
        "numerical_profile": num_profile,
        "categorical_profile": cat_profile,
        "correlation_matrix": corr_matrix,
    }

    return summary


def apply_pandas_patterns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Demonstrate idiomatic, high-performance vectorized pandas patterns.

    - Vectorized type casting and whitespace stripping (no slow row iterators)
    - Safe boolean masking without SettingWithCopyWarning
    - Memory footprint downcasting
    """
    initial_memory_kb = df.memory_usage(deep=True).sum() / 1024

    # Copy explicitly to prevent SettingWithCopyWarning
    df_clean = df.copy()

    # Vectorized string strip
    for col in df_clean.select_dtypes(include=["object", "string"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Vectorized numeric parsing for TotalCharges
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # Vectorized downcasting: float64 -> float32
    float_cols = df_clean.select_dtypes(include=["float64"]).columns
    df_clean[float_cols] = df_clean[float_cols].astype("float32")

    # Vectorized downcasting: SeniorCitizen int64 -> int8
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype("int8")

    final_memory_kb = df_clean.memory_usage(deep=True).sum() / 1024
    memory_savings_pct = round((1.0 - final_memory_kb / initial_memory_kb) * 100, 2)

    patterns_report = {
        "initial_memory_kb": round(float(initial_memory_kb), 2),
        "optimized_memory_kb": round(float(final_memory_kb), 2),
        "memory_savings_pct": memory_savings_pct,
        "vectorized_transforms_applied": [
            "Vectorized string strip across all object columns",
            "Vectorized coercion of TotalCharges whitespace to float32",
            "Downcasting float64 features to float32",
            "Downcasting integer flags to int8",
            "Safe deep-copy memory indexing avoiding SettingWithCopyWarning",
        ],
    }

    return df_clean, patterns_report


def generate_eda_figures(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> List[str]:
    """Generate and save publication-grade EDA figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    # 1. Churn Class Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET_COLUMN].value_counts()
    colors = ["#2b5c8f", "#d9534f"]
    bars = ax.bar(["Retained (No)", "Churned (Yes)"], counts.values, color=colors, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,} ({yval/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title("Customer Churn Distribution (Class Imbalance)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Customer Count")
    ax.set_ylim(0, max(counts.values) * 1.15)
    sns.despine()
    plt.tight_layout()
    churn_fig_path = output_dir / "churn_distribution.png"
    plt.savefig(churn_fig_path, dpi=200)
    plt.close()
    generated_plots.append(str(churn_fig_path))

    # 2. Tenure Distribution by Churn Status
    fig, ax = plt.subplots(figsize=(8, 4))
    tenure_no = pd.to_numeric(df[df[TARGET_COLUMN] == "No"]["tenure"], errors="coerce").dropna()
    tenure_yes = pd.to_numeric(df[df[TARGET_COLUMN] == "Yes"]["tenure"], errors="coerce").dropna()
    ax.hist(tenure_no, bins=30, alpha=0.6, label="Retained (No)", color="#2b5c8f", density=True)
    ax.hist(tenure_yes, bins=30, alpha=0.6, label="Churned (Yes)", color="#d9534f", density=True)
    ax.set_title("Customer Tenure Density: Retained vs Churned", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Probability Density")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    tenure_fig_path = output_dir / "tenure_distribution.png"
    plt.savefig(tenure_fig_path, dpi=200)
    plt.close()
    generated_plots.append(str(tenure_fig_path))

    # 3. Correlation Heatmap
    numeric_df = pd.DataFrame()
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            numeric_df[col] = pd.to_numeric(df[col], errors="coerce")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax, square=True)
    ax.set_title("Numerical Features Correlation Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    corr_fig_path = output_dir / "numerical_correlations.png"
    plt.savefig(corr_fig_path, dpi=200)
    plt.close()
    generated_plots.append(str(corr_fig_path))

    return generated_plots
