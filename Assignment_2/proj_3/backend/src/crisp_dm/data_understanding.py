"""
CRISP-DM Phase 2: Data Understanding Module
Provides Kaggle Credit Card dataset ingestion, synthetic fallback generation with
realistic marginal distributions and correlation preservation, descriptive profiling,
missingness audit, skewness analysis, correlation matrices, and Hopkins clustering tendency.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist


CREDIT_CARD_COLUMNS = [
    "CUST_ID",
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

NUMERIC_FEATURE_COLUMNS = [col for col in CREDIT_CARD_COLUMNS if col != "CUST_ID"]


def generate_synthetic_credit_card_data(
    n_samples: int = 1000,
    random_state: int = 42,
    inject_missing: bool = True,
) -> pd.DataFrame:
    """
    Generates realistic synthetic credit card behavioral data modeled after the Kaggle CC dataset.
    Preserves log-normal heavy tails for financial amounts, Beta distributions for frequencies,
    Poisson/Negative Binomial counts, realistic correlation structures, and realistic missingness rates.
    """
    rng = np.random.default_rng(random_state)

    # 1. Base customer latent behaviors (archetype mixture):
    # - 35% Transactors / Regular Spenders
    # - 30% Revolvers (carry balance, low full payment)
    # - 20% Cash Advance Seekers (liquidity driven)
    # - 10% Inactive / Low-Balance
    # - 5% VIP / High Spenders
    archetypes = rng.choice(
        ["transactor", "revolver", "cash_advance", "inactive", "vip"],
        size=n_samples,
        p=[0.35, 0.30, 0.20, 0.10, 0.05],
    )

    cust_ids = [f"C{10001 + i}" for i in range(n_samples)]

    balance = np.zeros(n_samples, dtype=float)
    balance_freq = np.zeros(n_samples, dtype=float)
    purchases = np.zeros(n_samples, dtype=float)
    oneoff_purchases = np.zeros(n_samples, dtype=float)
    installments_purchases = np.zeros(n_samples, dtype=float)
    cash_advance = np.zeros(n_samples, dtype=float)
    purchases_freq = np.zeros(n_samples, dtype=float)
    oneoff_freq = np.zeros(n_samples, dtype=float)
    purchases_inst_freq = np.zeros(n_samples, dtype=float)
    cash_adv_freq = np.zeros(n_samples, dtype=float)
    cash_adv_trx = np.zeros(n_samples, dtype=int)
    purchases_trx = np.zeros(n_samples, dtype=int)
    credit_limit = np.zeros(n_samples, dtype=float)
    payments = np.zeros(n_samples, dtype=float)
    minimum_payments = np.zeros(n_samples, dtype=float)
    prc_full_payment = np.zeros(n_samples, dtype=float)
    tenure = np.zeros(n_samples, dtype=int)

    credit_limit_tiers = np.array([500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7500, 10000, 15000, 20000])

    for i in range(n_samples):
        arch = archetypes[i]
        t = int(rng.choice([6, 7, 8, 9, 10, 11, 12], p=[0.02, 0.02, 0.02, 0.02, 0.04, 0.08, 0.80]))
        tenure[i] = t

        if arch == "transactor":
            clim = float(rng.choice(credit_limit_tiers[3:10]))
            credit_limit[i] = clim
            bal = rng.lognormal(mean=6.5, sigma=0.8)
            balance[i] = min(bal, clim * 0.9)
            balance_freq[i] = float(np.clip(rng.beta(8, 2), 0.0, 1.0))
            
            tot_purch = float(rng.lognormal(mean=7.2, sigma=0.9))
            purchases[i] = tot_purch
            oneoff_ratio = float(rng.beta(3, 2))
            oneoff_purchases[i] = tot_purch * oneoff_ratio
            installments_purchases[i] = tot_purch * (1.0 - oneoff_ratio)
            
            cash_advance[i] = float(rng.lognormal(mean=4.0, sigma=1.0)) if rng.random() < 0.15 else 0.0
            purchases_freq[i] = float(np.clip(rng.beta(7, 2), 0.0, 1.0))
            oneoff_freq[i] = float(np.clip(purchases_freq[i] * oneoff_ratio, 0.0, 1.0))
            purchases_inst_freq[i] = float(np.clip(purchases_freq[i] * (1.0 - oneoff_ratio), 0.0, 1.0))
            cash_adv_freq[i] = 0.05 if cash_advance[i] > 0 else 0.0
            cash_adv_trx[i] = int(rng.poisson(1)) if cash_advance[i] > 0 else 0
            purchases_trx[i] = int(max(1, rng.poisson(25)))
            
            pay = float(tot_purch * rng.uniform(0.8, 1.2) + bal * 0.1)
            payments[i] = pay
            minimum_payments[i] = float(max(50.0, bal * 0.05 + rng.normal(50, 10)))
            prc_full_payment[i] = float(np.clip(rng.beta(5, 2), 0.0, 1.0))

        elif arch == "revolver":
            clim = float(rng.choice(credit_limit_tiers[2:8]))
            credit_limit[i] = clim
            bal = float(clim * rng.uniform(0.6, 0.95))
            balance[i] = bal
            balance_freq[i] = float(np.clip(rng.beta(9, 1), 0.8, 1.0))
            
            tot_purch = float(rng.lognormal(mean=5.5, sigma=1.0)) if rng.random() < 0.6 else 0.0
            purchases[i] = tot_purch
            oneoff_purchases[i] = tot_purch * 0.5
            installments_purchases[i] = tot_purch * 0.5
            
            cash_advance[i] = float(rng.lognormal(mean=6.0, sigma=1.0)) if rng.random() < 0.4 else 0.0
            purchases_freq[i] = float(np.clip(rng.beta(2, 4), 0.0, 0.6)) if tot_purch > 0 else 0.0
            oneoff_freq[i] = purchases_freq[i] * 0.5
            purchases_inst_freq[i] = purchases_freq[i] * 0.5
            cash_adv_freq[i] = float(np.clip(rng.beta(3, 4), 0.0, 0.6)) if cash_advance[i] > 0 else 0.0
            cash_adv_trx[i] = int(rng.poisson(3)) if cash_advance[i] > 0 else 0
            purchases_trx[i] = int(rng.poisson(5)) if tot_purch > 0 else 0
            
            pay = float(max(100.0, bal * rng.uniform(0.05, 0.15)))
            payments[i] = pay
            minimum_payments[i] = float(bal * rng.uniform(0.04, 0.08) + 30.0)
            prc_full_payment[i] = float(np.clip(rng.beta(1, 10), 0.0, 0.2))

        elif arch == "cash_advance":
            clim = float(rng.choice(credit_limit_tiers[2:9]))
            credit_limit[i] = clim
            bal = float(clim * rng.uniform(0.5, 0.9))
            balance[i] = bal
            balance_freq[i] = float(np.clip(rng.beta(8, 2), 0.7, 1.0))
            
            purchases[i] = float(rng.lognormal(mean=4.0, sigma=1.2)) if rng.random() < 0.25 else 0.0
            oneoff_purchases[i] = purchases[i] * 0.7
            installments_purchases[i] = purchases[i] * 0.3
            
            cadv = float(rng.lognormal(mean=7.8, sigma=0.8))
            cash_advance[i] = cadv
            purchases_freq[i] = 0.1 if purchases[i] > 0 else 0.0
            oneoff_freq[i] = purchases_freq[i]
            purchases_inst_freq[i] = 0.0
            cash_adv_freq[i] = float(np.clip(rng.beta(6, 2), 0.3, 1.0))
            cash_adv_trx[i] = int(max(2, rng.poisson(12)))
            purchases_trx[i] = int(rng.poisson(2)) if purchases[i] > 0 else 0
            
            pay = float(cadv * rng.uniform(0.3, 0.8) + 200)
            payments[i] = pay
            minimum_payments[i] = float(bal * 0.06 + rng.normal(80, 20))
            prc_full_payment[i] = float(np.clip(rng.beta(1, 8), 0.0, 0.25))

        elif arch == "inactive":
            clim = float(rng.choice(credit_limit_tiers[0:5]))
            credit_limit[i] = clim
            bal = float(rng.uniform(0.0, 200.0))
            balance[i] = bal
            balance_freq[i] = float(rng.uniform(0.0, 0.4))
            purchases[i] = 0.0
            oneoff_purchases[i] = 0.0
            installments_purchases[i] = 0.0
            cash_advance[i] = 0.0
            purchases_freq[i] = 0.0
            oneoff_freq[i] = 0.0
            purchases_inst_freq[i] = 0.0
            cash_adv_freq[i] = 0.0
            cash_adv_trx[i] = 0
            purchases_trx[i] = 0
            pay = float(rng.uniform(0.0, 150.0)) if bal > 0 else 0.0
            payments[i] = pay
            minimum_payments[i] = float(rng.uniform(0.0, 80.0)) if bal > 0 else 0.0
            prc_full_payment[i] = 0.0

        else:  # VIP / High Spenders
            clim = float(rng.choice(credit_limit_tiers[8:13]))
            credit_limit[i] = clim
            bal = float(rng.lognormal(mean=8.0, sigma=0.6))
            balance[i] = min(bal, clim * 0.8)
            balance_freq[i] = 1.0
            tot_purch = float(rng.lognormal(mean=8.8, sigma=0.7))
            purchases[i] = tot_purch
            oneoff_purchases[i] = tot_purch * 0.7
            installments_purchases[i] = tot_purch * 0.3
            cash_advance[i] = float(rng.lognormal(mean=6.0, sigma=1.0)) if rng.random() < 0.2 else 0.0
            purchases_freq[i] = 1.0
            oneoff_freq[i] = float(np.clip(rng.beta(7, 2), 0.5, 1.0))
            purchases_inst_freq[i] = float(np.clip(rng.beta(6, 3), 0.3, 1.0))
            cash_adv_freq[i] = 0.1 if cash_advance[i] > 0 else 0.0
            cash_adv_trx[i] = int(rng.poisson(2)) if cash_advance[i] > 0 else 0
            purchases_trx[i] = int(max(10, rng.poisson(60)))
            pay = float(tot_purch * 0.95 + rng.lognormal(mean=7.0, sigma=0.5))
            payments[i] = pay
            minimum_payments[i] = float(rng.lognormal(mean=6.5, sigma=0.5))
            prc_full_payment[i] = float(np.clip(rng.beta(6, 2), 0.4, 1.0))

    df = pd.DataFrame(
        {
            "CUST_ID": cust_ids,
            "BALANCE": np.round(balance, 4),
            "BALANCE_FREQUENCY": np.round(balance_freq, 4),
            "PURCHASES": np.round(purchases, 2),
            "ONEOFF_PURCHASES": np.round(oneoff_purchases, 2),
            "INSTALLMENTS_PURCHASES": np.round(installments_purchases, 2),
            "CASH_ADVANCE": np.round(cash_advance, 2),
            "PURCHASES_FREQUENCY": np.round(purchases_freq, 4),
            "ONEOFF_PURCHASES_FREQUENCY": np.round(oneoff_freq, 4),
            "PURCHASES_INSTALLMENTS_FREQUENCY": np.round(purchases_inst_freq, 4),
            "CASH_ADVANCE_FREQUENCY": np.round(cash_adv_freq, 4),
            "CASH_ADVANCE_TRX": cash_adv_trx,
            "PURCHASES_TRX": purchases_trx,
            "CREDIT_LIMIT": np.round(credit_limit, 2),
            "PAYMENTS": np.round(payments, 2),
            "MINIMUM_PAYMENTS": np.round(minimum_payments, 2),
            "PRC_FULL_PAYMENT": np.round(prc_full_payment, 4),
            "TENURE": tenure,
        }
    )

    if inject_missing:
        # Realistic Kaggle CC missingness: MINIMUM_PAYMENTS ~3.5%, CREDIT_LIMIT ~0.01%
        n_min_missing = int(max(1, np.round(n_samples * 0.035)))
        missing_min_idx = rng.choice(n_samples, size=n_min_missing, replace=False)
        df.loc[missing_min_idx, "MINIMUM_PAYMENTS"] = np.nan

        if n_samples >= 100:
            missing_lim_idx = rng.choice(n_samples, size=1, replace=False)
            df.loc[missing_lim_idx, "CREDIT_LIMIT"] = np.nan

    return df


def load_credit_card_data(
    filepath: Optional[str] = None,
    fallback_to_synthetic: bool = True,
    n_synthetic: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Loads credit card dataset from CSV if available, or generates synthetic data if missing or fallback requested.
    Validates required columns and data integrity.
    """
    if filepath is not None:
        try:
            df = pd.read_csv(filepath)
            # Check required numeric columns
            missing_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c not in df.columns]
            if len(missing_cols) == 0:
                return df
            else:
                if not fallback_to_synthetic:
                    raise ValueError(f"Dataset missing required columns: {missing_cols}")
        except Exception as e:
            if not fallback_to_synthetic:
                raise e

    return generate_synthetic_credit_card_data(n_samples=n_synthetic, random_state=random_state)


def compute_hopkins_statistic(
    X: np.ndarray,
    m: Optional[int] = None,
    random_state: int = 42,
) -> float:
    r"""
    Computes the Hopkins statistic for clustering tendency.
    A score near 0.5 indicates random/uniform data; values > 0.7 indicate strong clustering structure.
    
    H = sum(u_i) / (sum(u_i) + sum(w_i))
    where u_i is dist from synthetic uniform point p_i to nearest neighbor in X,
    and w_i is dist from sampled real point q_i to nearest neighbor in X \ {q_i}.
    """
    X_clean = np.asarray(X, dtype=float)
    n_samples, n_features = X_clean.shape

    if n_samples < 10:
        return 0.5

    if m is None:
        m = max(5, int(min(n_samples * 0.1, 100)))

    m = min(m, n_samples - 1)
    rng = np.random.default_rng(random_state)

    # 1. Sample m real points from X without replacement
    sample_indices = rng.choice(n_samples, size=m, replace=False)
    Q = X_clean[sample_indices]

    # Compute distances to nearest neighbors in X \ {q_i}
    # Distance matrix between Q and all X
    dist_Q_X = cdist(Q, X_clean)
    # Mask out self-distance by setting diagonal (or indexed) elements to infinity
    for row_idx, sample_idx in enumerate(sample_indices):
        dist_Q_X[row_idx, sample_idx] = np.inf
    w = np.min(dist_Q_X, axis=1)

    # 2. Generate m synthetic points uniformly within the bounding hypercube of X
    mins = np.min(X_clean, axis=0)
    maxs = np.max(X_clean, axis=0)
    # Avoid zero-range dimensions
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    P = rng.uniform(low=mins, high=mins + ranges, size=(m, n_features))

    # Compute distances from P to nearest point in X
    dist_P_X = cdist(P, X_clean)
    u = np.min(dist_P_X, axis=1)

    sum_u = float(np.sum(u))
    sum_w = float(np.sum(w))

    if sum_u + sum_w == 0:
        return 0.5

    h = sum_u / (sum_u + sum_w)
    return float(np.clip(h, 0.0, 1.0))


def compute_data_understanding_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comprehensive data understanding metrics:
    - Descriptive stats (mean, std, median, min, max, skewness, kurtosis, IQR)
    - Missingness profile
    - Skewness analysis
    - Pearson and Spearman correlation matrices
    - Hopkins clustering tendency statistic
    """
    numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in df.columns]
    num_df = df[numeric_cols].astype(float)

    stats_summary: Dict[str, Dict[str, float]] = {}
    missing_summary: Dict[str, Any] = {
        "total_missing_cells": int(num_df.isna().sum().sum()),
        "columns_with_missing": {},
    }

    skewness_dict: Dict[str, float] = {}

    for col in numeric_cols:
        series = num_df[col].dropna()
        n_missing = int(num_df[col].isna().sum())
        missing_pct = float(n_missing / len(df) * 100.0)

        if n_missing > 0:
            missing_summary["columns_with_missing"][col] = {
                "missing_count": n_missing,
                "missing_pct": round(missing_pct, 2),
            }

        if len(series) > 0:
            mean_val = float(series.mean())
            std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            median_val = float(series.median())
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr_val = float(q75 - q25)
            min_val = float(series.min())
            max_val = float(series.max())
            skew_val = float(stats.skew(series, bias=False)) if len(series) > 2 else 0.0
            kurt_val = float(stats.kurtosis(series, bias=False)) if len(series) > 3 else 0.0
            zero_pct = float((series == 0).sum() / len(series) * 100.0)

            skewness_dict[col] = round(skew_val, 4)
            stats_summary[col] = {
                "count": int(len(series)),
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "median": round(median_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "q25": round(q25, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr_val, 4),
                "skewness": round(skew_val, 4),
                "kurtosis": round(kurt_val, 4),
                "zero_pct": round(zero_pct, 2),
                "missing_count": n_missing,
                "missing_pct": round(missing_pct, 2),
            }

    # Correlation Matrices
    corr_pearson = num_df.corr(method="pearson").fillna(0.0).round(4).to_dict()
    corr_spearman = num_df.corr(method="spearman").fillna(0.0).round(4).to_dict()

    # Impute temporarily for Hopkins calculation
    imputed_matrix = num_df.fillna(num_df.median()).values
    # Standardize for Hopkins calculation
    std_devs = np.std(imputed_matrix, axis=0)
    std_devs = np.where(std_devs == 0, 1.0, std_devs)
    standardized_X = (imputed_matrix - np.mean(imputed_matrix, axis=0)) / std_devs
    hopkins_stat = compute_hopkins_statistic(standardized_X, random_state=42)

    return {
        "n_samples": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "descriptive_statistics": stats_summary,
        "missing_summary": missing_summary,
        "skewness_ranking": sorted(skewness_dict.items(), key=lambda x: abs(x[1]), reverse=True),
        "correlation_pearson": corr_pearson,
        "correlation_spearman": corr_spearman,
        "hopkins_statistic": round(hopkins_stat, 4),
    }
