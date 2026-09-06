"""
CRISP-DM Phase 6: Deployment, Personas & Profiling Module
Extracts cluster-level centroid summaries, ANOVA feature importance,
normalized multi-axis radar profiles, and business customer personas.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


# Key features highlighted in radar charts
RADAR_KEY_FEATURES = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "PRC_FULL_PAYMENT",
]


def compute_cluster_centroids(
    df_features: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Computes mean feature values for each cluster.
    """
    df_work = df_features.select_dtypes(include=[np.number]).copy()
    df_work["CLUSTER"] = labels
    centroids = df_work.groupby("CLUSTER").mean()
    return centroids


def compute_radar_profiles(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    features_to_include: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Computes normalized [0, 1] radar chart metrics for each cluster.
    """
    df_num = df_features.select_dtypes(include=[np.number]).copy()
    if features_to_include is None:
        target_features = [f for f in RADAR_KEY_FEATURES if f in df_num.columns]
        if len(target_features) == 0:
            target_features = list(df_num.columns)[:8]
    else:
        target_features = [f for f in features_to_include if f in df_num.columns]

    df_subset = df_num[target_features].copy()

    # Min-max normalize each feature across global dataset
    mins = df_subset.min()
    maxs = df_subset.max()
    ranges = maxs - mins
    ranges = ranges.replace(0.0, 1.0)
    df_norm = (df_subset - mins) / ranges

    df_norm["CLUSTER"] = labels
    cluster_radar = df_norm.groupby("CLUSTER").mean()

    result: Dict[str, Dict[str, float]] = {}
    for cluster_id, row in cluster_radar.iterrows():
        c_key = f"cluster_{cluster_id}" if cluster_id >= 0 else "noise"
        result[c_key] = {feat: round(float(val), 4) for feat, val in row.items()}

    return result


def compute_feature_importance(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Computes ANOVA F-statistic variance ratio for each feature across clusters.
    Higher F-statistic indicates the feature strongly differentiates the clusters.
    """
    X_arr = np.asarray(X, dtype=float)
    non_noise = labels >= 0
    X_valid = X_arr[non_noise]
    labels_valid = labels[non_noise]
    unique_labels = np.unique(labels_valid)

    if len(unique_labels) < 2:
        return {feat: 1.0 / len(feature_names) for feat in feature_names}

    f_scores = []
    for j in range(X_valid.shape[1]):
        groups = [X_valid[labels_valid == c, j] for c in unique_labels]
        # Filter empty groups
        groups = [g for g in groups if len(g) > 1]
        if len(groups) > 1:
            try:
                f_val, _ = stats.f_oneway(*groups)
                f_scores.append(float(np.nan_to_num(f_val, nan=0.0)))
            except Exception:
                f_scores.append(0.0)
        else:
            f_scores.append(0.0)

    total_f = sum(f_scores)
    if total_f == 0:
        normalized_scores = [1.0 / len(f_scores)] * len(f_scores)
    else:
        normalized_scores = [f / total_f for f in f_scores]

    # Map to feature names
    importance_map = {}
    for idx, name in enumerate(feature_names[:len(normalized_scores)]):
        importance_map[name] = round(float(normalized_scores[idx]), 4)

    # Sort descending by importance
    return dict(sorted(importance_map.items(), key=lambda x: x[1], reverse=True))


def generate_cluster_personas(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Synthesizes actionable domain business personas for each discovered cluster
    based on relative deviations from global dataset means (Z-scores).
    """
    df_num = df_features.select_dtypes(include=[np.number]).copy()
    if feature_names is not None:
        valid_cols = [c for c in feature_names if c in df_num.columns]
        df_num = df_num[valid_cols]

    total_samples = len(labels)
    unique_labels = np.unique(labels)

    # Compute global statistics
    global_means = df_num.mean()
    global_stds = df_num.std().replace(0.0, 1.0)

    personas = []
    radar_profiles = compute_radar_profiles(df_num, labels)

    for c in unique_labels:
        cluster_mask = labels == c
        n_samples = int(np.sum(cluster_mask))
        percentage = round(float(n_samples / total_samples * 100.0), 2)
        c_name = f"Cluster {c}" if c >= 0 else "Noise Points"

        cluster_pts = df_num[cluster_mask]
        cluster_means = cluster_pts.mean()

        # Compute relative Z-scores
        z_scores = (cluster_means - global_means) / global_stds
        top_distinguishing = []
        for col, z in z_scores.sort_values(ascending=False).items():
            top_distinguishing.append({
                "feature": str(col),
                "cluster_mean": round(float(cluster_means[col]), 2),
                "global_mean": round(float(global_means[col]), 2),
                "z_score": round(float(z), 2),
            })

        # Heuristic archetype classification
        arch_type = "Standard Customer Segment"
        description = "Customers with balanced behavioral attributes across all credit features."
        marketing = "Standard automated customer retention and loyalty point reminders."

        if c == -1:
            arch_type = "Anomalies / Noise"
            description = "Irregular customer accounts that do not conform to standard spending or payment densities."
            marketing = "Flag for risk review, credit limit audit, or manual fraud investigation."
        else:
            # Check archetype criteria
            purch_freq = cluster_means.get("PURCHASES_FREQUENCY", 0.5)
            cadv_freq = cluster_means.get("CASH_ADVANCE_FREQUENCY", 0.1)
            prc_full = cluster_means.get("PRC_FULL_PAYMENT", 0.1)
            bal_z = z_scores.get("BALANCE", 0.0)
            purch_z = z_scores.get("PURCHASES", 0.0)
            cadv_z = z_scores.get("CASH_ADVANCE", 0.0)
            limit_z = z_scores.get("CREDIT_LIMIT", 0.0)

            if purch_z > 1.0 and limit_z > 0.8:
                arch_type = "VIP / High-Value Spenders"
                description = "Affluent customers with substantial credit limits, high transaction volume, and major one-off purchases."
                marketing = "Offer premium concierge perks, exclusive travel rewards, and premium card tier upgrades."
            elif purch_freq > 0.6 and prc_full > 0.3 and cadv_z < 0.0:
                arch_type = "Transactors / Active Spenders"
                description = "Customers who use credit cards frequently for everyday transactions and pay off full balances promptly."
                marketing = "Incentivize with point multipliers, cash-back on groceries/dining, and co-branded retail promotions."
            elif cadv_z > 0.8 or cadv_freq > 0.4:
                arch_type = "Cash Advance / Liquidity Seekers"
                description = "Customers who frequently take cash advances against credit lines, carrying high balances with low purchase activity."
                marketing = "Promote structured debt-consolidation loans, lower APR balance transfers, and emergency installment options."
            elif bal_z > 0.5 and prc_full < 0.15:
                arch_type = "Revolvers / Interest Payers"
                description = "Customers maintaining heavy revolving balances month-over-month while making minimum or partial payments."
                marketing = "Cross-sell balance transfer cards, 0% intro APR promotions, and automated minimum payment nudges."
            elif purch_z < -0.3 and bal_z < -0.3:
                arch_type = "Low-Engagement / Inactive"
                description = "Dormant accounts with minimal spending, low balances, and infrequent card interactions."
                marketing = "Deploy re-engagement marketing campaigns with statement credits, renewal bonuses, and fee waivers."

        radar_key = f"cluster_{c}" if c >= 0 else "noise"
        cluster_radar = radar_profiles.get(radar_key, {})

        personas.append({
            "cluster_id": int(c),
            "persona_name": arch_type,
            "sample_count": n_samples,
            "percentage": percentage,
            "business_description": description,
            "marketing_strategy": marketing,
            "top_distinguishing_features": top_distinguishing[:5],
            "radar_metrics": cluster_radar,
        })

    return personas
