"""
Autoresearch Experiment Logger and Ablation Analysis Module.
Provides:
- Formatted JSONL and CSV logging of step metrics, trajectories, and parameters
- Leaderboard extraction and best-so-far trajectory tracking
- Systematic ablation breakdown quantifying parameter importance and stage contributions
"""

import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

from .hill_climber import StepLog
from .search_space import SearchSpace


class ExperimentLogger:
    """
    Manages telemetry logging, persistence (JSONL, CSV), and trajectory analytics
    for autonomous clustering research runs.
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        self.logs: List[StepLog] = []

    def log_step(self, step_log: StepLog) -> None:
        """Appends a StepLog record to the in-memory ledger."""
        self.logs.append(step_log)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Returns all step logs as list of serializable dictionaries."""
        return [log.to_dict() for log in self.logs]

    def to_dataframe(self) -> pd.DataFrame:
        """Converts logged trajectory into a structured flattened pandas DataFrame."""
        rows = []
        for log in self.logs:
            row = {
                "step": log.step,
                "timestamp": log.timestamp,
                "algorithm": log.algorithm,
                "fitness": log.fitness,
                "delta_fitness": log.delta_fitness,
                "current_fitness": log.current_fitness,
                "best_fitness": log.best_fitness,
                "silhouette": log.silhouette,
                "davies_bouldin": log.davies_bouldin,
                "calinski_harabasz": log.calinski_harabasz,
                "stability_ari": log.stability_ari,
                "n_clusters": log.n_clusters,
                "noise_ratio": log.noise_ratio,
                "temperature": log.temperature,
                "accepted": log.accepted,
                "restart": log.restart,
                "stagnation_count": log.stagnation_count,
                "execution_time_ms": log.execution_time_ms,
                "error": log.error or "",
            }
            # Flatten candidate params
            flat_cand = SearchSpace.flatten_config(log.candidate_theta)
            row.update(flat_cand)
            rows.append(row)

        return pd.DataFrame(rows)

    def save_jsonl(self, filepath: Union[str, Path]) -> str:
        """
        Saves experiment history to a formatted JSONL file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for log in self.logs:
                f.write(json.dumps(log.to_dict()) + "\n")

        return str(path.resolve())

    def save_csv(self, filepath: Union[str, Path]) -> str:
        """
        Saves flattened experiment history to a CSV file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return str(path.resolve())

    def get_best_trajectory(self) -> List[Dict[str, Any]]:
        """
        Extracts monotonic best-fitness trajectory across optimization steps.
        """
        trajectory = []
        best_so_far = -np.inf

        for log in self.logs:
            if log.fitness > best_so_far:
                best_so_far = log.fitness
                trajectory.append({
                    "step": log.step,
                    "timestamp": log.timestamp,
                    "fitness": round(log.fitness, 4),
                    "algorithm": log.algorithm,
                    "silhouette": round(log.silhouette, 4),
                    "davies_bouldin": round(log.davies_bouldin, 4),
                    "calinski_harabasz": round(log.calinski_harabasz, 4),
                    "stability_ari": round(log.stability_ari, 4),
                    "n_clusters": log.n_clusters,
                    "config": copy.deepcopy(log.candidate_theta),
                })

        return trajectory

    def get_leaderboard(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Returns top N distinct configurations ranked by fitness score.
        """
        seen_hashes = set()
        ranked_candidates = []

        # Sort logs by fitness descending
        valid_logs = [log for log in self.logs if log.fitness > 0 and log.error is None]
        sorted_logs = sorted(valid_logs, key=lambda x: x.fitness, reverse=True)

        for log in sorted_logs:
            c_hash = SearchSpace.config_hash(log.candidate_theta)
            if c_hash not in seen_hashes:
                seen_hashes.add(c_hash)
                ranked_candidates.append({
                    "rank": len(ranked_candidates) + 1,
                    "step_discovered": log.step,
                    "fitness": round(log.fitness, 4),
                    "algorithm": log.algorithm,
                    "silhouette": round(log.silhouette, 4),
                    "davies_bouldin": round(log.davies_bouldin, 4),
                    "calinski_harabasz": round(log.calinski_harabasz, 4),
                    "stability_ari": round(log.stability_ari, 4),
                    "n_clusters": log.n_clusters,
                    "noise_ratio": round(log.noise_ratio, 4),
                    "configuration": copy.deepcopy(log.candidate_theta),
                })
                if len(ranked_candidates) >= top_n:
                    break

        return ranked_candidates

    def get_summary(self) -> Dict[str, Any]:
        """
        Computes global summary metrics of the experiment run.
        """
        if not self.logs:
            return {"status": "empty", "total_steps": 0}

        fitnesses = [log.fitness for log in self.logs if log.fitness >= 0]
        accepted_count = sum(1 for log in self.logs if log.accepted)
        restart_count = sum(1 for log in self.logs if log.restart)
        best_log = max(self.logs, key=lambda x: x.fitness)
        initial_log = self.logs[0]

        return {
            "run_id": self.run_id,
            "total_steps": len(self.logs),
            "accepted_steps": accepted_count,
            "acceptance_rate": round(accepted_count / len(self.logs), 4) if self.logs else 0.0,
            "restart_count": restart_count,
            "initial_fitness": round(initial_log.fitness, 4),
            "best_fitness": round(best_log.fitness, 4),
            "fitness_gain": round(best_log.fitness - initial_log.fitness, 4),
            "mean_fitness": round(float(np.mean(fitnesses)), 4) if fitnesses else 0.0,
            "best_algorithm": best_log.algorithm,
            "best_config": copy.deepcopy(best_log.candidate_theta),
        }


def export_ablation_analysis(
    history: List[Union[StepLog, Dict[str, Any]]],
    baseline_theta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Performs comprehensive ablation analysis over optimization history:
    1. Parameter Importance: mean fitness and standard deviation grouped by categorical choices.
    2. Stage Contributions: marginal utility of Imputation, Outlier Handling, Scaling, PCA, and Ratios.
    3. Baseline vs. Optimized Comparison with delta improvements.
    """
    if not history:
        return {"error": "Empty history provided for ablation analysis."}

    # Convert StepLogs to dicts if needed
    step_dicts = []
    for item in history:
        if isinstance(item, StepLog):
            step_dicts.append(item.to_dict())
        else:
            step_dicts.append(item)

    df_rows = []
    for d in step_dicts:
        row = {
            "step": d.get("step", 0),
            "fitness": float(d.get("fitness", -1.0)),
            "silhouette": float(d.get("silhouette", 0.0)),
            "davies_bouldin": float(d.get("davies_bouldin", 0.0)),
            "calinski_harabasz": float(d.get("calinski_harabasz", 0.0)),
            "stability_ari": float(d.get("stability_ari", 0.0)),
            "n_clusters": d.get("n_clusters", 0),
            "algorithm": d.get("algorithm", "kmeans"),
        }
        cand = d.get("candidate_theta", {})
        flat = SearchSpace.flatten_config(cand)
        row.update(flat)
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    valid_df = df[df["fitness"] >= 0].copy()

    if valid_df.empty:
        return {"error": "No valid evaluated steps found."}

    # 1. Parameter Stage Groupings & Marginal Means
    stage_breakdown = {}
    categorical_columns = [
        ("prep_imputer", "Imputation Strategy"),
        ("prep_outlier_handler", "Outlier Strategy"),
        ("prep_scaler", "Scaling Strategy"),
        ("prep_pca_components", "PCA Components"),
        ("prep_feature_engineering", "Feature Engineering"),
        ("algorithm", "Clustering Algorithm"),
    ]

    for col, stage_name in categorical_columns:
        if col in valid_df.columns:
            group = valid_df.groupby(col)["fitness"].agg(["mean", "std", "count", "max"]).reset_index()
            group_list = []
            for _, r in group.iterrows():
                val = r[col]
                group_list.append({
                    "choice": str(val) if val is not None else "None",
                    "mean_fitness": round(float(r["mean"]), 4),
                    "std_fitness": round(float(r["std"]) if pd.notna(r["std"]) else 0.0, 4),
                    "max_fitness": round(float(r["max"]), 4),
                    "sample_count": int(r["count"]),
                })
            # Sort by mean_fitness descending
            group_list.sort(key=lambda x: x["mean_fitness"], reverse=True)
            stage_breakdown[col] = {
                "stage_name": stage_name,
                "results": group_list,
            }

    # 2. Parameter Importance Ranking (via Variance of Means)
    parameter_importance = []
    for col, stage_name in categorical_columns:
        if col in valid_df.columns:
            means = valid_df.groupby(col)["fitness"].mean()
            if len(means) > 1:
                spread = float(means.max() - means.min())
                parameter_importance.append({
                    "parameter": stage_name,
                    "column_key": col,
                    "fitness_spread": round(spread, 4),
                    "best_choice": str(means.idxmax()),
                })

    parameter_importance.sort(key=lambda x: x["fitness_spread"], reverse=True)

    # 3. Baseline vs Best Comparison
    best_row = valid_df.loc[valid_df["fitness"].idxmax()]
    initial_row = valid_df.iloc[0]

    baseline_fitness = float(initial_row["fitness"])
    best_fitness = float(best_row["fitness"])
    delta_fitness = best_fitness - baseline_fitness

    comparison = {
        "baseline_step": int(initial_row["step"]),
        "baseline_fitness": round(baseline_fitness, 4),
        "baseline_silhouette": round(float(initial_row["silhouette"]), 4),
        "baseline_davies_bouldin": round(float(initial_row["davies_bouldin"]), 4),
        "baseline_calinski_harabasz": round(float(initial_row["calinski_harabasz"]), 4),
        "baseline_stability_ari": round(float(initial_row["stability_ari"]), 4),
        "baseline_algorithm": str(initial_row["algorithm"]),
        "best_step": int(best_row["step"]),
        "best_fitness": round(best_fitness, 4),
        "best_silhouette": round(float(best_row["silhouette"]), 4),
        "best_davies_bouldin": round(float(best_row["davies_bouldin"]), 4),
        "best_calinski_harabasz": round(float(best_row["calinski_harabasz"]), 4),
        "best_stability_ari": round(float(best_row["stability_ari"]), 4),
        "best_algorithm": str(best_row["algorithm"]),
        "delta_fitness": round(delta_fitness, 4),
        "delta_silhouette": round(float(best_row["silhouette"] - initial_row["silhouette"]), 4),
        "delta_davies_bouldin": round(float(best_row["davies_bouldin"] - initial_row["davies_bouldin"]), 4),
        "delta_calinski_harabasz": round(float(best_row["calinski_harabasz"] - initial_row["calinski_harabasz"]), 4),
        "delta_stability_ari": round(float(best_row["stability_ari"] - initial_row["stability_ari"]), 4),
    }

    return {
        "stage_breakdown": stage_breakdown,
        "parameter_importance": parameter_importance,
        "comparison": comparison,
        "total_evaluated_steps": len(valid_df),
    }
