"""
Comprehensive Unit & Integration Test Suite for Autoresearch & Hill-Climbing Engine.
Tests:
- Search space boundaries, sampling, mutation, hashing, neighbor generation
- Composite multi-objective fitness calculation, normalization, penalties, and edge cases
- Hill-climbing optimizer step transitions, simulated annealing acceptance, restarts, async streaming
- Experiment logger persistence (JSONL, CSV), trajectory extraction, leaderboard ranking
- Ablation analysis breakdown and parameter importance
"""

import asyncio
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crisp_dm.data_understanding import generate_synthetic_credit_card_data
from autoresearch.search_space import (
    SearchSpace,
    PREPROCESSING_SPACE,
    ALGORITHM_CHOICES,
    ALGORITHM_HYPERPARAMETERS,
)
from autoresearch.objective import (
    CompositeObjective,
    ObjectiveWeights,
    ObjectiveEvaluation,
)
from autoresearch.hill_climber import (
    HillClimbingOptimizer,
    StepLog,
    OptimizationResult,
)
from autoresearch.experiment_logger import (
    ExperimentLogger,
    export_ablation_analysis,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Provides a small synthetic dataset for fast deterministic testing."""
    return generate_synthetic_credit_card_data(n_samples=150, random_state=42)


# ==============================================================================
# 1. Search Space Unit Tests
# ==============================================================================

def test_search_space_initialization():
    space = SearchSpace()
    assert len(space.algorithms) == 6
    assert set(space.algorithms) == {"kmeans", "kmedoids", "dbscan", "hdbscan", "agglomerative", "gmm"}
    assert "yeo_johnson" in space.preprocessing_space["scaler"]
    assert "median" in space.preprocessing_space["imputer"]
    assert "winsorize" in space.preprocessing_space["outlier_handler"]


@pytest.mark.parametrize("algorithm", ALGORITHM_CHOICES)
def test_search_space_default_configurations(algorithm: str):
    space = SearchSpace()
    default_config = space.get_default_configuration(algorithm)
    assert default_config["algorithm"] == algorithm
    assert "preprocessing" in default_config
    assert "hyperparameters" in default_config
    assert default_config["preprocessing"]["scaler"] == "yeo_johnson"
    assert default_config["preprocessing"]["feature_engineering"] is True


def test_search_space_sample_random_configuration():
    space = SearchSpace()
    for _ in range(20):
        config = space.sample_random_configuration(random_state=None)
        assert config["algorithm"] in ALGORITHM_CHOICES
        prep = config["preprocessing"]
        assert prep["imputer"] in PREPROCESSING_SPACE["imputer"]
        assert prep["outlier_handler"] in PREPROCESSING_SPACE["outlier_handler"]
        assert prep["scaler"] in PREPROCESSING_SPACE["scaler"]
        assert prep["pca_components"] in PREPROCESSING_SPACE["pca_components"]
        assert isinstance(prep["feature_engineering"], bool)

        alg = config["algorithm"]
        hyp = config["hyperparameters"]
        alg_spec = ALGORITHM_HYPERPARAMETERS[alg]
        for p_name, p_spec in alg_spec.items():
            assert p_name in hyp
            if p_spec["type"] == "int":
                assert p_spec["low"] <= hyp[p_name] <= p_spec["high"]
            elif p_spec["type"] == "float":
                assert p_spec["low"] <= hyp[p_name] <= p_spec["high"]
            elif p_spec["type"] == "choice":
                assert hyp[p_name] in p_spec["choices"]


def test_search_space_mutation_operator():
    space = SearchSpace()
    base_config = space.get_default_configuration("kmeans")
    base_hash = space.config_hash(base_config)

    mutated_configs = []
    for i in range(10):
        mutated = space.mutate(base_config, mutation_rate=0.5, random_state=42 + i)
        mutated_hash = space.config_hash(mutated)
        assert mutated_hash != base_hash
        assert mutated["algorithm"] == "kmeans"  # algorithm mutation off by default
        mutated_configs.append(mutated)

    # Test algorithm mutation enabled
    alg_mutated = space.mutate(base_config, mutation_rate=1.0, allow_algorithm_mutation=True, random_state=99)
    assert alg_mutated["algorithm"] in ALGORITHM_CHOICES


def test_search_space_neighbor_generation():
    space = SearchSpace()
    base_config = space.get_default_configuration("kmeans")
    neighbors = space.get_neighbors(base_config, n_neighbors=5, random_state=42)

    assert len(neighbors) == 5
    hashes = [space.config_hash(n) for n in neighbors]
    # All neighbors should be unique and distinct from base
    assert len(set(hashes)) == 5
    assert space.config_hash(base_config) not in set(hashes)


def test_search_space_hashing_determinism():
    space = SearchSpace()
    config_a = {
        "preprocessing": {"scaler": "standard", "imputer": "median", "outlier_handler": "iqr", "pca_components": None, "feature_engineering": True},
        "algorithm": "kmeans",
        "hyperparameters": {"n_clusters": 4, "init": "k-means++", "max_iter": 300},
    }
    config_b = {
        "hyperparameters": {"max_iter": 300, "n_clusters": 4, "init": "k-means++"},
        "algorithm": "kmeans",
        "preprocessing": {"feature_engineering": True, "outlier_handler": "iqr", "imputer": "median", "scaler": "standard", "pca_components": None},
    }
    # Keys in different order should produce identical hash
    hash_a = space.config_hash(config_a)
    hash_b = space.config_hash(config_b)
    assert hash_a == hash_b

    # Modifying a value should change hash
    config_c = copy_dict = dict(config_a)
    config_c["algorithm"] = "gmm"
    assert space.config_hash(config_c) != hash_a


def test_search_space_flatten_and_unflatten():
    space = SearchSpace()
    orig = space.get_default_configuration("kmeans")
    flat = space.flatten_config(orig)
    assert "prep_scaler" in flat
    assert "hyp_n_clusters" in flat
    assert flat["algorithm"] == "kmeans"

    unflattened = space.unflatten_config(flat)
    assert unflattened["preprocessing"]["scaler"] == orig["preprocessing"]["scaler"]
    assert unflattened["hyperparameters"]["n_clusters"] == orig["hyperparameters"]["n_clusters"]
    assert unflattened["algorithm"] == orig["algorithm"]


def test_search_space_validation_and_bounds_clamping():
    space = SearchSpace()
    invalid_config = {
        "preprocessing": {"imputer": "invalid_strat", "scaler": "bad_scaler", "outlier_handler": "unknown", "pca_components": 999, "feature_engineering": True},
        "algorithm": "kmeans",
        "hyperparameters": {"n_clusters": 99, "init": "bad_init", "max_iter": -100},
    }
    validated = space.validate_configuration(invalid_config)
    assert validated["preprocessing"]["imputer"] == "median"
    assert validated["preprocessing"]["scaler"] == "yeo_johnson"
    assert validated["preprocessing"]["pca_components"] is None
    assert validated["hyperparameters"]["n_clusters"] == 12  # clamped to high bound
    assert validated["hyperparameters"]["init"] == "k-means++"  # reset to default choice
    assert validated["hyperparameters"]["max_iter"] == 100  # clamped to low bound


# ==============================================================================
# 2. Composite Objective Unit Tests
# ==============================================================================

def test_objective_weights_normalization():
    weights = ObjectiveWeights(silhouette=0.8, davies_bouldin=0.4, calinski_harabasz=0.4, stability=0.4)
    assert pytest.approx(weights.silhouette + weights.davies_bouldin + weights.calinski_harabasz + weights.stability, 1e-4) == 1.0
    assert pytest.approx(weights.silhouette, 1e-4) == 0.40
    assert pytest.approx(weights.davies_bouldin, 1e-4) == 0.20


def test_objective_fitness_calculation_formula():
    dataset = np.random.default_rng(42).normal(size=(50, 4))
    obj = CompositeObjective(dataset=dataset, stability_bootstraps=1)

    labels = np.array([0] * 25 + [1] * 25)
    fitness, norm_metrics, penalties = obj.compute_fitness(
        silhouette=0.6,
        davies_bouldin=1.0,
        calinski_harabasz=500.0,
        stability_ari=0.9,
        labels=labels,
    )

    assert 0.0 <= fitness <= 1.0
    assert norm_metrics["s_norm"] == pytest.approx((0.6 + 1.0) / 2.0, 1e-3)
    assert norm_metrics["db_norm"] == pytest.approx(1.0 - (1.0 / 5.0), 1e-3)
    assert penalties["p_noise"] == 0.0  # 0 noise
    assert penalties["p_imbalance"] == pytest.approx(0.0, 1e-2)  # perfectly balanced 25/25


def test_objective_evaluate_kmeans(synthetic_df: pd.DataFrame):
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=2, random_state=42)
    space = SearchSpace()
    config = space.get_default_configuration("kmeans")

    res = obj.evaluate(config)
    assert isinstance(res, ObjectiveEvaluation)
    assert 0.0 <= res.fitness <= 1.0
    assert res.n_clusters == 4
    assert res.silhouette > -1.0
    assert res.davies_bouldin > 0.0
    assert res.calinski_harabasz > 0.0
    assert 0.0 <= res.stability_ari <= 1.0
    assert res.execution_time_ms > 0.0
    assert res.error is None


@pytest.mark.parametrize("algorithm", ALGORITHM_CHOICES)
def test_objective_evaluate_all_algorithms(synthetic_df: pd.DataFrame, algorithm: str):
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
    space = SearchSpace()
    config = space.get_default_configuration(algorithm)

    res = obj.evaluate(config)
    assert isinstance(res, ObjectiveEvaluation)
    assert res.execution_time_ms > 0
    if res.error is None:
        assert res.fitness >= 0.0
        assert res.n_clusters >= 2
    else:
        assert res.fitness == -1.0


def test_objective_evaluate_pca_reduction(synthetic_df: pd.DataFrame):
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
    space = SearchSpace()
    config = space.get_default_configuration("kmeans")
    config["preprocessing"]["pca_components"] = 3

    res = obj.evaluate(config)
    assert res.error is None
    assert res.fitness >= 0.0


def test_objective_edge_case_degenerate_noise(synthetic_df: pd.DataFrame):
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
    # DBSCAN with tiny eps and huge min_samples forces 100% noise
    degenerate_config = {
        "preprocessing": {"imputer": "median", "outlier_handler": "none", "scaler": "standard", "pca_components": None, "feature_engineering": False},
        "algorithm": "dbscan",
        "hyperparameters": {"eps": 0.001, "min_samples": 100},
    }
    res = obj.evaluate(degenerate_config)
    assert res.fitness == -1.0
    assert res.n_clusters < 2
    assert res.error is not None


def test_objective_caching_mechanism(synthetic_df: pd.DataFrame):
    obj = CompositeObjective(dataset=synthetic_df, cache_preprocessing=True)
    space = SearchSpace()
    config_a = space.get_default_configuration("kmeans")
    config_b = space.get_default_configuration("kmeans")
    config_b["hyperparameters"]["n_clusters"] = 5

    # First call fills cache
    _ = obj.evaluate(config_a)
    cache_len_1 = len(obj._prep_cache)
    assert cache_len_1 >= 1

    # Second call with same preprocessing config uses cache
    _ = obj.evaluate(config_b)
    cache_len_2 = len(obj._prep_cache)
    assert cache_len_1 == cache_len_2


# ==============================================================================
# 3. Hill-Climbing Optimizer Unit Tests
# ==============================================================================

def test_hill_climber_initialization(synthetic_df: pd.DataFrame):
    space = SearchSpace()
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
    optimizer = HillClimbingOptimizer(
        search_space=space,
        objective=obj,
        target_algorithm="kmeans",
        initial_temperature=0.5,
        patience=5,
        random_state=42,
    )

    assert optimizer.step_index == 0
    assert optimizer.restart_count == 0
    assert optimizer.current_theta["algorithm"] == "kmeans"
    assert optimizer.temperature == 0.5

    init_eval = optimizer.initialize()
    assert init_eval.fitness >= 0.0
    assert optimizer.is_initialized is True


def test_hill_climber_single_step(synthetic_df: pd.DataFrame):
    space = SearchSpace()
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
    optimizer = HillClimbingOptimizer(
        search_space=space,
        objective=obj,
        target_algorithm="kmeans",
        random_state=42,
    )

    step_log = optimizer.step()
    assert isinstance(step_log, StepLog)
    assert step_log.step == 1
    assert step_log.algorithm == "kmeans"
    assert step_log.temperature < 0.5  # temperature cooled
    assert len(optimizer.history) == 1
    assert len(optimizer.tabu_history) >= 2  # initial + 1 step


def test_hill_climber_plateau_random_restart(synthetic_df: pd.DataFrame):
    space = SearchSpace()
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
    # Set patience=2 to quickly trigger restart
    optimizer = HillClimbingOptimizer(
        search_space=space,
        objective=obj,
        target_algorithm="kmeans",
        patience=2,
        initial_temperature=0.001,  # low temp rejects inferior mutations
        random_state=42,
    )
    optimizer.initialize()
    # Artificially set best fitness super high so normal mutations stagnate
    optimizer.best_eval.fitness = 1.0

    step_1 = optimizer.step()
    assert step_1.restart is False
    assert optimizer.stagnation_count == 1

    step_2 = optimizer.step()
    assert step_2.restart is False
    assert optimizer.stagnation_count == 2

    # Step 3 exceeds patience and executes random restart
    step_3 = optimizer.step()
    assert step_3.restart is True
    assert optimizer.restart_count == 1
    assert optimizer.stagnation_count == 0


def test_hill_climber_run_synchronous(synthetic_df: pd.DataFrame):
    space = SearchSpace()
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
    optimizer = HillClimbingOptimizer(
        search_space=space,
        objective=obj,
        target_algorithm="kmeans",
        patience=4,
        random_state=42,
    )

    callback_logs = []
    result = optimizer.run(max_steps=5, callback=lambda log: callback_logs.append(log))

    assert isinstance(result, OptimizationResult)
    assert result.total_steps == 5
    assert len(callback_logs) == 5
    assert result.best_fitness >= result.initial_fitness
    assert result.best_eval is not None
    assert result.total_time_ms > 0


def test_hill_climber_async_run_stream(synthetic_df: pd.DataFrame):
    async def _run():
        space = SearchSpace()
        obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
        optimizer = HillClimbingOptimizer(
            search_space=space,
            objective=obj,
            target_algorithm="kmeans",
            random_state=42,
        )

        streamed_steps = []
        async for step_log in optimizer.run_stream(max_steps=3, delay_seconds=0.0):
            assert isinstance(step_log, StepLog)
            streamed_steps.append(step_log)

        assert len(streamed_steps) == 3
        assert streamed_steps[0].step == 1
        assert streamed_steps[2].step == 3

    asyncio.run(_run())


# ==============================================================================
# 4. Experiment Logger & Ablation Analysis Unit Tests
# ==============================================================================

def test_experiment_logger_telemetry(synthetic_df: pd.DataFrame):
    space = SearchSpace()
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
    optimizer = HillClimbingOptimizer(search_space=space, objective=obj, random_state=42)
    logger = ExperimentLogger(run_id="test_run_01")

    res = optimizer.run(max_steps=4)
    for log in res.history:
        logger.log_step(log)

    assert len(logger.logs) == 4
    summary = logger.get_summary()
    assert summary["run_id"] == "test_run_01"
    assert summary["total_steps"] == 4
    assert summary["best_fitness"] >= summary["initial_fitness"]

    leaderboard = logger.get_leaderboard(top_n=3)
    assert len(leaderboard) <= 3
    assert leaderboard[0]["fitness"] >= leaderboard[-1]["fitness"]

    trajectory = logger.get_best_trajectory()
    assert len(trajectory) >= 1
    # Monotonically non-decreasing
    for i in range(1, len(trajectory)):
        assert trajectory[i]["fitness"] >= trajectory[i - 1]["fitness"]


def test_experiment_logger_save_jsonl_and_csv(synthetic_df: pd.DataFrame):
    space = SearchSpace()
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
    optimizer = HillClimbingOptimizer(search_space=space, objective=obj, random_state=42)
    logger = ExperimentLogger(run_id="test_io_run")

    res = optimizer.run(max_steps=3)
    for log in res.history:
        logger.log_step(log)

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "experiment.jsonl"
        csv_path = Path(tmpdir) / "experiment.csv"

        logger.save_jsonl(jsonl_path)
        logger.save_csv(csv_path)

        assert jsonl_path.exists()
        assert csv_path.exists()

        # Check JSONL rows
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 3

        # Check CSV rows
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == 3
        assert "prep_scaler" in df_csv.columns
        assert "fitness" in df_csv.columns


def test_export_ablation_analysis(synthetic_df: pd.DataFrame):
    space = SearchSpace()
    obj = CompositeObjective(dataset=synthetic_df, stability_bootstraps=1, random_state=42)
    optimizer = HillClimbingOptimizer(search_space=space, objective=obj, random_state=42)
    res = optimizer.run(max_steps=6)

    ablation = export_ablation_analysis(res.history)
    assert "stage_breakdown" in ablation
    assert "parameter_importance" in ablation
    assert "comparison" in ablation
    assert ablation["total_evaluated_steps"] >= 1

    comparison = ablation["comparison"]
    assert "delta_fitness" in comparison
    assert "baseline_fitness" in comparison
    assert "best_fitness" in comparison


# ==============================================================================
# 5. End-to-End Autoresearch Optimization Verification
# ==============================================================================

def test_end_to_end_autoresearch_optimization_improvement(synthetic_df: pd.DataFrame):
    """
    Verifies that running hill climbing searches configurations, discovers valid parameter
    improvements, logs structured trajectories, and generates complete ablation reports.
    """
    space = SearchSpace()
    obj = CompositeObjective(
        dataset=synthetic_df,
        weights=ObjectiveWeights(silhouette=0.45, davies_bouldin=0.25, calinski_harabasz=0.15, stability=0.15),
        stability_bootstraps=1,
        random_state=42,
    )

    optimizer = HillClimbingOptimizer(
        search_space=space,
        objective=obj,
        target_algorithm="kmeans",
        initial_temperature=0.4,
        cooling_rate=0.90,
        patience=4,
        random_state=42,
    )

    logger = ExperimentLogger(run_id="e2e_autoresearch_test")

    # Run 8 steps of hill-climbing search
    result = optimizer.run(max_steps=8, callback=logger.log_step)

    # 1. State integrity
    assert result.total_steps == 8
    assert result.best_fitness >= result.initial_fitness
    assert result.best_eval.n_clusters >= 2
    assert result.best_eval.silhouette > -1.0

    # 2. Telemetry validation
    summary = logger.get_summary()
    assert summary["total_steps"] == 8
    assert summary["acceptance_rate"] >= 0.0

    # 3. Ablation report validation
    ablation = export_ablation_analysis(logger.logs)
    assert ablation["total_evaluated_steps"] == 8
    assert len(ablation["stage_breakdown"]) >= 3
    assert len(ablation["parameter_importance"]) >= 1
