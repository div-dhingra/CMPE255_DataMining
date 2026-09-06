"""Tests for autoresearch parameter search space, fitness evaluation, and hill-climbing."""

from backend.src.autoresearch.space import SearchSpace, MiningConfiguration
from backend.src.autoresearch.fitness import CompositeFitnessEvaluator
from backend.src.autoresearch.optimizer import HillClimbingOptimizer
from backend.src.core.dataset import SyntheticTransactionGenerator


def test_search_space_sampling():
    cfg = SearchSpace.sample_random()
    assert SearchSpace.MIN_SUPPORT_BOUNDS[0] <= cfg.min_support <= SearchSpace.MIN_SUPPORT_BOUNDS[1]
    assert SearchSpace.MIN_CONFIDENCE_BOUNDS[0] <= cfg.min_confidence <= SearchSpace.MIN_CONFIDENCE_BOUNDS[1]
    assert cfg.algorithm in SearchSpace.ALGORITHMS
    assert 2 <= cfg.max_itemset_length <= 5


def test_search_space_perturbation():
    initial = SearchSpace.sample_default()
    perturbed = SearchSpace.perturb(initial, temperature=0.5)

    assert isinstance(perturbed, MiningConfiguration)
    assert SearchSpace.MIN_SUPPORT_BOUNDS[0] <= perturbed.min_support <= SearchSpace.MIN_SUPPORT_BOUNDS[1]
    assert SearchSpace.MIN_CONFIDENCE_BOUNDS[0] <= perturbed.min_confidence <= SearchSpace.MIN_CONFIDENCE_BOUNDS[1]


def test_composite_fitness_empty_rules():
    evaluator = CompositeFitnessEvaluator()
    breakdown = evaluator.evaluate(
        rules=[],
        frequent_itemsets=[],
        total_unique_items=20,
        execution_time_ms=10.0,
        raw_candidate_rules_count=0
    )
    assert breakdown.composite_fitness == 0.0
    assert breakdown.rule_count == 0


def test_hill_climbing_optimizer_execution():
    ds = SyntheticTransactionGenerator.generate(num_transactions=200, seed=99)
    optimizer = HillClimbingOptimizer(
        dataset=ds,
        cooling_rate=0.90,
        max_stagnation=3
    )

    baseline_fit = optimizer.best_fitness
    assert 0.0 <= baseline_fit <= 1.0

    # Execute 6 steps
    run_res = optimizer.run(max_steps=6)

    assert run_res["completed_iterations"] == 6
    assert optimizer.iteration == 6
    assert optimizer.best_fitness >= baseline_fit
    assert len(optimizer.ledger.logs) == 7  # 1 initial baseline + 6 steps

    trajectory = optimizer.ledger.get_trajectory()
    assert len(trajectory["iterations"]) == 7
    assert len(trajectory["best_fitness"]) == 7
    assert len(trajectory["temperatures"]) == 7


def test_stagnation_restart_trigger():
    ds = SyntheticTransactionGenerator.generate(num_transactions=150, seed=55)
    optimizer = HillClimbingOptimizer(dataset=ds, max_stagnation=2)

    # Set best_fitness artificially high to simulate stagnation plateau
    optimizer.best_fitness = 1.0
    optimizer.stagnation_count = 2

    # Next step must trigger random restart
    log = optimizer.step()
    assert log.action == "RESTART"
    assert optimizer.restarts_count >= 1
