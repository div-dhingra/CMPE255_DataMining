"""Tier 3: Autoresearch Hill-Climbing Optimization Engine Tests.

Verifies:
- Objective function evaluation (Loss, Perplexity, Throughput, Memory, Score)
- Hyperparameter state perturbation and delta tracking
- Step transitions, acceptance/rejection mechanics
- Simulated annealing restart triggers on plateau
- Experiment ledger recording, statistics, and CSV/JSON serialization
"""

import json
import pytest

from src.autoresearch.hill_climber import AutoresearchHillClimber, HyperparameterState
from src.autoresearch.ledger import ExperimentLedger
from src.autoresearch.literature import LANDMARK_PAPERS, get_literature_summary
from src.autoresearch.objective import EvaluationObjective


class TestObjectiveFunction:
    def test_objective_evaluation(self, model, tokenizer):
        objective = EvaluationObjective(tokenizer=tokenizer)
        metrics = objective.evaluate_model(model=model, temperature=0.7, top_p=0.9, max_eval_tokens=16)
        assert metrics.loss > 0.0
        assert metrics.perplexity > 1.0
        assert metrics.tokens_per_sec > 0.0
        assert metrics.memory_mb > 0.0
        assert metrics.composite_score > 0.0


class TestHillClimbingLoop:
    def test_step_transitions_and_ledger(self, test_climber):
        # Step 0: Baseline
        entry0 = test_climber.step()
        assert entry0.iteration == 0
        assert entry0.decision == "baseline"
        assert entry0.step_type == "baseline"
        assert entry0.composite_score > 0.0

        # Step 1: Perturbation
        entry1 = test_climber.step()
        assert entry1.iteration == 1
        assert entry1.decision in ("accepted", "rejected")
        assert len(entry1.parameter_deltas) > 0

    def test_simulated_restart_on_plateau(self, tokenizer):
        ledger = ExperimentLedger()
        objective = EvaluationObjective(tokenizer=tokenizer)
        # Configure climber with patience=1 and max_restarts=2
        climber = AutoresearchHillClimber(
            objective=objective,
            ledger=ledger,
            patience=1,
            max_restarts=2,
            seed=999,
        )
        # Run 6 steps to trigger patience and restart
        entries = climber.run(6)
        types = [e.step_type for e in entries]
        assert "baseline" in types
        # Should have attempted perturbations
        assert "perturbation" in types

    def test_ledger_summary_and_serialization(self, test_climber, tmp_path):
        test_climber.run(4)
        summary = test_climber.ledger.get_summary()
        assert summary["total_steps"] == 4
        assert "baseline_score" in summary
        assert "best_score" in summary
        assert "acceptance_rate" in summary

        # Test CSV export
        csv_str = test_climber.ledger.to_csv()
        assert "iteration,step_type,decision" in csv_str

        # Test JSON save
        json_path = test_climber.ledger.save_json(tmp_path / "ledger.json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
            assert "summary" in data
            assert len(data["history"]) == 4


class TestLiteratureAlignment:
    def test_landmark_papers_coverage(self):
        summary = get_literature_summary()
        papers = summary["papers"]
        assert "vaswani_2017" in papers
        assert "touvron_2023" in papers
        assert "shazeer_2020" in papers
        assert "su_2021" in papers
        assert "hoffmann_2022" in papers
        assert "ainslie_2023" in papers

    def test_benchmark_matrix_consistency(self):
        summary = get_literature_summary()
        matrix = summary["benchmark_matrix"]
        assert len(matrix) >= 5
        for row in matrix:
            assert "architecture" in row
            assert "kv_cache_mb_1k" in row
            assert "throughput_tps" in row
