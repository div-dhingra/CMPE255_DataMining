"""Autonomous hill-climbing optimization engine with simulated annealing and random restarts."""

from __future__ import annotations
import time
import math
import random
from typing import Dict, Any, Optional, List, Callable, Set

from .space import SearchSpace, MiningConfiguration
from .fitness import CompositeFitnessEvaluator, FitnessBreakdown
from .ledger import ExperimentLedger, StepLog
from ..core.dataset import TransactionDataset
from ..core.rules import RuleGenerator, AssociationRule
from ..algorithms.apriori import AprioriMiner
from ..algorithms.fpgrowth import FPGrowthMiner
from ..algorithms.eclat import EclatMiner
from ..algorithms.base import MiningResult


class HillClimbingOptimizer:
    """
    Stochastic Hill-Climber with Simulated Annealing acceptance,
    Tabu hash cache, stagnation detection, and random restarts.
    """

    def __init__(
        self,
        dataset: TransactionDataset,
        initial_config: Optional[MiningConfiguration] = None,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.94,
        max_stagnation: int = 5,
        fitness_evaluator: Optional[CompositeFitnessEvaluator] = None
    ):
        self.dataset = dataset
        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_stagnation = max_stagnation
        self.evaluator = fitness_evaluator or CompositeFitnessEvaluator()

        self.current_config: MiningConfiguration = initial_config or SearchSpace.sample_default()
        self.current_fitness: float = 0.0
        self.current_breakdown: Optional[FitnessBreakdown] = None

        self.best_config: MiningConfiguration = self.current_config
        self.best_fitness: float = 0.0
        self.best_breakdown: Optional[FitnessBreakdown] = None

        self.ledger = ExperimentLedger()
        self.tabu_hashes: Set[str] = set()
        self.stagnation_count: int = 0
        self.iteration: int = 0
        self.restarts_count: int = 0

        # Miners instances
        self.miners = {
            "apriori": AprioriMiner(),
            "fpgrowth": FPGrowthMiner(),
            "eclat": EclatMiner()
        }

        # Initialize baseline evaluation
        self._evaluate_initial()

    def _evaluate_initial(self) -> None:
        """Evaluates baseline initial configuration."""
        breakdown, rules, _ = self._evaluate_config(self.current_config)
        self.current_fitness = breakdown.composite_fitness
        self.current_breakdown = breakdown
        self.best_fitness = breakdown.composite_fitness
        self.best_breakdown = breakdown
        self.tabu_hashes.add(self.current_config.get_hash())

        initial_log = StepLog(
            iteration=0,
            timestamp=time.time(),
            action="INITIAL_BASELINE",
            configuration=self.current_config.to_dict(),
            candidate_fitness=self.current_fitness,
            current_fitness=self.current_fitness,
            best_fitness=self.best_fitness,
            fitness_delta=0.0,
            accepted=True,
            temperature=self.temperature,
            execution_time_ms=breakdown.execution_time_ms,
            rule_count=breakdown.rule_count,
            mean_lift=breakdown.mean_lift,
            mean_kulczynski=breakdown.mean_kulczynski,
            coverage_score=breakdown.coverage_score,
            diversity_score=breakdown.diversity_score,
            details="Initial configuration evaluated as baseline."
        )
        self.ledger.record_step(initial_log)

    def _evaluate_config(self, cfg: MiningConfiguration) -> Tuple[FitnessBreakdown, List[AssociationRule], MiningResult]:
        """Runs the chosen mining algorithm and scores rules with composite fitness."""
        miner = self.miners.get(cfg.algorithm, self.miners["fpgrowth"])

        # Execute mining
        t0 = time.perf_counter()
        mining_result = miner.mine(
            transactions=self.dataset.transactions,
            min_support=cfg.min_support,
            max_len=cfg.max_itemset_length
        )

        # Generate rules
        rules = RuleGenerator.generate_rules(
            frequent_itemsets=mining_result.itemsets,
            min_confidence=cfg.min_confidence,
            min_lift=cfg.min_lift,
            max_consequent_len=1,
            prune_redundant=True
        )

        exec_time_ms = (time.perf_counter() - t0) * 1000.0

        # Raw candidate rules count (before redundancy pruning) for diversity evaluation
        raw_rules = RuleGenerator.generate_rules(
            frequent_itemsets=mining_result.itemsets,
            min_confidence=cfg.min_confidence,
            min_lift=cfg.min_lift,
            max_consequent_len=1,
            prune_redundant=False
        )
        raw_count = len(raw_rules)

        breakdown = self.evaluator.evaluate(
            rules=rules,
            frequent_itemsets=mining_result.itemsets,
            total_unique_items=self.dataset.num_unique_items,
            execution_time_ms=exec_time_ms,
            raw_candidate_rules_count=raw_count
        )

        return breakdown, rules, mining_result

    def step(self) -> StepLog:
        """Executes one autonomous hill-climbing search iteration."""
        self.iteration += 1
        is_restart = False

        # Check for stagnation
        if self.stagnation_count >= self.max_stagnation:
            # Trigger Random Restart
            candidate = SearchSpace.sample_random()
            is_restart = True
            self.restarts_count += 1
            self.stagnation_count = 0
            # Reset temperature upon restart
            self.temperature = self.initial_temperature * 0.8
            action_type = "RESTART"
            details = f"Stagnation threshold reached ({self.max_stagnation} steps). Performed global random restart."
        else:
            # Targeted neighborhood perturbation
            candidate = SearchSpace.perturb(self.current_config, temperature=self.temperature)
            # Try to avoid immediate tabu duplicate
            retries = 0
            while candidate.get_hash() in self.tabu_hashes and retries < 3:
                candidate = SearchSpace.perturb(self.current_config, temperature=self.temperature)
                retries += 1
            action_type = "STEP"
            details = "Neighborhood perturbation executed."

        # Add to tabu cache
        self.tabu_hashes.add(candidate.get_hash())

        # Evaluate candidate
        breakdown, rules, _ = self._evaluate_config(candidate)
        cand_fitness = breakdown.composite_fitness
        delta = cand_fitness - self.current_fitness

        # Simulated Annealing acceptance rule
        accepted = False
        if delta > 0:
            accepted = True
            details = f"Improved fitness by +{round(delta, 4)}. Configuration accepted."
        else:
            # Acceptance probability for worse configuration
            prob = math.exp(delta / max(1e-4, self.temperature))
            if random.random() < prob:
                accepted = True
                details = f"Stochastic annealing accepted candidate (P={round(prob, 3)}, delta={round(delta, 4)})."
            else:
                accepted = False
                details = f"Candidate rejected (delta={round(delta, 4)})."

        if accepted:
            self.current_config = candidate
            self.current_fitness = cand_fitness
            self.current_breakdown = breakdown
            if not is_restart:
                action_type = "ACCEPT"

        # Check global best improvement
        improved_global = False
        if cand_fitness > self.best_fitness:
            self.best_fitness = cand_fitness
            self.best_config = candidate
            self.best_breakdown = breakdown
            self.stagnation_count = 0
            improved_global = True
            action_type = "IMPROVE_GLOBAL_BEST"
            details = f"New global optimum discovered! Fitness = {round(self.best_fitness, 4)}."
        else:
            self.stagnation_count += 1

        # Cool temperature
        self.temperature = max(0.04, self.temperature * self.cooling_rate)

        log = StepLog(
            iteration=self.iteration,
            timestamp=time.time(),
            action=action_type,
            configuration=candidate.to_dict(),
            candidate_fitness=cand_fitness,
            current_fitness=self.current_fitness,
            best_fitness=self.best_fitness,
            fitness_delta=delta,
            accepted=accepted,
            temperature=self.temperature,
            execution_time_ms=breakdown.execution_time_ms,
            rule_count=breakdown.rule_count,
            mean_lift=breakdown.mean_lift,
            mean_kulczynski=breakdown.mean_kulczynski,
            coverage_score=breakdown.coverage_score,
            diversity_score=breakdown.diversity_score,
            details=details
        )
        self.ledger.record_step(log)
        return log

    def run(self, max_steps: int = 15, callback: Optional[Callable[[StepLog], None]] = None) -> Dict[str, Any]:
        """Executes a series of autonomous iterations."""
        logs: List[StepLog] = []
        for _ in range(max_steps):
            log = self.step()
            logs.append(log)
            if callback:
                callback(log)

        return {
            "completed_iterations": max_steps,
            "total_iterations": self.iteration,
            "best_fitness": round(self.best_fitness, 4),
            "best_configuration": self.best_config.to_dict(),
            "best_breakdown": self.best_breakdown.to_dict() if self.best_breakdown else {},
            "restarts_count": self.restarts_count,
            "leaderboard": self.ledger.get_leaderboard(top_n=5),
            "recent_logs": [log.to_dict() for log in logs]
        }
