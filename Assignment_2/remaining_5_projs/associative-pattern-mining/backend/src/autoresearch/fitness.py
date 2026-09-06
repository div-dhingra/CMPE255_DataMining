"""Multi-objective composite fitness evaluation for association pattern mining configurations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Set
import numpy as np
from ..core.rules import AssociationRule
from ..algorithms.base import FrequentItemset


@dataclass(frozen=True)
class FitnessBreakdown:
    """Detailed score decomposition across all objective axes."""
    composite_fitness: float
    coverage_score: float
    interestingness_score: float
    diversity_score: float
    runtime_penalty: float
    volume_penalty: float
    rule_count: int
    unique_items_covered: int
    mean_lift: float
    mean_kulczynski: float
    execution_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_fitness": round(self.composite_fitness, 4),
            "coverage_score": round(self.coverage_score, 4),
            "interestingness_score": round(self.interestingness_score, 4),
            "diversity_score": round(self.diversity_score, 4),
            "runtime_penalty": round(self.runtime_penalty, 4),
            "volume_penalty": round(self.volume_penalty, 4),
            "rule_count": self.rule_count,
            "unique_items_covered": self.unique_items_covered,
            "mean_lift": round(self.mean_lift, 3),
            "mean_kulczynski": round(self.mean_kulczynski, 3),
            "execution_time_ms": round(self.execution_time_ms, 2)
        }


class CompositeFitnessEvaluator:
    """
    Evaluates hyperparameter fitness balancing:
    1. Coverage: Percentage of catalog items represented in rules
    2. Interestingness: Mean normalized Lift, Conviction, and Kulczynski
    3. Diversity: Ratio of non-redundant, distinct antecedent patterns
    4. Efficiency: Latency penalty for slow candidate explosions
    5. Viability: Penalties for rule starvation (< 5) or excessive explosion (> 300)
    """

    def __init__(
        self,
        weight_coverage: float = 0.30,
        weight_interestingness: float = 0.35,
        weight_diversity: float = 0.25,
        weight_runtime: float = 0.10,
        target_runtime_ms: float = 120.0
    ):
        self.w_cov = weight_coverage
        self.w_int = weight_interestingness
        self.w_div = weight_diversity
        self.w_run = weight_runtime
        self.target_runtime_ms = target_runtime_ms

    def evaluate(
        self,
        rules: List[AssociationRule],
        frequent_itemsets: List[FrequentItemset],
        total_unique_items: int,
        execution_time_ms: float,
        raw_candidate_rules_count: int
    ) -> FitnessBreakdown:
        n_rules = len(rules)

        # 1. Rule Starvation / Volume Penalty
        if n_rules == 0:
            return FitnessBreakdown(
                composite_fitness=0.0,
                coverage_score=0.0,
                interestingness_score=0.0,
                diversity_score=0.0,
                runtime_penalty=0.0,
                volume_penalty=0.5,
                rule_count=0,
                unique_items_covered=0,
                mean_lift=0.0,
                mean_kulczynski=0.0,
                execution_time_ms=execution_time_ms
            )

        volume_penalty = 0.0
        if n_rules < 5:
            volume_penalty += 0.35 * (5 - n_rules) / 5.0
        elif n_rules > 350:
            volume_penalty += min(0.30, (n_rules - 350) / 1000.0)

        # 2. Coverage Score: distinct items in rules / total available items
        items_in_rules: Set[str] = set()
        for r in rules:
            items_in_rules.update(r.antecedent)
            items_in_rules.update(r.consequent)

        coverage = len(items_in_rules) / max(1, total_unique_items)
        coverage_score = min(1.0, max(0.0, coverage))

        # 3. Interestingness Score
        lifts = [r.metrics.lift for r in rules]
        kulcs = [r.metrics.kulczynski for r in rules]
        convs = [min(10.0, r.metrics.conviction) for r in rules]

        mean_lift = float(np.mean(lifts)) if lifts else 1.0
        mean_kulc = float(np.mean(kulcs)) if kulcs else 0.5
        mean_conv = float(np.mean(convs)) if convs else 1.0

        # Normalize components into [0, 1]
        norm_lift = min(1.0, max(0.0, (mean_lift - 1.0) / 4.0))
        norm_kulc = min(1.0, max(0.0, (mean_kulc - 0.4) / 0.6))
        norm_conv = min(1.0, max(0.0, (mean_conv - 1.0) / 5.0))

        interestingness_score = (0.45 * norm_lift) + (0.35 * norm_kulc) + (0.20 * norm_conv)

        # 4. Diversity Score
        # Measures ratio of non-redundant rules retained from candidate generation
        if raw_candidate_rules_count > 0:
            retention_ratio = n_rules / raw_candidate_rules_count
            # Sweet spot is high retention of diverse antecedents
            diversity_score = min(1.0, max(0.2, retention_ratio))
        else:
            diversity_score = 0.5

        # 5. Runtime Penalty
        if execution_time_ms > self.target_runtime_ms:
            runtime_penalty = min(0.30, (execution_time_ms - self.target_runtime_ms) / 1000.0)
        else:
            runtime_penalty = 0.0

        # Composite Fitness Calculation
        raw_fitness = (
            (self.w_cov * coverage_score) +
            (self.w_int * interestingness_score) +
            (self.w_div * diversity_score) -
            (self.w_run * runtime_penalty) -
            volume_penalty
        )

        final_fitness = max(0.0, min(1.0, raw_fitness))

        return FitnessBreakdown(
            composite_fitness=final_fitness,
            coverage_score=coverage_score,
            interestingness_score=interestingness_score,
            diversity_score=diversity_score,
            runtime_penalty=runtime_penalty,
            volume_penalty=volume_penalty,
            rule_count=n_rules,
            unique_items_covered=len(items_in_rules),
            mean_lift=mean_lift,
            mean_kulczynski=mean_kulc,
            execution_time_ms=execution_time_ms
        )
