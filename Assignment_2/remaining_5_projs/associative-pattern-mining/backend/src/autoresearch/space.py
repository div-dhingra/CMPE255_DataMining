"""Search space configuration and mutation operators for associative pattern autoresearch."""

from __future__ import annotations
from dataclasses import dataclass
import random
import math
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class MiningConfiguration:
    """Hyperparameter configuration vector theta for association pattern mining."""
    min_support: float       # In [0.005, 0.25]
    min_confidence: float    # In [0.10, 0.95]
    max_itemset_length: int  # In [2, 5]
    algorithm: str           # "apriori", "fpgrowth", "eclat"
    min_lift: float          # In [1.0, 5.0]
    top_k: int               # In [10, 100]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_support": round(self.min_support, 4),
            "min_confidence": round(self.min_confidence, 4),
            "max_itemset_length": self.max_itemset_length,
            "algorithm": self.algorithm,
            "min_lift": round(self.min_lift, 4),
            "top_k": self.top_k
        }

    def get_hash(self) -> str:
        """Coarse-grained hash string for Tabu cache matching."""
        supp_bucket = round(self.min_support, 3)
        conf_bucket = round(self.min_confidence, 2)
        lift_bucket = round(self.min_lift, 2)
        return f"{self.algorithm}|{supp_bucket}|{conf_bucket}|{self.max_itemset_length}|{lift_bucket}"


class SearchSpace:
    """Defines bounds, random sampling, and perturbation operators for MiningConfiguration."""

    ALGORITHMS = ["apriori", "fpgrowth", "eclat"]

    MIN_SUPPORT_BOUNDS = (0.01, 0.20)
    MIN_CONFIDENCE_BOUNDS = (0.15, 0.90)
    MAX_LENGTH_BOUNDS = (2, 5)
    MIN_LIFT_BOUNDS = (1.05, 4.0)
    TOP_K_BOUNDS = (10, 100)

    @classmethod
    def sample_random(cls) -> MiningConfiguration:
        """Samples a uniform random configuration from the search space."""
        return MiningConfiguration(
            min_support=round(random.uniform(*cls.MIN_SUPPORT_BOUNDS), 4),
            min_confidence=round(random.uniform(*cls.MIN_CONFIDENCE_BOUNDS), 4),
            max_itemset_length=random.randint(*cls.MAX_LENGTH_BOUNDS),
            algorithm=random.choice(cls.ALGORITHMS),
            min_lift=round(random.uniform(*cls.MIN_LIFT_BOUNDS), 2),
            top_k=random.choice([20, 30, 50, 75, 100])
        )

    @classmethod
    def sample_default(cls) -> MiningConfiguration:
        """Returns standard canonical baseline configuration."""
        return MiningConfiguration(
            min_support=0.03,
            min_confidence=0.40,
            max_itemset_length=3,
            algorithm="fpgrowth",
            min_lift=1.20,
            top_k=50
        )

    @classmethod
    def perturb(cls, current: MiningConfiguration, temperature: float = 1.0) -> MiningConfiguration:
        """
        Perturbs the current configuration state to produce a neighbor configuration.
        Mutation step sizes scale proportionally with the cooling temperature.
        """
        # Decide which parameter(s) to mutate
        mutate_supp = random.random() < 0.60
        mutate_conf = random.random() < 0.60
        mutate_alg = random.random() < 0.25
        mutate_len = random.random() < 0.30
        mutate_lift = random.random() < 0.40

        # Guarantee at least one mutation
        if not any([mutate_supp, mutate_conf, mutate_alg, mutate_len, mutate_lift]):
            mutate_supp = True

        new_supp = current.min_support
        if mutate_supp:
            # Log-normal or Gaussian perturbation
            step = random.gauss(0, 0.015 * temperature)
            new_supp = max(cls.MIN_SUPPORT_BOUNDS[0], min(cls.MIN_SUPPORT_BOUNDS[1], current.min_support + step))

        new_conf = current.min_confidence
        if mutate_conf:
            step = random.gauss(0, 0.08 * temperature)
            new_conf = max(cls.MIN_CONFIDENCE_BOUNDS[0], min(cls.MIN_CONFIDENCE_BOUNDS[1], current.min_confidence + step))

        new_alg = current.algorithm
        if mutate_alg:
            choices = [a for a in cls.ALGORITHMS if a != current.algorithm]
            new_alg = random.choice(choices)

        new_len = current.max_itemset_length
        if mutate_len:
            delta = random.choice([-1, 1])
            new_len = max(cls.MAX_LENGTH_BOUNDS[0], min(cls.MAX_LENGTH_BOUNDS[1], current.max_itemset_length + delta))

        new_lift = current.min_lift
        if mutate_lift:
            step = random.gauss(0, 0.25 * temperature)
            new_lift = max(cls.MIN_LIFT_BOUNDS[0], min(cls.MIN_LIFT_BOUNDS[1], current.min_lift + step))

        return MiningConfiguration(
            min_support=round(new_supp, 4),
            min_confidence=round(new_conf, 4),
            max_itemset_length=new_len,
            algorithm=new_alg,
            min_lift=round(new_lift, 3),
            top_k=current.top_k
        )
