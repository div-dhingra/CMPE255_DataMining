"""Autoresearch and Hill-Climbing Optimization Engine."""

from .space import SearchSpace, MiningConfiguration
from .fitness import CompositeFitnessEvaluator, FitnessBreakdown
from .optimizer import HillClimbingOptimizer, StepLog
from .ledger import ExperimentLedger
from .literature import LiteratureAlignmentDatabase

__all__ = [
    "SearchSpace",
    "MiningConfiguration",
    "CompositeFitnessEvaluator",
    "FitnessBreakdown",
    "HillClimbingOptimizer",
    "StepLog",
    "ExperimentLedger",
    "LiteratureAlignmentDatabase"
]
