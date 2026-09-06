"""Autoresearch hill-climbing optimization engine and research paper synthesis."""

from src.autoresearch.hill_climber import AutoresearchHillClimber, HyperparameterState
from src.autoresearch.ledger import ExperimentLedger, LedgerEntry
from src.autoresearch.literature import (
    ABLATION_STUDIES,
    BENCHMARK_MATRIX,
    LANDMARK_PAPERS,
    get_literature_summary,
)
from src.autoresearch.objective import EvaluationMetrics, EvaluationObjective

__all__ = [
    "EvaluationMetrics",
    "EvaluationObjective",
    "ExperimentLedger",
    "LedgerEntry",
    "HyperparameterState",
    "AutoresearchHillClimber",
    "LANDMARK_PAPERS",
    "BENCHMARK_MATRIX",
    "ABLATION_STUDIES",
    "get_literature_summary",
]
