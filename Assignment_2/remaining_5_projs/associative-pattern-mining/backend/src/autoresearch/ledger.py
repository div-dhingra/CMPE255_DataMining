"""Experiment ledger tracking autoresearch trials, deltas, trajectory curves, and exports."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import json


@dataclass
class StepLog:
    """Detailed log record of an optimization step."""
    iteration: int
    timestamp: float
    action: str  # "STEP", "ACCEPT", "REJECT", "RESTART", "IMPROVE_GLOBAL_BEST"
    configuration: Dict[str, Any]
    candidate_fitness: float
    current_fitness: float
    best_fitness: float
    fitness_delta: float
    accepted: bool
    temperature: float
    execution_time_ms: float
    rule_count: int
    mean_lift: float
    mean_kulczynski: float
    coverage_score: float
    diversity_score: float
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "timestamp": round(self.timestamp, 3),
            "action": self.action,
            "configuration": self.configuration,
            "candidate_fitness": round(self.candidate_fitness, 4),
            "current_fitness": round(self.current_fitness, 4),
            "best_fitness": round(self.best_fitness, 4),
            "fitness_delta": round(self.fitness_delta, 4),
            "accepted": self.accepted,
            "temperature": round(self.temperature, 4),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "rule_count": self.rule_count,
            "mean_lift": round(self.mean_lift, 3),
            "mean_kulczynski": round(self.mean_kulczynski, 3),
            "coverage_score": round(self.coverage_score, 4),
            "diversity_score": round(self.diversity_score, 4),
            "details": self.details or ""
        }


class ExperimentLedger:
    """In-memory and serializable ledger for autonomous parameter search experiments."""

    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.logs: List[StepLog] = []
        self.restarts_count: int = 0
        self.start_time: float = time.time()

    def record_step(self, log: StepLog) -> None:
        self.logs.append(log)
        if len(self.logs) > self.max_history:
            self.logs.pop(0)

    def get_trajectory(self) -> Dict[str, Any]:
        """Returns trajectory time-series for plotting and telemetry."""
        return {
            "iterations": [log.iteration for log in self.logs],
            "candidate_fitness": [log.candidate_fitness for log in self.logs],
            "current_fitness": [log.current_fitness for log in self.logs],
            "best_fitness": [log.best_fitness for log in self.logs],
            "temperatures": [log.temperature for log in self.logs],
            "latencies_ms": [log.execution_time_ms for log in self.logs],
            "min_supports": [log.configuration["min_support"] for log in self.logs],
            "min_confidences": [log.configuration["min_confidence"] for log in self.logs],
            "algorithms": [log.configuration["algorithm"] for log in self.logs],
            "rule_counts": [log.rule_count for log in self.logs],
            "accepted": [log.accepted for log in self.logs]
        }

    def get_leaderboard(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Returns the top configurations ranked by candidate fitness."""
        # De-duplicate by configuration hash
        seen_configs = set()
        unique_logs = []
        for log in sorted(self.logs, key=lambda x: x.candidate_fitness, reverse=True):
            cfg_key = (
                log.configuration.get("algorithm"),
                log.configuration.get("min_support"),
                log.configuration.get("min_confidence"),
                log.configuration.get("min_lift"),
                log.configuration.get("max_itemset_length")
            )
            if cfg_key not in seen_configs:
                seen_configs.add(cfg_key)
                unique_logs.append(log)
                if len(unique_logs) >= top_n:
                    break

        return [log.to_dict() for log in unique_logs]

    def clear(self) -> None:
        self.logs.clear()
        self.restarts_count = 0
        self.start_time = time.time()
