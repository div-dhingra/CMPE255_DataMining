"""
Autoresearch Stochastic Hill-Climber with Simulated Annealing & Restarts.
Implements:
- Candidate neighbor generation via stochastic mutation
- Metropolis / Simulated Annealing acceptance probability: P = exp(Delta / T_t)
- Tabu hash memory to prevent cycling over previously visited states
- Stagnation detection and Global Random Restarts upon reaching local plateaus
- Synchronous step execution, batch runner, and async SSE generator
"""

import asyncio
import copy
from datetime import datetime, timezone
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable, AsyncGenerator
import numpy as np

from .search_space import SearchSpace
from .objective import CompositeObjective, ObjectiveEvaluation


@dataclass
class StepLog:
    """Detailed telemetry record for a single optimization step."""
    step: int
    timestamp: str
    algorithm: str
    candidate_theta: Dict[str, Any]
    current_theta: Dict[str, Any]
    best_theta: Dict[str, Any]
    fitness: float
    delta_fitness: float
    current_fitness: float
    best_fitness: float
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability_ari: float
    n_clusters: int
    noise_ratio: float
    temperature: float
    accepted: bool
    restart: bool
    stagnation_count: int
    execution_time_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "algorithm": self.algorithm,
            "candidate_theta": copy.deepcopy(self.candidate_theta),
            "current_theta": copy.deepcopy(self.current_theta),
            "best_theta": copy.deepcopy(self.best_theta),
            "fitness": round(self.fitness, 4),
            "delta_fitness": round(self.delta_fitness, 4),
            "current_fitness": round(self.current_fitness, 4),
            "best_fitness": round(self.best_fitness, 4),
            "silhouette": round(self.silhouette, 4),
            "davies_bouldin": round(self.davies_bouldin, 4),
            "calinski_harabasz": round(self.calinski_harabasz, 4),
            "stability_ari": round(self.stability_ari, 4),
            "n_clusters": self.n_clusters,
            "noise_ratio": round(self.noise_ratio, 4),
            "temperature": round(self.temperature, 5),
            "accepted": self.accepted,
            "restart": self.restart,
            "stagnation_count": self.stagnation_count,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "error": self.error,
        }


@dataclass
class OptimizationResult:
    """Final summary outcome of a completed hill-climbing optimization run."""
    best_theta: Dict[str, Any]
    best_fitness: float
    best_eval: ObjectiveEvaluation
    initial_theta: Dict[str, Any]
    initial_fitness: float
    fitness_improvement: float
    total_steps: int
    restart_count: int
    history: List[StepLog]
    total_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_theta": copy.deepcopy(self.best_theta),
            "best_fitness": round(self.best_fitness, 4),
            "best_metrics": self.best_eval.to_dict(),
            "initial_theta": copy.deepcopy(self.initial_theta),
            "initial_fitness": round(self.initial_fitness, 4),
            "fitness_improvement": round(self.fitness_improvement, 4),
            "total_steps": self.total_steps,
            "restart_count": self.restart_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "history_length": len(self.history),
        }


class HillClimbingOptimizer:
    """
    Stochastic Hill-Climbing Optimization Engine with Simulated Annealing Cooling
    and Global Random Restarts.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        objective: Optional[CompositeObjective] = None,
        initial_theta: Optional[Dict[str, Any]] = None,
        target_algorithm: Optional[str] = "kmeans",
        allow_algorithm_mutation: bool = False,
        initial_temperature: float = 0.5,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.001,
        patience: int = 8,
        mutation_rate: float = 0.35,
        random_state: int = 42,
    ):
        self.search_space = search_space or SearchSpace()
        self.objective = objective
        self.target_algorithm = target_algorithm
        self.allow_algorithm_mutation = allow_algorithm_mutation
        self.initial_temperature = float(initial_temperature)
        self.temperature = float(initial_temperature)
        self.cooling_rate = float(cooling_rate)
        self.min_temperature = float(min_temperature)
        self.patience = int(patience)
        self.mutation_rate = float(mutation_rate)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)

        # Optimization State
        self.step_index: int = 0
        self.restart_count: int = 0
        self.stagnation_count: int = 0
        self.tabu_history: Set[str] = set()
        self.history: List[StepLog] = []

        # Initialize start point
        if initial_theta is not None:
            self.current_theta = self.search_space.validate_configuration(initial_theta)
        elif self.target_algorithm is not None:
            self.current_theta = self.search_space.get_default_configuration(self.target_algorithm)
        else:
            self.current_theta = self.search_space.sample_random_configuration(random_state=self.random_state)

        self.initial_theta = copy.deepcopy(self.current_theta)
        self.current_eval: Optional[ObjectiveEvaluation] = None
        self.best_theta: Dict[str, Any] = copy.deepcopy(self.current_theta)
        self.best_eval: Optional[ObjectiveEvaluation] = None
        self.is_initialized: bool = False

    def initialize(self) -> ObjectiveEvaluation:
        """
        Evaluates the starting configuration theta_0.
        """
        if self.objective is None:
            raise ValueError("CompositeObjective must be provided before initialization.")

        init_eval = self.objective.evaluate(self.current_theta)
        self.current_eval = init_eval
        self.best_theta = copy.deepcopy(self.current_theta)
        self.best_eval = copy.deepcopy(init_eval)

        # Record hash in tabu
        c_hash = self.search_space.config_hash(self.current_theta)
        self.tabu_history.add(c_hash)
        self.is_initialized = True
        return init_eval

    def step(self) -> StepLog:
        """
        Performs one single optimization transition:
        1. Checks stagnation / trigger Random Restart if stagnation_count >= patience.
        2. Else perturbs current state to candidate theta'.
        3. Evaluates candidate F(theta').
        4. Applies Simulated Annealing Metropolis acceptance rule.
        5. Updates temperature, history, and state.
        """
        if not self.is_initialized:
            self.initialize()

        assert self.current_eval is not None
        assert self.best_eval is not None

        step_start_time = time.perf_counter()
        now_iso = datetime.now(timezone.utc).isoformat()
        is_restart = False

        # --- 1. Check Stagnation & Random Restart ---
        if self.stagnation_count >= self.patience:
            is_restart = True
            # Sample fresh random configuration
            candidate_theta = self._sample_fresh_candidate()
            eval_res = self.objective.evaluate(candidate_theta)
            elapsed_ms = (time.perf_counter() - step_start_time) * 1000.0

            # Jump to new starting location
            self.current_theta = copy.deepcopy(candidate_theta)
            self.current_eval = eval_res
            delta = eval_res.fitness - self.best_eval.fitness

            if eval_res.fitness > self.best_eval.fitness:
                self.best_theta = copy.deepcopy(candidate_theta)
                self.best_eval = copy.deepcopy(eval_res)

            self.stagnation_count = 0
            self.restart_count += 1
            self.step_index += 1

            # Reset temperature on restart to allow wide exploration
            self.temperature = max(self.temperature, self.initial_temperature * 0.75)

            step_log = StepLog(
                step=self.step_index,
                timestamp=now_iso,
                algorithm=candidate_theta.get("algorithm", "kmeans"),
                candidate_theta=candidate_theta,
                current_theta=copy.deepcopy(self.current_theta),
                best_theta=copy.deepcopy(self.best_theta),
                fitness=eval_res.fitness,
                delta_fitness=delta,
                current_fitness=self.current_eval.fitness,
                best_fitness=self.best_eval.fitness,
                silhouette=eval_res.silhouette,
                davies_bouldin=eval_res.davies_bouldin,
                calinski_harabasz=eval_res.calinski_harabasz,
                stability_ari=eval_res.stability_ari,
                n_clusters=eval_res.n_clusters,
                noise_ratio=eval_res.noise_ratio,
                temperature=self.temperature,
                accepted=True,
                restart=True,
                stagnation_count=0,
                execution_time_ms=elapsed_ms,
                error=eval_res.error,
            )
            self.history.append(step_log)
            return step_log

        # --- 2. Standard Mutation Neighbor Generation ---
        candidate_theta = self._generate_neighbor_candidate()
        c_hash = self.search_space.config_hash(candidate_theta)
        self.tabu_history.add(c_hash)

        # --- 3. Evaluate Candidate ---
        eval_cand = self.objective.evaluate(candidate_theta)
        elapsed_ms = (time.perf_counter() - step_start_time) * 1000.0

        # --- 4. Simulated Annealing Acceptance Decision ---
        delta = eval_cand.fitness - self.current_eval.fitness
        accepted = False

        if delta > 0.0:
            accepted = True
        elif eval_cand.fitness >= 0.0 and self.temperature > 1e-5:
            # Metropolis criterion: P = exp(Delta / T)
            # Clip exponent to avoid math overflow/underflow
            exponent = np.clip(delta / self.temperature, -50.0, 0.0)
            p_accept = float(np.exp(exponent))
            rand_val = float(self.rng.uniform(0.0, 1.0))
            if rand_val < p_accept:
                accepted = True

        # Update current state if accepted
        if accepted:
            self.current_theta = copy.deepcopy(candidate_theta)
            self.current_eval = eval_cand

        # Update best-so-far candidate
        if eval_cand.fitness > self.best_eval.fitness:
            self.best_theta = copy.deepcopy(candidate_theta)
            self.best_eval = copy.deepcopy(eval_cand)
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        # --- 5. Cool Temperature ---
        self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)
        self.step_index += 1

        step_log = StepLog(
            step=self.step_index,
            timestamp=now_iso,
            algorithm=candidate_theta.get("algorithm", "kmeans"),
            candidate_theta=candidate_theta,
            current_theta=copy.deepcopy(self.current_theta),
            best_theta=copy.deepcopy(self.best_theta),
            fitness=eval_cand.fitness,
            delta_fitness=delta,
            current_fitness=self.current_eval.fitness,
            best_fitness=self.best_eval.fitness,
            silhouette=eval_cand.silhouette,
            davies_bouldin=eval_cand.davies_bouldin,
            calinski_harabasz=eval_cand.calinski_harabasz,
            stability_ari=eval_cand.stability_ari,
            n_clusters=eval_cand.n_clusters,
            noise_ratio=eval_cand.noise_ratio,
            temperature=self.temperature,
            accepted=accepted,
            restart=False,
            stagnation_count=self.stagnation_count,
            execution_time_ms=elapsed_ms,
            error=eval_cand.error,
        )
        self.history.append(step_log)
        return step_log

    def _generate_neighbor_candidate(self) -> Dict[str, Any]:
        """
        Generates a non-tabu candidate by mutating current state, falling back to fresh sampling.
        """
        for _ in range(12):
            seed = int(self.rng.integers(0, 1_000_000))
            cand = self.search_space.mutate(
                self.current_theta,
                mutation_rate=self.mutation_rate,
                allow_algorithm_mutation=self.allow_algorithm_mutation,
                random_state=seed,
            )
            c_hash = self.search_space.config_hash(cand)
            if c_hash not in self.tabu_history:
                return cand

        # Fallback to fresh random configuration if all immediate neighbors are tabu
        return self._sample_fresh_candidate()

    def _sample_fresh_candidate(self) -> Dict[str, Any]:
        """
        Samples a random valid configuration not present in tabu memory.
        """
        for _ in range(20):
            seed = int(self.rng.integers(0, 1_000_000))
            cand = self.search_space.sample_random_configuration(
                algorithm=self.target_algorithm if not self.allow_algorithm_mutation else None,
                random_state=seed,
            )
            c_hash = self.search_space.config_hash(cand)
            if c_hash not in self.tabu_history:
                self.tabu_history.add(c_hash)
                return cand

        # If thoroughly saturated, reset tabu cache and return sampled config
        self.tabu_history.clear()
        return self.search_space.sample_random_configuration(
            algorithm=self.target_algorithm if not self.allow_algorithm_mutation else None,
            random_state=int(self.rng.integers(0, 1_000_000)),
        )

    def run(
        self,
        max_steps: int = 30,
        callback: Optional[Callable[[StepLog], None]] = None,
    ) -> OptimizationResult:
        """
        Runs optimization synchronously for max_steps iterations.
        """
        if not self.is_initialized:
            self.initialize()

        assert self.current_eval is not None
        assert self.best_eval is not None

        start_time = time.perf_counter()
        initial_fitness = self.current_eval.fitness

        for _ in range(max_steps):
            step_log = self.step()
            if callback is not None:
                callback(step_log)

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        improvement = self.best_eval.fitness - initial_fitness

        return OptimizationResult(
            best_theta=copy.deepcopy(self.best_theta),
            best_fitness=self.best_eval.fitness,
            best_eval=copy.deepcopy(self.best_eval),
            initial_theta=copy.deepcopy(self.initial_theta),
            initial_fitness=initial_fitness,
            fitness_improvement=improvement,
            total_steps=len(self.history),
            restart_count=self.restart_count,
            history=list(self.history),
            total_time_ms=total_elapsed_ms,
        )

    async def run_stream(
        self,
        max_steps: int = 30,
        delay_seconds: float = 0.0,
    ) -> AsyncGenerator[StepLog, None]:
        """
        Asynchronous generator yielding each StepLog in real time (for SSE/WebSockets).
        """
        if not self.is_initialized:
            self.initialize()

        for _ in range(max_steps):
            step_log = self.step()
            yield step_log
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            else:
                await asyncio.sleep(0)  # Yield event loop
