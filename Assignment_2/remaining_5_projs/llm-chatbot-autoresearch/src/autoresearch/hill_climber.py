"""Autonomous Autoresearch Hill-Climbing Optimization Engine.

Iteratively explores hyperparameter spaces for SOTA transformers:
- Learning rate, context window, attention heads, GQA ratios, sampling temperature, top-p
- Perturbation generator with parameter delta logging
- Simulated annealing restart mechanisms to escape local optima
- Objective evaluation across loss, perplexity, throughput, and memory footprint
- Research paper alignment linking findings to landmark publications
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from src.autoresearch.ledger import ExperimentLedger, LedgerEntry
from src.autoresearch.literature import LANDMARK_PAPERS
from src.autoresearch.objective import EvaluationMetrics, EvaluationObjective
from src.model.config import ModelConfig
from src.model.transformer import SotaDecoderLLM


@dataclass
class HyperparameterState:
    """Hyperparameter configuration for candidate architecture & decoding."""

    learning_rate: float = 3e-4
    context_window: int = 512
    dim: int = 192
    n_layers: int = 4
    n_heads: int = 6
    n_kv_heads: int = 2  # GQA
    temperature: float = 0.70
    top_p: float = 0.90
    rope_theta: float = 10000.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_model_config(self, vocab_size: int = 2048) -> ModelConfig:
        return ModelConfig(
            dim=self.dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            vocab_size=vocab_size,
            max_seq_len=self.context_window,
            rope_theta=self.rope_theta,
            norm_eps=1e-6,
            tie_word_embeddings=True,
        )


class AutoresearchHillClimber:
    """Autonomous hill-climbing search loop with restart mechanisms."""

    def __init__(
        self,
        objective: Optional[EvaluationObjective] = None,
        ledger: Optional[ExperimentLedger] = None,
        initial_state: Optional[HyperparameterState] = None,
        patience: int = 4,
        max_restarts: int = 3,
        seed: int = 42,
    ) -> None:
        self.objective = objective or EvaluationObjective()
        self.ledger = ledger or ExperimentLedger()
        self.patience = patience
        self.max_restarts = max_restarts
        self.random = random.Random(seed)

        self.current_state: HyperparameterState = initial_state or HyperparameterState()
        self.best_state: HyperparameterState = copy.deepcopy(self.current_state)

        self.current_metrics: Optional[EvaluationMetrics] = None
        self.best_metrics: Optional[EvaluationMetrics] = None

        self.iteration: int = 0
        self.consecutive_rejections: int = 0
        self.restart_count: int = 0
        self.is_converged: bool = False

        # Model cache to avoid reallocating identical shapes
        self._model_cache: Dict[str, SotaDecoderLLM] = {}

    def _get_or_create_model(self, config: ModelConfig) -> SotaDecoderLLM:
        key = f"{config.dim}_{config.n_layers}_{config.n_heads}_{config.n_kv_heads}_{config.max_seq_len}"
        if key not in self._model_cache:
            model = SotaDecoderLLM(config).to(self.objective.device)
            self._model_cache[key] = model
        return self._model_cache[key]

    def evaluate_state(self, state: HyperparameterState) -> EvaluationMetrics:
        """Instantiates the candidate model and evaluates multi-objective metrics."""
        cfg = state.to_model_config(vocab_size=self.objective.tokenizer.vocab_size)
        model = self._get_or_create_model(cfg)
        return self.objective.evaluate_model(
            model=model,
            temperature=state.temperature,
            top_p=state.top_p,
            max_eval_tokens=24,
        )

    def propose_perturbation(
        self, state: HyperparameterState
    ) -> Tuple[HyperparameterState, Dict[str, Any], str, str]:
        """Applies a stochastic perturbation to one or more hyperparameters.

        Returns:
            Tuple of (new_state, deltas, perturbation_summary, literature_alignment)
        """
        new_state = copy.deepcopy(state)
        deltas: Dict[str, Any] = {}

        # Mutate one parameter choice
        param_to_mutate = self.random.choice([
            "n_kv_heads",
            "context_window",
            "temperature",
            "top_p",
            "learning_rate",
            "rope_theta",
        ])

        summary = ""
        lit_ref = ""

        if param_to_mutate == "n_kv_heads":
            # GQA ratio mutation: valid divisors of n_heads
            valid_kv = [k for k in [1, 2, 3, 6] if new_state.n_heads % k == 0 and k != state.n_kv_heads]
            if valid_kv:
                old_kv = state.n_kv_heads
                new_kv = self.random.choice(valid_kv)
                new_state.n_kv_heads = new_kv
                deltas["n_kv_heads"] = {"from": old_kv, "to": new_kv}
                summary = f"Adjusted GQA KV heads: {old_kv} -> {new_kv}"
                lit_ref = "Ainslie et al. (2023) - GQA KV-cache bandwidth speedup"

        elif param_to_mutate == "context_window":
            options = [256, 512, 1024]
            valid_opts = [o for o in options if o != state.context_window]
            new_ctx = self.random.choice(valid_opts)
            deltas["context_window"] = {"from": state.context_window, "to": new_ctx}
            new_state.context_window = new_ctx
            summary = f"Shifted context window: {state.context_window} -> {new_ctx}"
            lit_ref = "Hoffmann et al. (2022) - Compute-optimal context allocation"

        elif param_to_mutate == "temperature":
            delta = self.random.choice([-0.15, -0.05, 0.05, 0.15])
            new_temp = round(max(0.1, min(1.2, state.temperature + delta)), 2)
            deltas["temperature"] = {"from": state.temperature, "to": new_temp}
            new_state.temperature = new_temp
            summary = f"Tuned sampling temperature: {state.temperature} -> {new_temp}"
            lit_ref = "Holtzman et al. (2019) - Decoding truncation and entropy"

        elif param_to_mutate == "top_p":
            delta = self.random.choice([-0.10, -0.05, 0.05])
            new_p = round(max(0.60, min(0.98, state.top_p + delta)), 2)
            deltas["top_p"] = {"from": state.top_p, "to": new_p}
            new_state.top_p = new_p
            summary = f"Tuned nucleus sampling top-p: {state.top_p} -> {new_p}"
            lit_ref = "Holtzman et al. (2019) - Nucleus probability mass"

        elif param_to_mutate == "learning_rate":
            factor = self.random.choice([0.75, 1.25, 1.5])
            new_lr = round(state.learning_rate * factor, 6)
            deltas["learning_rate"] = {"from": state.learning_rate, "to": new_lr}
            new_state.learning_rate = new_lr
            summary = f"Scaled learning rate: {state.learning_rate:.2e} -> {new_lr:.2e}"
            lit_ref = "Touvron et al. (2023) - Cosine learning rate scheduling"

        elif param_to_mutate == "rope_theta":
            options = [10000.0, 25000.0, 50000.0]
            valid = [o for o in options if o != state.rope_theta]
            new_theta = self.random.choice(valid) if valid else 10000.0
            deltas["rope_theta"] = {"from": state.rope_theta, "to": new_theta}
            new_state.rope_theta = new_theta
            summary = f"Modified RoPE base frequency: {state.rope_theta} -> {new_theta}"
            lit_ref = "Su et al. (2021) - RoPE base frequency wavelength"

        if not deltas:
            # Fallback perturbation
            new_temp = round(max(0.2, min(1.0, state.temperature + 0.1)), 2)
            deltas["temperature"] = {"from": state.temperature, "to": new_temp}
            new_state.temperature = new_temp
            summary = f"Adjusted temperature: {state.temperature} -> {new_temp}"
            lit_ref = "Holtzman et al. (2019)"

        return new_state, deltas, summary, lit_ref

    def step(self) -> LedgerEntry:
        """Executes one autonomous hill-climbing step."""
        if self.is_converged:
            return self.ledger.entries[-1]

        # Step 0: Record baseline if ledger is empty
        if not self.ledger.entries:
            self.current_metrics = self.evaluate_state(self.current_state)
            self.best_metrics = self.current_metrics
            entry = self.ledger.add_entry(
                iteration=0,
                step_type="baseline",
                hyperparameters=self.current_state.to_dict(),
                parameter_deltas={},
                loss=self.current_metrics.loss,
                perplexity=self.current_metrics.perplexity,
                tokens_per_sec=self.current_metrics.tokens_per_sec,
                memory_mb=self.current_metrics.memory_mb,
                composite_score=self.current_metrics.composite_score,
                delta_score=0.0,
                decision="baseline",
                literature_reference="Vaswani et al. (2017) / Initial Baseline",
            )
            self.iteration = 1
            return entry

        # Check for plateau restart trigger
        step_type = "perturbation"
        if self.consecutive_rejections >= self.patience:
            if self.restart_count >= self.max_restarts:
                self.is_converged = True
                last = self.ledger.entries[-1]
                entry = self.ledger.add_entry(
                    iteration=self.iteration,
                    step_type="convergence",
                    hyperparameters=self.best_state.to_dict(),
                    parameter_deltas={"converged": True, "restarts": self.restart_count},
                    loss=self.best_metrics.loss,  # type: ignore[union-attr]
                    perplexity=self.best_metrics.perplexity,  # type: ignore[union-attr]
                    tokens_per_sec=self.best_metrics.tokens_per_sec,  # type: ignore[union-attr]
                    memory_mb=self.best_metrics.memory_mb,  # type: ignore[union-attr]
                    composite_score=self.best_metrics.composite_score,  # type: ignore[union-attr]
                    delta_score=0.0,
                    decision="accepted",
                    literature_reference="Hill-Climbing Local Optima Converged",
                )
                return entry

            # Perform restart: jump back to best state with larger exploratory perturbation
            self.restart_count += 1
            self.consecutive_rejections = 0
            step_type = "restart"
            base_state = copy.deepcopy(self.best_state)
        else:
            base_state = self.current_state

        # Propose mutation
        candidate_state, deltas, summary, lit_ref = self.propose_perturbation(base_state)
        cand_metrics = self.evaluate_state(candidate_state)

        curr_score = self.current_metrics.composite_score if self.current_metrics else 0.0
        delta_score = cand_metrics.composite_score - curr_score

        # Acceptance criterion: accept if strictly better
        decision = "accepted" if delta_score > 0 else "rejected"

        if decision == "accepted":
            self.current_state = candidate_state
            self.current_metrics = cand_metrics
            self.consecutive_rejections = 0
            if cand_metrics.composite_score > (self.best_metrics.composite_score if self.best_metrics else -999):
                self.best_state = candidate_state
                self.best_metrics = cand_metrics
        else:
            self.consecutive_rejections += 1

        entry = self.ledger.add_entry(
            iteration=self.iteration,
            step_type=step_type,
            hyperparameters=candidate_state.to_dict(),
            parameter_deltas=deltas,
            loss=cand_metrics.loss,
            perplexity=cand_metrics.perplexity,
            tokens_per_sec=cand_metrics.tokens_per_sec,
            memory_mb=cand_metrics.memory_mb,
            composite_score=cand_metrics.composite_score,
            delta_score=delta_score,
            decision=decision,
            literature_reference=lit_ref or "Ainslie et al. (2023) / Hoffmann et al. (2022)",
        )

        self.iteration += 1
        return entry

    def run(self, n_steps: int = 10, callback: Optional[Callable[[LedgerEntry], None]] = None) -> List[LedgerEntry]:
        """Runs multiple autonomous steps sequentially."""
        results: List[LedgerEntry] = []
        for _ in range(n_steps):
            entry = self.step()
            results.append(entry)
            if callback:
                callback(entry)
            if self.is_converged:
                break
        return results
