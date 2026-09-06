"""Experiment Ledger for Autoresearch Hill-Climbing Optimization.

Maintains an immutable record of all step transitions, parameter perturbations,
metric deltas, accept/reject decisions, and research paper alignments.
Supports export to JSON and CSV formats.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class LedgerEntry:
    """A single recorded iteration step in the hill-climbing trajectory."""

    iteration: int
    step_type: str  # 'baseline', 'perturbation', 'restart', 'convergence'
    hyperparameters: Dict[str, Any]
    parameter_deltas: Dict[str, Any]
    loss: float
    perplexity: float
    tokens_per_sec: float
    memory_mb: float
    composite_score: float
    delta_score: float
    decision: str  # 'accepted', 'rejected', 'baseline'
    literature_reference: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["loss"] = round(self.loss, 4)
        d["perplexity"] = round(self.perplexity, 4)
        d["tokens_per_sec"] = round(self.tokens_per_sec, 2)
        d["memory_mb"] = round(self.memory_mb, 2)
        d["composite_score"] = round(self.composite_score, 4)
        d["delta_score"] = round(self.delta_score, 4)
        return d


class ExperimentLedger:
    """Tracks and persists the complete trajectory of the autoresearch engine."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[LedgerEntry] = []
        self.best_entry: Optional[LedgerEntry] = None

    def add_entry(
        self,
        iteration: int,
        step_type: str,
        hyperparameters: Dict[str, Any],
        parameter_deltas: Dict[str, Any],
        loss: float,
        perplexity: float,
        tokens_per_sec: float,
        memory_mb: float,
        composite_score: float,
        delta_score: float,
        decision: str,
        literature_reference: str,
    ) -> LedgerEntry:
        """Appends an iteration result to the ledger."""
        entry = LedgerEntry(
            iteration=iteration,
            step_type=step_type,
            hyperparameters=hyperparameters,
            parameter_deltas=parameter_deltas,
            loss=loss,
            perplexity=perplexity,
            tokens_per_sec=tokens_per_sec,
            memory_mb=memory_mb,
            composite_score=composite_score,
            delta_score=delta_score,
            decision=decision,
            literature_reference=literature_reference,
            timestamp=time.time(),
        )
        self.entries.append(entry)

        if self.best_entry is None or (
            decision in ("accepted", "baseline") and entry.composite_score > self.best_entry.composite_score
        ):
            self.best_entry = entry

        return entry

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns all entries formatted as dictionaries."""
        return [e.to_dict() for e in self.entries]

    def get_summary(self) -> Dict[str, Any]:
        """Calculates statistical summary of the optimization trajectory."""
        if not self.entries:
            return {"total_steps": 0, "status": "empty"}

        baseline = self.entries[0]
        accepted_count = sum(1 for e in self.entries if e.decision == "accepted")
        rejected_count = sum(1 for e in self.entries if e.decision == "rejected")
        best = self.best_entry or baseline

        score_improvement_pct = 0.0
        if baseline.composite_score > 0:
            score_improvement_pct = (
                (best.composite_score - baseline.composite_score) / baseline.composite_score
            ) * 100.0

        return {
            "total_steps": len(self.entries),
            "accepted_steps": accepted_count,
            "rejected_steps": rejected_count,
            "acceptance_rate": round(accepted_count / max(1, accepted_count + rejected_count), 4),
            "baseline_score": round(baseline.composite_score, 4),
            "best_score": round(best.composite_score, 4),
            "score_improvement_pct": round(score_improvement_pct, 2),
            "best_hyperparameters": best.hyperparameters,
            "best_iteration": best.iteration,
            "best_perplexity": round(best.perplexity, 4),
            "best_tokens_per_sec": round(best.tokens_per_sec, 2),
            "best_memory_mb": round(best.memory_mb, 2),
        }

    def to_csv(self) -> str:
        """Serializes the experiment ledger into CSV string format."""
        if not self.entries:
            return ""

        output = io.StringIO()
        fieldnames = [
            "iteration",
            "step_type",
            "decision",
            "composite_score",
            "delta_score",
            "loss",
            "perplexity",
            "tokens_per_sec",
            "memory_mb",
            "learning_rate",
            "dim",
            "n_heads",
            "n_kv_heads",
            "temperature",
            "top_p",
            "literature_reference",
            "timestamp",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for e in self.entries:
            row = {
                "iteration": e.iteration,
                "step_type": e.step_type,
                "decision": e.decision,
                "composite_score": round(e.composite_score, 4),
                "delta_score": round(e.delta_score, 4),
                "loss": round(e.loss, 4),
                "perplexity": round(e.perplexity, 4),
                "tokens_per_sec": round(e.tokens_per_sec, 2),
                "memory_mb": round(e.memory_mb, 2),
                "learning_rate": e.hyperparameters.get("learning_rate"),
                "dim": e.hyperparameters.get("dim"),
                "n_heads": e.hyperparameters.get("n_heads"),
                "n_kv_heads": e.hyperparameters.get("n_kv_heads"),
                "temperature": e.hyperparameters.get("temperature"),
                "top_p": e.hyperparameters.get("top_p"),
                "literature_reference": e.literature_reference,
                "timestamp": e.timestamp,
            }
            writer.writerow(row)

        return output.getvalue()

    def save_json(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Saves ledger data to JSON."""
        target = Path(filepath) if filepath else (self.output_dir / "autoresearch_ledger.json")  # type: ignore[operator]
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.get_summary(),
            "history": self.get_history(),
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return target

    def save_csv(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Saves ledger data to CSV."""
        target = Path(filepath) if filepath else (self.output_dir / "autoresearch_ledger.csv")  # type: ignore[operator]
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_csv())
        return target
