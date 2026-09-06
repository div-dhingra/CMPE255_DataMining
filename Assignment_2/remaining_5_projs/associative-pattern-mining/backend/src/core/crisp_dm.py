"""CRISP-DM framework lifecycle management, telemetry, and pipeline audit tracker."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time


@dataclass
class PhaseStatus:
    phase_id: int
    name: str
    status: str  # "PENDING", "IN_PROGRESS", "COMPLETED", "FAILED"
    description: str
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class CrispDMLifecycle:
    """Tracks and validates all 6 phases of the CRISP-DM standard."""

    def __init__(self):
        self.phases: Dict[int, PhaseStatus] = {
            1: PhaseStatus(
                phase_id=1,
                name="Business Understanding",
                status="COMPLETED",
                description="Formulate retail basket affinity goals, shelf placement optimization, and cross-sell uplift targets.",
                key_metrics={
                    "target_metric": "Lift >= 1.25, Confidence >= 0.50",
                    "objective": "Maximize Average Order Value (AOV) via high-affinity bundling",
                    "primary_kpi": "Cross-sell bundle conversion rate"
                },
                artifacts=["docs/CRISP_DM.md#phase-1-business-understanding"]
            ),
            2: PhaseStatus(
                phase_id=2,
                name="Data Understanding",
                status="COMPLETED",
                description="Exploratory data analysis on Kaggle Online Retail / Market Basket transactions.",
                key_metrics={
                    "source": "Kaggle Online Retail / Synthetic Benchmark",
                    "sparsity_analysis": "Completed",
                    "frequency_distribution": "Power-law / Zipfian"
                },
                artifacts=["docs/CRISP_DM.md#phase-2-data-understanding", "data/online_retail_sample.csv"]
            ),
            3: PhaseStatus(
                phase_id=3,
                name="Data Preparation",
                status="COMPLETED",
                description="Filter cancellations (InvoiceNo 'C'), clean zero-quantities, one-hot encode transactions.",
                key_metrics={
                    "cancellations_filtered": True,
                    "invalid_sku_scrubbed": True,
                    "encoding_format": "Sparse Boolean One-Hot & Vertical TID Bitsets"
                },
                artifacts=["docs/CRISP_DM.md#phase-3-data-preparation", "backend/src/core/dataset.py"]
            ),
            4: PhaseStatus(
                phase_id=4,
                name="Modeling",
                status="COMPLETED",
                description="Frequent itemset mining via Apriori, FP-Growth, and ECLAT algorithms.",
                key_metrics={
                    "algorithms_active": ["Apriori", "FP-Growth", "ECLAT"],
                    "candidate_pruning": "Downward-Closure & Header-Table Conditional Trees",
                    "execution_mode": "Multi-Paradigm Benchmark"
                },
                artifacts=["docs/CRISP_DM.md#phase-4-modeling", "backend/src/algorithms/"]
            ),
            5: PhaseStatus(
                phase_id=5,
                name="Evaluation",
                status="COMPLETED",
                description="Rule interestingness evaluation across 7 metrics with non-redundancy pruning.",
                key_metrics={
                    "evaluated_metrics": ["Support", "Confidence", "Lift", "Conviction", "Leverage", "Zhang", "Kulczynski"],
                    "pruning_strategy": "Parsimonious Non-Redundant Rule Filter"
                },
                artifacts=["docs/CRISP_DM.md#phase-5-evaluation", "backend/src/core/metrics.py"]
            ),
            6: PhaseStatus(
                phase_id=6,
                name="Deployment",
                status="COMPLETED",
                description="FastAPI service, Real-time Cart Recommendation Playground, and Autoresearch Optimizer.",
                key_metrics={
                    "serving_layer": "FastAPI REST + SSE Telemetry",
                    "admin_dashboard": "Interactive Network Graph & Matrix Studio",
                    "autoresearch": "Autonomous Hill-Climbing Optimization Engine"
                },
                artifacts=["docs/CRISP_DM.md#phase-6-deployment", "backend/src/main.py"]
            ),
        }

    def update_phase(self, phase_id: int, status: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        if phase_id in self.phases:
            self.phases[phase_id].status = status
            self.phases[phase_id].updated_at = time.time()
            if metrics:
                self.phases[phase_id].key_metrics.update(metrics)

    def get_lifecycle_summary(self) -> Dict[str, Any]:
        return {
            "total_phases": len(self.phases),
            "completed_phases": sum(1 for p in self.phases.values() if p.status == "COMPLETED"),
            "phases": [
                {
                    "phase_id": p.phase_id,
                    "name": p.name,
                    "status": p.status,
                    "description": p.description,
                    "key_metrics": p.key_metrics,
                    "artifacts": p.artifacts,
                    "updated_at": p.updated_at
                }
                for p in sorted(self.phases.values(), key=lambda x: x.phase_id)
            ]
        }
