"""FastAPI routes for CRISP-DM lifecycle monitoring and governance."""

from fastapi import APIRouter
from typing import Dict, Any
from ..core.state import app_state

router = APIRouter(prefix="/crisp-dm", tags=["CRISP-DM"])


@router.get("", response_model=Dict[str, Any])
def get_lifecycle_overview():
    """Returns the full 6-phase CRISP-DM lifecycle state, metrics, and artifact references."""
    summary = app_state.lifecycle.get_lifecycle_summary()
    # Enrich with live dataset and model telemetry
    summary["dataset_transactions"] = app_state.dataset.num_transactions
    summary["dataset_unique_items"] = app_state.dataset.num_unique_items
    summary["rules_active"] = len(app_state.cached_rules)
    summary["autoresearch_best_fitness"] = (
        round(app_state.optimizer.best_fitness, 4) if app_state.optimizer else 0.0
    )
    return summary
