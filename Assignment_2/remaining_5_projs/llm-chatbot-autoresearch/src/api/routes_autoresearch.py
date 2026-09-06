"""FastAPI Router for Autonomous Hill-Climbing Autoresearch Engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from src.autoresearch.hill_climber import AutoresearchHillClimber, HyperparameterState
from src.autoresearch.ledger import ExperimentLedger
from src.autoresearch.objective import EvaluationObjective

router = APIRouter(prefix="/api/autoresearch", tags=["Autoresearch Studio"])

# Singleton climber and ledger
_ledger = ExperimentLedger()
_objective = EvaluationObjective()
_climber = AutoresearchHillClimber(objective=_objective, ledger=_ledger, patience=4, max_restarts=3)


class RunStepsRequest(BaseModel):
    steps: int = Field(default=5, ge=1, le=50)


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Returns current hill-climbing status and best parameters."""
    return {
        "iteration": _climber.iteration,
        "is_converged": _climber.is_converged,
        "consecutive_rejections": _climber.consecutive_rejections,
        "restart_count": _climber.restart_count,
        "current_state": _climber.current_state.to_dict(),
        "best_state": _climber.best_state.to_dict(),
        "summary": _ledger.get_summary(),
    }


@router.post("/step")
async def execute_step() -> Dict[str, Any]:
    """Executes a single autonomous hill-climbing step."""
    entry = _climber.step()
    return {
        "status": "success",
        "entry": entry.to_dict(),
        "summary": _ledger.get_summary(),
    }


@router.post("/run")
async def run_multiple_steps(req: RunStepsRequest) -> Dict[str, Any]:
    """Executes multiple autonomous hill-climbing steps."""
    new_entries = _climber.run(n_steps=req.steps)
    return {
        "status": "success",
        "steps_executed": len(new_entries),
        "latest_entries": [e.to_dict() for e in new_entries],
        "summary": _ledger.get_summary(),
    }


@router.get("/ledger")
async def get_ledger(limit: Optional[int] = Query(default=None, ge=1, le=500)) -> Dict[str, Any]:
    """Retrieves the experiment ledger with iteration records."""
    history = _ledger.get_history()
    if limit:
        history = history[-limit:]
    return {
        "total_entries": len(_ledger.entries),
        "summary": _ledger.get_summary(),
        "history": history,
    }


@router.get("/export/csv")
async def export_csv() -> Response:
    """Exports the full iteration ledger as a downloadable CSV."""
    csv_data = _ledger.to_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=autoresearch_ledger.csv"},
    )


@router.post("/reset")
async def reset_autoresearch() -> Dict[str, str]:
    """Resets the hill climber and experiment ledger."""
    global _climber, _ledger
    _ledger = ExperimentLedger()
    _climber = AutoresearchHillClimber(objective=_objective, ledger=_ledger, patience=4, max_restarts=3)
    return {"status": "ok", "message": "Autoresearch engine reset to baseline"}
