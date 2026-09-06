"""FastAPI routes for Autoresearch hill-climbing engine, telemetry, SSE stream, and literature."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import json
import asyncio

from ..core.state import app_state
from ..autoresearch.optimizer import HillClimbingOptimizer
from ..autoresearch.literature import LiteratureAlignmentDatabase
from .schemas import AutoresearchTriggerRequest, AutoresearchStepResponse

router = APIRouter(prefix="/autoresearch", tags=["Autoresearch"])


@router.post("/step", response_model=AutoresearchStepResponse)
def execute_single_step():
    """Executes a single step of the hill-climbing search loop."""
    if not app_state.optimizer:
        app_state.optimizer = HillClimbingOptimizer(dataset=app_state.dataset)

    step_log = app_state.optimizer.step()

    return AutoresearchStepResponse(
        iteration=step_log.iteration,
        action=step_log.action,
        candidate_fitness=step_log.candidate_fitness,
        current_fitness=step_log.current_fitness,
        best_fitness=step_log.best_fitness,
        fitness_delta=step_log.fitness_delta,
        accepted=step_log.accepted,
        temperature=step_log.temperature,
        execution_time_ms=step_log.execution_time_ms,
        configuration=step_log.configuration,
        details=step_log.details or "",
        rule_count=step_log.rule_count,
        mean_lift=step_log.mean_lift,
        mean_kulczynski=step_log.mean_kulczynski
    )


@router.post("/trigger")
def trigger_batch_autoresearch(req: AutoresearchTriggerRequest) -> Dict[str, Any]:
    """Runs a batch of autonomous hill-climbing search steps."""
    if not app_state.optimizer:
        app_state.optimizer = HillClimbingOptimizer(
            dataset=app_state.dataset,
            cooling_rate=req.cooling_rate,
            max_stagnation=req.max_stagnation
        )
    else:
        app_state.optimizer.cooling_rate = req.cooling_rate
        app_state.optimizer.max_stagnation = req.max_stagnation

    result = app_state.optimizer.run(max_steps=req.max_steps)
    return result


@router.get("/stream")
async def stream_autoresearch(steps: int = 15):
    """Server-Sent Events (SSE) streaming real-time optimization steps to dashboard."""
    if not app_state.optimizer:
        app_state.optimizer = HillClimbingOptimizer(dataset=app_state.dataset)

    async def event_generator():
        for _ in range(steps):
            # Run 1 step in background executor or synchronously
            step_log = app_state.optimizer.step()
            payload = json.dumps(step_log.to_dict())
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.12)  # Smooth cadence for UI chart rendering

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/trajectory")
def get_trajectory():
    """Retrieves complete optimization trajectory time-series for plotting."""
    if not app_state.optimizer:
        return {"iterations": [], "best_fitness": []}
    return app_state.optimizer.ledger.get_trajectory()


@router.get("/leaderboard")
def get_leaderboard(top_n: int = 10) -> List[Dict[str, Any]]:
    """Retrieves top configurations discovered across all experiments."""
    if not app_state.optimizer:
        return []
    return app_state.optimizer.ledger.get_leaderboard(top_n=top_n)


@router.get("/literature")
def get_literature_citations() -> List[Dict[str, Any]]:
    """Returns theoretical literature alignment database entries."""
    return LiteratureAlignmentDatabase.get_all()


@router.post("/apply-best")
def apply_best_configuration() -> Dict[str, Any]:
    """Applies the best discovered configuration to the active application mining rules."""
    if not app_state.optimizer or not app_state.optimizer.best_config:
        raise HTTPException(status_code=400, detail="No optimization run available yet.")

    cfg = app_state.optimizer.best_config

    # Re-run mining with best config
    breakdown, rules, mining_res = app_state.optimizer._evaluate_config(cfg)
    app_state.cached_rules = rules
    app_state.cached_mining_result = mining_res

    return {
        "applied_configuration": cfg.to_dict(),
        "resulting_fitness": round(breakdown.composite_fitness, 4),
        "rule_count": len(rules),
        "message": f"Successfully updated system rules with best configuration (Fitness: {round(breakdown.composite_fitness, 4)})."
    }


@router.post("/reset")
def reset_autoresearch():
    """Resets the autoresearch optimizer and history ledger."""
    app_state.optimizer = HillClimbingOptimizer(dataset=app_state.dataset)
    return {"status": "success", "message": "Autoresearch engine reset."}
