"""
Autoresearch Autonomous Optimization & Telemetry Route Handlers.
Endpoints:
- POST /api/v1/autoresearch/start: Trigger new hill-climbing optimization job
- GET  /api/v1/autoresearch/status: Poll job status, convergence history, and best state
- GET  /api/v1/autoresearch/stream: Server-Sent Events (SSE) real-time step streaming
- POST /api/v1/autoresearch/pause: Pause running optimization job
- POST /api/v1/autoresearch/stop: Terminate running optimization job
- GET  /api/v1/autoresearch/leaderboard: Query top-performing configurations leaderboard
- GET  /api/v1/autoresearch/ablations: Export ablation analysis and stage contributions
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Query, HTTPException, status
from fastapi.responses import StreamingResponse

from ..schemas import (
    AutoresearchStartRequest,
    AutoresearchStartResponse,
    AutoresearchStatusResponse,
    StepLogSchema,
    LeaderboardResponse,
    LeaderboardEntry,
    AutoresearchAblationSummary,
)
from ..state import state, AutoresearchJob
from autoresearch.objective import ObjectiveWeights
from autoresearch.experiment_logger import export_ablation_analysis

router = APIRouter(prefix="/autoresearch", tags=["Autoresearch Engine"])


@router.post(
    "/start",
    response_model=AutoresearchStartResponse,
    summary="Trigger new autonomous hill-climbing optimization run",
)
async def start_autoresearch(request: AutoresearchStartRequest) -> AutoresearchStartResponse:
    """
    Spawns a new background autonomous hill-climbing optimization task.
    Explores the pipeline and hyperparameter space $\\Theta$ to maximize composite clustering fitness.
    """
    weights = None
    if request.weights is not None:
        weights = ObjectiveWeights(
            w_silhouette=request.weights.w_silhouette,
            w_davies_bouldin=request.weights.w_davies_bouldin,
            w_calinski_harabasz=request.weights.w_calinski_harabasz,
            w_stability=request.weights.w_stability,
            noise_penalty_weight=request.weights.noise_penalty_weight,
            imbalance_penalty_weight=request.weights.imbalance_penalty_weight,
        )

    job = state.create_autoresearch_job(
        max_steps=request.max_steps,
        patience=request.patience,
        target_algorithm=request.target_algorithm,
        allow_algorithm_mutation=request.allow_algorithm_mutation,
        initial_config=request.initial_config,
        weights=weights,
        initial_temperature=request.initial_temperature,
        cooling_rate=request.cooling_rate,
        random_state=request.random_state,
    )

    # Launch background async task
    job._task = asyncio.create_task(job.run_loop())

    return AutoresearchStartResponse(
        job_id=job.job_id,
        status="running",
        message=f"Autoresearch optimization job '{job.job_id}' started with max {request.max_steps} iterations.",
        max_steps=request.max_steps,
    )


@router.get(
    "/status",
    response_model=AutoresearchStatusResponse,
    summary="Poll job execution status, telemetry trajectory, and best candidate",
)
def get_autoresearch_status(
    job_id: Optional[str] = Query(default=None, description="Job ID (or latest if omitted)")
) -> AutoresearchStatusResponse:
    """
    Returns current execution state, step progress, best fitness achieved so far,
    and complete step log history for the requested job.
    """
    job = state.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active or matching autoresearch job found. Start a run first via /start.",
        )

    history_schemas = [
        StepLogSchema(**s.to_dict()) for s in job.history
    ]

    best_alg = job.best_theta.get("algorithm") if job.best_theta else None

    return AutoresearchStatusResponse(
        job_id=job.job_id,
        status=job.status,
        current_step=job.current_step,
        max_steps=job.max_steps,
        best_fitness=round(float(job.best_fitness), 4),
        best_algorithm=best_alg,
        best_config=job.best_theta,
        restart_count=job.restart_count,
        total_time_ms=round(float(job.total_time_ms), 2),
        history=history_schemas,
        error=job.error,
    )


@router.get(
    "/stream",
    summary="Server-Sent Events (SSE) real-time optimization step stream",
)
async def stream_autoresearch_events(
    job_id: Optional[str] = Query(default=None, description="Job ID to stream (or latest if omitted)")
):
    """
    Streams live optimization telemetry events using standard Server-Sent Events (SSE).
    Clients receive step records, accept/reject decisions, restarts, and convergence updates in real time.
    """
    job = state.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No autoresearch job available for streaming. Start a job first.",
        )

    q = job.register_queue()

    async def event_generator():
        try:
            # Yield initial connect event
            init_payload = {
                "event": "connected",
                "job_id": job.job_id,
                "status": job.status,
                "max_steps": job.max_steps,
            }
            yield f"data: {json.dumps(init_payload)}\n\n"

            # Replay any already logged steps
            for prev_step in job.history:
                step_dict = prev_step.to_dict()
                step_dict["event"] = "step"
                yield f"data: {json.dumps(step_dict)}\n\n"

            if job.status in ("completed", "stopped", "error"):
                complete_payload = {
                    "event": "completed",
                    "job_id": job.job_id,
                    "status": job.status,
                    "best_fitness": round(float(job.best_fitness), 4),
                    "best_theta": job.best_theta,
                }
                yield f"data: {json.dumps(complete_payload)}\n\n"
                return

            # Stream real-time incoming steps
            while True:
                try:
                    step = await asyncio.wait_for(q.get(), timeout=3.0)
                except asyncio.TimeoutError:
                    if job.status in ("completed", "stopped", "error"):
                        break
                    continue

                if step is None:
                    # End of stream sentinel
                    complete_payload = {
                        "event": "completed",
                        "job_id": job.job_id,
                        "status": job.status,
                        "best_fitness": round(float(job.best_fitness), 4),
                        "best_theta": job.best_theta,
                    }
                    yield f"data: {json.dumps(complete_payload)}\n\n"
                    break

                step_dict = step.to_dict()
                step_dict["event"] = "step"
                yield f"data: {json.dumps(step_dict)}\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            job.unregister_queue(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/pause",
    summary="Pause a running autoresearch optimization run",
)
def pause_autoresearch(
    job_id: Optional[str] = Query(default=None, description="Job ID (or latest if omitted)")
):
    """
    Pauses an in-progress hill-climbing optimization run.
    """
    job = state.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No autoresearch job found to pause.",
        )

    if job.status == "running":
        job.pause()
        return {"status": "paused", "job_id": job.job_id, "message": "Job successfully paused."}
    elif job.status == "paused":
        job.resume()
        return {"status": "running", "job_id": job.job_id, "message": "Job resumed."}
    else:
        return {"status": job.status, "job_id": job.job_id, "message": f"Job is in '{job.status}' state."}


@router.post(
    "/stop",
    summary="Stop / abort a running autoresearch job",
)
def stop_autoresearch(
    job_id: Optional[str] = Query(default=None, description="Job ID (or latest if omitted)")
):
    """
    Permanently stops/aborts an active hill-climbing optimization run.
    """
    job = state.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No autoresearch job found to stop.",
        )

    job.stop()
    return {"status": "stopped", "job_id": job.job_id, "message": "Job successfully stopped."}


@router.get(
    "/leaderboard",
    response_model=LeaderboardResponse,
    summary="Query top-performing configurations leaderboard",
)
def get_leaderboard(
    job_id: Optional[str] = Query(default=None, description="Job ID (or latest if omitted)"),
    top_n: int = Query(default=10, ge=1, le=50, description="Max entries to return"),
) -> LeaderboardResponse:
    """
    Returns a ranked leaderboard of unique configurations evaluated during the optimization run,
    sorted in descending order of composite fitness score.
    """
    job = state.get_job(job_id)
    if job is None or len(job.history) == 0:
        # Fallback to empty leaderboard
        return LeaderboardResponse(total_unique_evaluated=0, top_entries=[])

    leaderboard_raw = job.logger.get_leaderboard(top_n=top_n)

    entries: List[LeaderboardEntry] = []
    for rank_idx, item in enumerate(leaderboard_raw, start=1):
        cfg = item.get("config") or item.get("configuration") or {}
        c_hash = item.get("config_hash") or SearchSpace.config_hash(cfg)
        step_val = item.get("step") or item.get("step_discovered") or 1
        entries.append(
            LeaderboardEntry(
                rank=rank_idx,
                config_hash=c_hash,
                algorithm=item.get("algorithm", "unknown"),
                config=cfg,
                fitness=item.get("fitness", 0.0),
                silhouette=item.get("silhouette", 0.0),
                davies_bouldin=item.get("davies_bouldin", 0.0),
                calinski_harabasz=item.get("calinski_harabasz", 0.0),
                stability_ari=item.get("stability_ari", 0.0),
                n_clusters=item.get("n_clusters", 0),
                noise_ratio=item.get("noise_ratio", 0.0),
                step=step_val,
            )
        )

    return LeaderboardResponse(
        total_unique_evaluated=len(leaderboard_raw),
        top_entries=entries,
    )


@router.get(
    "/ablations",
    response_model=AutoresearchAblationSummary,
    summary="Export ablation analysis and parameter importance breakdown",
)
def get_autoresearch_ablations(
    job_id: Optional[str] = Query(default=None, description="Job ID (or latest if omitted)")
) -> AutoresearchAblationSummary:
    """
    Computes parameter variance spread, pipeline stage contributions,
    and baseline vs. best configuration comparison deltas.
    """
    job = state.get_job(job_id)
    if job is None or len(job.history) == 0:
        return AutoresearchAblationSummary(
            stage_contributions={},
            parameter_importance_variance={},
            baseline_vs_best_comparison={"status": "No experiments recorded yet"},
        )

    analysis = export_ablation_analysis(job.history)

    return AutoresearchAblationSummary(
        stage_contributions=analysis.get("stage_contributions", {}),
        parameter_importance_variance=analysis.get("parameter_importance_variance", {}),
        baseline_vs_best_comparison=analysis.get("baseline_vs_best_comparison", {}),
    )
