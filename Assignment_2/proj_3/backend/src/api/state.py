"""
State Management & Background Job Registry for FastAPI Backend.
Maintains in-memory singleton state for:
- Active dataset DataFrame
- Fitted preprocessing pipeline
- Active clustering model and cached evaluation results
- Autoresearch optimization jobs, thread/task management, and SSE broadcasting queues
"""

import asyncio
import copy
from datetime import datetime, timezone
import threading
import time
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
import uuid
import numpy as np
import pandas as pd

from crisp_dm.data_understanding import (
    load_credit_card_data,
    compute_data_understanding_summary,
    NUMERIC_FEATURE_COLUMNS,
)
from crisp_dm.data_preparation import (
    PipelineConfig,
    DataPreparationPipeline,
)
from crisp_dm.models import (
    ClusteringModelBase,
    KMeansModel,
    KMedoidsModel,
    DBSCANModel,
    HDBSCANModel,
    AgglomerativeModel,
    GaussianMixtureModel,
)
from crisp_dm.evaluation import (
    evaluate_clustering_solution,
    compute_silhouette_score,
    compute_inertia_elbow_curve,
    compute_subsampling_stability,
)
from crisp_dm.projections import project_coordinates
from crisp_dm.profiling import (
    compute_cluster_centroids,
    compute_radar_profiles,
    compute_feature_importance,
    generate_cluster_personas,
)
from autoresearch.search_space import SearchSpace
from autoresearch.objective import CompositeObjective, ObjectiveWeights
from autoresearch.hill_climber import HillClimbingOptimizer, StepLog, OptimizationResult
from autoresearch.experiment_logger import ExperimentLogger, export_ablation_analysis


class AutoresearchJob:
    """Manages an active or historical autoresearch optimization run."""

    def __init__(
        self,
        job_id: str,
        df: pd.DataFrame,
        max_steps: int = 30,
        patience: int = 8,
        target_algorithm: Optional[str] = None,
        allow_algorithm_mutation: bool = True,
        initial_config: Optional[Dict[str, Any]] = None,
        weights: Optional[ObjectiveWeights] = None,
        initial_temperature: float = 0.5,
        cooling_rate: float = 0.95,
        random_state: int = 42,
    ):
        self.job_id = job_id
        self.df = df.copy()
        self.max_steps = max_steps
        self.patience = patience
        self.target_algorithm = target_algorithm
        self.allow_algorithm_mutation = allow_algorithm_mutation
        self.initial_config = initial_config
        self.weights = weights or ObjectiveWeights()
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.random_state = random_state

        self.status = "created"  # created, running, paused, stopped, completed, error
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.current_step = 0
        self.history: List[StepLog] = []
        self.best_theta: Optional[Dict[str, Any]] = None
        self.best_fitness: float = -1.0
        self.restart_count: int = 0
        self.total_time_ms: float = 0.0
        self.error: Optional[str] = None

        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._stop_event = asyncio.Event()
        self._step_queues: List[asyncio.Queue] = []
        self._task: Optional[asyncio.Task] = None
        self.optimizer: Optional[HillClimbingOptimizer] = None
        self.logger = ExperimentLogger(run_id=f"api_job_{job_id}")

    def register_queue(self) -> asyncio.Queue:
        """Registers an SSE listener queue."""
        q: asyncio.Queue = asyncio.Queue()
        self._step_queues.append(q)
        return q

    def unregister_queue(self, q: asyncio.Queue) -> None:
        """Removes an SSE listener queue."""
        if q in self._step_queues:
            self._step_queues.remove(q)

    def pause(self) -> None:
        """Pauses the running job."""
        if self.status == "running":
            self.status = "paused"
            self._pause_event.clear()

    def resume(self) -> None:
        """Resumes a paused job."""
        if self.status == "paused":
            self.status = "running"
            self._pause_event.set()

    def stop(self) -> None:
        """Stops/aborts the job."""
        self.status = "stopped"
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused to let loop exit

    async def run_loop(self) -> None:
        """Asynchronous execution loop for the optimizer."""
        self.status = "running"
        start_time = time.perf_counter()

        try:
            # 1. Initialize composite objective
            objective = CompositeObjective(
                df=self.df,
                weights=self.weights,
                stability_bootstraps=3,
                stability_sample_ratio=0.8,
            )

            # 2. Instantiate Hill-Climbing Optimizer
            self.optimizer = HillClimbingOptimizer(
                search_space=SearchSpace(),
                objective=objective,
                initial_theta=self.initial_config,
                target_algorithm=self.target_algorithm,
                allow_algorithm_mutation=self.allow_algorithm_mutation,
                patience=self.patience,
                initial_temperature=self.initial_temperature,
                cooling_rate=self.cooling_rate,
                random_state=self.random_state,
            )
            self.optimizer.initialize()

            for step_idx in range(1, self.max_steps + 1):
                if self._stop_event.is_set():
                    self.status = "stopped"
                    break

                # Wait if paused
                await self._pause_event.wait()

                if self._stop_event.is_set():
                    self.status = "stopped"
                    break

                # Execute one step in executor to avoid blocking event loop
                loop = asyncio.get_running_loop()
                step_log = await loop.run_in_executor(None, self.optimizer.step)

                self.current_step = step_idx
                self.history.append(step_log)
                self.logger.log_step(step_log)
                self.best_theta = self.optimizer.best_theta
                self.best_fitness = self.optimizer.best_eval.fitness if self.optimizer.best_eval else step_log.fitness
                self.restart_count = self.optimizer.restart_count
                self.total_time_ms = (time.perf_counter() - start_time) * 1000.0

                # Broadcast to SSE queues
                for q in list(self._step_queues):
                    try:
                        q.put_nowait(step_log)
                    except Exception:
                        pass

                # Small async yield
                await asyncio.sleep(0.01)

            if self.status != "stopped":
                self.status = "completed"

        except Exception as ex:
            self.status = "error"
            self.error = str(ex)
        finally:
            self.total_time_ms = (time.perf_counter() - start_time) * 1000.0
            # Send sentinel None to all queues to signal completion
            for q in list(self._step_queues):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass


class BackendState:
    """Singleton state container for active dataset, pipeline, model, and jobs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._df: Optional[pd.DataFrame] = None
        self._pipeline: Optional[DataPreparationPipeline] = None
        self._model: Optional[ClusteringModelBase] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._jobs: Dict[str, AutoresearchJob] = {}
        self._latest_job_id: Optional[str] = None

    def get_dataset(self) -> pd.DataFrame:
        """Retrieves or lazy-loads the active customer dataset."""
        with self._lock:
            if self._df is None:
                self._df = load_credit_card_data()
            return self._df.copy()

    def set_dataset(self, df: pd.DataFrame) -> None:
        """Replaces the active dataset in memory and clears cached models."""
        with self._lock:
            self._df = df.copy()
            self._pipeline = None
            self._model = None
            self._last_result = None

    def get_model(self) -> Optional[ClusteringModelBase]:
        """Gets active fitted clustering model."""
        with self._lock:
            return self._model

    def get_pipeline(self) -> Optional[DataPreparationPipeline]:
        """Gets active fitted pipeline."""
        with self._lock:
            return self._pipeline

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Gets cached clustering result."""
        with self._lock:
            return copy.deepcopy(self._last_result)

    def set_clustering_state(
        self,
        pipeline: DataPreparationPipeline,
        model: ClusteringModelBase,
        result: Dict[str, Any],
    ) -> None:
        """Updates active clustering pipeline, model, and result."""
        with self._lock:
            self._pipeline = pipeline
            self._model = model
            self._last_result = result

    def create_autoresearch_job(
        self,
        max_steps: int = 30,
        patience: int = 8,
        target_algorithm: Optional[str] = None,
        allow_algorithm_mutation: bool = True,
        initial_config: Optional[Dict[str, Any]] = None,
        weights: Optional[ObjectiveWeights] = None,
        initial_temperature: float = 0.5,
        cooling_rate: float = 0.95,
        random_state: int = 42,
    ) -> AutoresearchJob:
        """Creates and registers a new autoresearch optimization job."""
        df = self.get_dataset()
        job_id = str(uuid.uuid4())[:8]
        job = AutoresearchJob(
            job_id=job_id,
            df=df,
            max_steps=max_steps,
            patience=patience,
            target_algorithm=target_algorithm,
            allow_algorithm_mutation=allow_algorithm_mutation,
            initial_config=initial_config,
            weights=weights,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            random_state=random_state,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._latest_job_id = job_id
        return job

    def get_job(self, job_id: Optional[str] = None) -> Optional[AutoresearchJob]:
        """Retrieves job by ID or gets the latest job if job_id is None."""
        with self._lock:
            if job_id is None:
                job_id = self._latest_job_id
            if job_id is None:
                return None
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[AutoresearchJob]:
        """Lists all registered jobs."""
        with self._lock:
            return list(self._jobs.values())

    def get_active_job_count(self) -> int:
        """Counts currently running jobs."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "running")


# Global singleton instance
state = BackendState()
