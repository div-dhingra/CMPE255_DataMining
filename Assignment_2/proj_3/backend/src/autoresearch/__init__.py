"""
Autoresearch & Hill-Climbing Optimization Engine.
Provides:
- SearchSpace: Parameter bounds, neighborhood mutation operator, random config generator
- CompositeObjective: Multi-objective fitness function F(theta) with stability & penalties
- HillClimbingOptimizer: Stochastic hill climber with simulated annealing & restarts
- StepLog, OptimizationResult: Structured execution records
- ExperimentLogger, export_ablation_analysis: JSONL/CSV logging & ablation analytics
"""

from .search_space import (
    SearchSpace,
    PREPROCESSING_SPACE,
    ALGORITHM_CHOICES,
    ALGORITHM_HYPERPARAMETERS,
)

from .objective import (
    CompositeObjective,
    ObjectiveWeights,
    ObjectiveEvaluation,
)

from .hill_climber import (
    HillClimbingOptimizer,
    StepLog,
    OptimizationResult,
)

from .experiment_logger import (
    ExperimentLogger,
    export_ablation_analysis,
)

__all__ = [
    # Search Space
    "SearchSpace",
    "PREPROCESSING_SPACE",
    "ALGORITHM_CHOICES",
    "ALGORITHM_HYPERPARAMETERS",
    # Objective
    "CompositeObjective",
    "ObjectiveWeights",
    "ObjectiveEvaluation",
    # Hill Climber
    "HillClimbingOptimizer",
    "StepLog",
    "OptimizationResult",
    # Logger & Ablations
    "ExperimentLogger",
    "export_ablation_analysis",
]
