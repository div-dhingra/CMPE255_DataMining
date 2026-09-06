"""
FastAPI Route Handlers Package.
Exports modular routers for:
- data: Dataset ingestion, stats, correlations, distributions, and sample preview
- cluster: Model training, projections, personas, silhouette samples, elbow curves
- autoresearch: Start, poll, stream (SSE), pause, stop, leaderboards, and ablations
- research: Academic literature, benchmark matrix with bootstrap bounds, exports
- inference: Single and batch real-time customer segmentation scoring
"""

from .data import router as data_router
from .cluster import router as cluster_router
from .autoresearch import router as autoresearch_router
from .research import router as research_router
from .inference import router as inference_router

__all__ = [
    "data_router",
    "cluster_router",
    "autoresearch_router",
    "research_router",
    "inference_router",
]
