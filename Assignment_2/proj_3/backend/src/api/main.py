"""
FastAPI Application Entrypoint for CRISP-DM Clustering & Autoresearch Engine.
Configures:
- Application lifecycle and metadata
- CORS middleware for Next.js frontend integration
- Route registration under /api/v1
- Health check endpoints (/api/v1/health, /health)
- Global exception handling and OpenAPI docs at /docs
"""

from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import HealthResponse, ErrorResponse
from .state import state
from .routes import (
    data_router,
    cluster_router,
    autoresearch_router,
    research_router,
    inference_router,
)


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""

    app = FastAPI(
        title="CRISP-DM Clustering & Autoresearch API",
        version="1.0.0",
        description=(
            "High-Performance Analytical Backend for Customer Segmentation, "
            "Autonomous Hill-Climbing Optimization, Multi-Paradigm Evaluation, "
            "and Research Benchmark Alignment."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/api/v1/openapi.json",
    )

    # 1. CORS Middleware (Supports local Next.js frontend on any port)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Global Exception Handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "detail": str(exc),
                "status_code": 500,
            },
        )

    # 3. Health Check Endpoints
    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["System Health"],
        summary="Service health and cache status",
    )
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System Health"],
        summary="Root health check alias",
    )
    def health_check() -> HealthResponse:
        """
        Returns API service health, active in-memory dataset status,
        cached model state, and active autoresearch job count.
        """
        df = state.get_dataset()
        model = state.get_model()
        active_jobs = state.get_active_job_count()

        return HealthResponse(
            status="ok",
            version="1.0.0",
            dataset_loaded=df is not None,
            dataset_rows=len(df) if df is not None else 0,
            model_cached=model is not None,
            active_algorithm=model.model_name if model is not None else None,
            active_autoresearch_jobs=active_jobs,
        )

    @app.get(
        "/",
        tags=["System Health"],
        summary="Root welcome and API metadata",
    )
    def root() -> Dict[str, Any]:
        """Root welcome endpoint returning API links and version."""
        return {
            "name": "CRISP-DM Clustering & Autoresearch API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/api/v1/health",
            "endpoints_prefix": "/api/v1",
        }

    # 4. Include Routers
    app.include_router(data_router, prefix="/api/v1")
    app.include_router(cluster_router, prefix="/api/v1")
    app.include_router(autoresearch_router, prefix="/api/v1")
    app.include_router(research_router, prefix="/api/v1")
    app.include_router(inference_router, prefix="/api/v1")

    return app


app = create_app()
