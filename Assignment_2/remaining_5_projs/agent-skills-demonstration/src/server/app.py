"""FastAPI Main Analytical Backend and Interactive Dashboard Application."""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import ARTIFACTS_DIR, BASE_DIR
from src.server.routes.crisp_router import router as crisp_router
from src.server.routes.inference_router import router as inference_router
from src.server.routes.monitoring_router import router as monitoring_router
from src.server.routes.skills_router import router as skills_router

app = FastAPI(
    title="Agent Skills CRISP-DM Demonstration Platform",
    description="Interactive analytical engine demonstrating 15 param087 agent-ml-skills and 31 nimrodfisher data-analytics-skills on Kaggle Telco Customer Churn.",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(skills_router)
app.include_router(crisp_router)
app.include_router(inference_router)
app.include_router(monitoring_router)

# Mount Static Files
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if ARTIFACTS_DIR.exists():
    app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "platform": "Agent Skills CRISP-DM Demonstration Engine",
        "dataset": "Kaggle Telco Customer Churn (IBM)",
        "skills_total": 46,
        "crisp_dm_phases": 6,
    }


@app.get("/", tags=["Dashboard UI"])
def serve_dashboard():
    """Serve the interactive single-page dashboard HTML application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Agent Skills Demonstration API running. UI index.html not yet initialized."}
