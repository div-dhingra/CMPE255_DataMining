"""
FastAPI Main Application.
Serves REST API and mounts Leaflet interactive frontend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from contextlib import asynccontextmanager

from src.config import STATIC_DIR
from src.api.routes import router
from src.models.predictor import get_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Attempt loading trained models on startup if available
    try:
        predictor = get_predictor()
        predictor.ensure_loaded()
    except Exception:
        pass
    yield


app = FastAPI(
    title="NYC Taxi ML & Trip Estimation Platform",
    description="CRISP-DM Data Science platform for NYC Taxi trip duration and fare prediction",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dev and embedded widgets
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Mount static folder for CSS, JS, Assets
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_frontend():
    """Serve the interactive map frontend."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "NYC Taxi DS API is running. Frontend static/index.html will be served once loaded.",
        "docs": "/docs",
    }
