"""FastAPI Application Entrypoint for SOTA LLM Chatbot & Autoresearch Engine."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes_autoresearch import router as autoresearch_router
from src.api.routes_benchmarks import router as benchmarks_router
from src.api.routes_chat import router as chat_router
from src.api.routes_model import router as model_router

app = FastAPI(
    title="SOTA LLM Chatbot & Autoresearch Engine API",
    description=(
        "Pure PyTorch Transformer Language Model featuring RoPE, SwiGLU, RMSNorm, "
        "Grouped-Query Attention (GQA), Low-Latency KV-Caching, CRISP-DM lifecycle, "
        "and an Autonomous Hill-Climbing Autoresearch Engine."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(chat_router)
app.include_router(model_router)
app.include_router(autoresearch_router)
app.include_router(benchmarks_router)

# Path to static dashboard files
STATIC_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "static"


@app.get("/health")
async def health_check():
    """Service health and liveness probe."""
    return {
        "status": "healthy",
        "service": "llm-chatbot-autoresearch",
        "version": "1.0.0",
    }


@app.get("/api/crisp-dm/summary")
async def get_crisp_dm():
    """Returns structured CRISP-DM lifecycle documentation summary."""
    from src.api.routes_benchmarks import get_crisp_dm_summary
    return await get_crisp_dm_summary()


# Mount static assets if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the AI Engineer Admin Dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(
        "<h2>Dashboard static files not found. Check src/dashboard/static/index.html</h2>"
    )
