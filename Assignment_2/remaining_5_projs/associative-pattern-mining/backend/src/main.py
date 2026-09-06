"""FastAPI entry point for the Associative Pattern Mining system."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .api import (
    crisp_dm_router,
    data_router,
    mining_router,
    autoresearch_router,
    cart_router
)

app = FastAPI(
    title="Associative Pattern Mining & Autoresearch Engine",
    description="CRISP-DM Associative Pattern Mining with Apriori, FP-Growth, ECLAT, 7 interestingness metrics, and hill-climbing optimization.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers under /api/v1
api_prefix = "/api/v1"
app.include_router(crisp_dm_router, prefix=api_prefix)
app.include_router(data_router, prefix=api_prefix)
app.include_router(mining_router, prefix=api_prefix)
app.include_router(autoresearch_router, prefix=api_prefix)
app.include_router(cart_router, prefix=api_prefix)


@app.get(f"{api_prefix}/health")
def health_check():
    return {
        "status": "healthy",
        "service": "associative-pattern-mining",
        "framework": "CRISP-DM",
        "algorithms": ["Apriori", "FP-Growth", "ECLAT"],
        "metrics_count": 7
    }


# Static frontend hosting
FRONTEND_DIR = Path(__file__).parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    def serve_dashboard():
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Frontend index.html not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.src.main:app", host="0.0.0.0", port=8000, reload=True)
