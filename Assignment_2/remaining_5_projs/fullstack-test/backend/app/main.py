import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.database import engine, Base, SessionLocal
from backend.app.routers import tasks, subtasks, tags, analytics
from backend.app.seed import seed_initial_data

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables and seed demo data if empty
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_initial_data(db)
    finally:
        db.close()
    yield
    # Shutdown logic if needed

app = FastAPI(
    title="TodoPro Modern Task Management API",
    description="Fullstack Dynamic Todo Application with Subtasks, Tags, Drag-and-Drop, and Productivity Analytics",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(tasks.router)
app.include_router(subtasks.router)
app.include_router(tags.router)
app.include_router(analytics.router)

# Base directory for frontend
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "TodoPro", "version": "1.0.0"}

@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to TodoPro API. Frontend index.html not yet built."}
