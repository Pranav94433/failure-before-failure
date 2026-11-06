
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database.storage import Database
from backend.routes.analysis import router as analysis_router
from backend.services.model_service import AnalysisService


@asynccontextmanager
async def lifespan(app: FastAPI):
    database = Database()
    database.initialize()
    app.state.database = database
    app.state.analysis_service = AnalysisService.from_saved_models()
    yield


# Create our FastAPI application
app = FastAPI(
    title="Failure Before Failure",
    description="AI-Powered Industrial Predictive Maintenance System",
    version="1.0.0",
    lifespan=lifespan,
)
configured_origins = os.getenv("ALLOWED_ORIGINS")
allowed_origins = (
    [origin.strip().rstrip("/") for origin in configured_origins.split(",") if origin.strip()]
    if configured_origins
    else ["http://localhost:5173", "http://127.0.0.1:5173"]
)
if not allowed_origins:
    raise ValueError("ALLOWED_ORIGINS must contain at least one website origin.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(analysis_router)

# Welcome endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Failure Before Failure",
        "project": "Industrial Predictive Maintenance System",
        "status": "Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Failure Before Failure API is working"
    }
