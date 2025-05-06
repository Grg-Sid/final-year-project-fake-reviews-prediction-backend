# main.py
import logging
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Import API routers
from api import predict, files, health

# Import core config and utilities
from core.config import (
    UPLOAD_DIR,
    RESULTS_DIR,
    LOGGING_LEVEL,
)
from utils.cleanup import cleanup_old_files

# --- Logging Setup ---
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Review Analysis API",
    description="API for analyzing reviews to detect authenticity and flag potential issues using ML.",
    version="1.1.0",
    openapi_tags=[  # Define tags for better Swagger UI organization
        {
            "name": "Prediction",
            "description": "Endpoints for making predictions on reviews.",
        },
        {
            "name": "File Analysis",
            "description": "Endpoints for uploading files and managing asynchronous analysis jobs.",
        },
        {"name": "Health", "description": "API health check."},
        {
            "name": "Deprecated",
            "description": "Endpoints that are deprecated and may be removed in future versions.",
        },
    ],
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(predict.router, tags=["Prediction"])
# app.include_router(files.router, prefix="/api/files", tags=["File Analysis"])
app.include_router(files.router, tags=["File Analysis"])
# app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(health.router, tags=["Health"])


# --- Root Endpoint ---
@app.get("/", tags=["Health"])
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Review Analysis API V1.1"}


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Tasks to run when the application starts."""
    logger.info("Application starting up...")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Results directory: {RESULTS_DIR}")

    background_tasks = BackgroundTasks()
    background_tasks.add_task(cleanup_old_files)
    logger.info("Scheduled initial cleanup task.")


# --- Shutdown Event ---
@app.on_event("shutdown")
async def shutdown_event():
    """Tasks to run when the application shuts down."""
    logger.info("Application shutting down...")
    # Add any cleanup logic needed on shutdown here (e.g., closing connections)


# --- Run with Uvicorn (for development) ---
# If you run this file directly (python main.py), it won't work correctly with FastAPI's async nature.
# Use Uvicorn: uvicorn main:app --reload
# Example:
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server directly (for development only)...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
