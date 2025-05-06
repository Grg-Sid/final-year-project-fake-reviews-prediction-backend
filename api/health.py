# api/health.py
from fastapi import APIRouter
from datetime import datetime
from schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Checks if the API is running and returns the current timestamp."""
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat())
