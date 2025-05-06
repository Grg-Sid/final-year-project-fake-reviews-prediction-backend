# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ReviewRequest(BaseModel):
    review_content: str
    useful_count: int = Field(ge=0, default=0)
    review_count: int = Field(ge=0, default=0)
    friend_count: int = Field(ge=0, default=0)


class AnalysisResponse(BaseModel):
    flagged: bool
    confidence: float
    process_time: float


class BatchAnalysisRequest(BaseModel):
    reviews: List[ReviewRequest]


class FileUploadResponse(BaseModel):
    upload_id: str
    filename: str
    status: str  # e.g., "uploaded"
    upload_time: str
    file_size: int


class AnalysisStatus(BaseModel):
    analysis_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0-100
    error_message: Optional[str] = None
    result_url: Optional[str] = None
    created_at: str
    updated_at: str
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    upload_id: Optional[str] = None
    file_name: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class AnalysisListItem(BaseModel):
    analysis_id: str
    file_name: str
    created_at: str
    status: str
    analysis_type: str
    progress: float
    updated_at: str
    error_message: Optional[str] = None
