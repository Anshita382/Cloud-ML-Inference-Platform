"""Request and response schemas."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input for text classification inference."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")
    model_name: Optional[str] = Field(None, description="Model override (future use)")

    model_config = {"json_schema_extra": {"examples": [{"text": "This movie was absolutely fantastic!"}]}}


class PredictionStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictionResponse(BaseModel):
    """Response after submitting a prediction request."""
    request_id: str
    status: PredictionStatus = PredictionStatus.QUEUED
    message: str = "Request queued for processing"


class PredictionResult(BaseModel):
    """Final prediction result."""
    request_id: str
    status: PredictionStatus
    label: Optional[str] = None
    score: Optional[float] = None
    latency_ms: Optional[float] = None
    batch_size: Optional[int] = None
    error: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction input."""
    texts: list[str] = Field(..., min_length=1, max_length=64, description="List of texts")


class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    model_loaded: bool
    queue_depth: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_requests: int
    total_completed: int
    total_failed: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_batch_size: float
    queue_depth: int
    requests_per_second: float
