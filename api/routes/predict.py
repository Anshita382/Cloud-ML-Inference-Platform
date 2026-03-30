"""Prediction endpoints."""

import time
from fastapi import APIRouter, HTTPException, Request

from api.schemas.models import (
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    PredictionStatus,
    BatchPredictionRequest,
)
from api.services.metrics import REQUESTS_TOTAL, REQUESTS_IN_QUEUE

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, payload: PredictionRequest):
    """
    Submit a text for sentiment classification.

    The request is queued and processed asynchronously.
    Use /result/{request_id} to poll for the result.
    """
    queue_service = request.app.state.queue_service

    try:
        request_id = await queue_service.enqueue(payload.text)
        REQUESTS_TOTAL.labels(status="queued").inc()

        # Update queue depth metric
        depth = await queue_service.get_queue_depth()
        REQUESTS_IN_QUEUE.set(depth)

        return PredictionResponse(
            request_id=request_id,
            status=PredictionStatus.QUEUED,
            message="Request queued for processing",
        )
    except Exception as e:
        REQUESTS_TOTAL.labels(status="failed").inc()
        raise HTTPException(status_code=503, detail=f"Failed to queue request: {str(e)}")


@router.get("/result/{request_id}", response_model=PredictionResult)
async def get_result(request: Request, request_id: str):
    """Poll for prediction result."""
    queue_service = request.app.state.queue_service
    result = await queue_service.get_result(request_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Request not found")

    return PredictionResult(**result)


@router.post("/predict/sync", response_model=PredictionResult)
async def predict_sync(request: Request, payload: PredictionRequest):
    """
    Synchronous prediction — queues and polls until result is ready.
    Simpler for testing but has higher latency.
    """
    import asyncio

    queue_service = request.app.state.queue_service
    request_id = await queue_service.enqueue(payload.text)
    REQUESTS_TOTAL.labels(status="queued").inc()

    # Poll for result (max 30 seconds)
    start = time.time()
    timeout = 30.0
    poll_interval = 0.05  # 50ms

    while time.time() - start < timeout:
        result = await queue_service.get_result(request_id)
        if result and result.get("status") in ("completed", "failed"):
            return PredictionResult(**result)
        await asyncio.sleep(poll_interval)
        # Back off slightly
        poll_interval = min(poll_interval * 1.2, 0.5)

    raise HTTPException(status_code=408, detail="Prediction timed out")


@router.post("/predict/batch")
async def predict_batch(request: Request, payload: BatchPredictionRequest):
    """Submit multiple texts for batch prediction."""
    queue_service = request.app.state.queue_service
    request_ids = []

    for text in payload.texts:
        request_id = await queue_service.enqueue(text)
        request_ids.append(request_id)
        REQUESTS_TOTAL.labels(status="queued").inc()

    return {
        "request_ids": request_ids,
        "count": len(request_ids),
        "message": f"{len(request_ids)} requests queued",
    }
