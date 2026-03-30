"""Health check endpoint."""

import time
from fastapi import APIRouter, Request
from api.schemas.models import HealthResponse

router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Check service health."""
    redis_ok = False
    queue_depth = 0

    try:
        await request.app.state.redis.ping()
        redis_ok = True
        queue_depth = await request.app.state.queue_service.get_queue_depth()
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if redis_ok else "degraded",
        redis_connected=redis_ok,
        model_loaded=True,  # Model loads in worker, not API
        queue_depth=queue_depth,
        uptime_seconds=round(time.time() - _start_time, 2),
    )
