"""Cloud ML Inference Platform — FastAPI Gateway."""

import os
import time
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from api.routes.predict import router as predict_router
from api.routes.health import router as health_router
from api.routes.metrics import router as metrics_router
from api.services.queue import QueueService

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Startup
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))

    logger.info("connecting_to_redis", host=redis_host, port=redis_port)

    app.state.redis = aioredis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True,
    )

    # Verify Redis connection
    try:
        await app.state.redis.ping()
        logger.info("redis_connected")
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        raise

    # Initialize queue service
    stream_key = os.getenv("REDIS_STREAM_KEY", "inference:stream")
    group_name = os.getenv("REDIS_CONSUMER_GROUP", "inference-workers")
    app.state.queue_service = QueueService(
        redis=app.state.redis,
        stream_key=stream_key,
        group_name=group_name,
    )
    await app.state.queue_service.initialize()

    logger.info("api_started", port=os.getenv("APP_PORT", 8000))
    yield

    # Shutdown
    await app.state.redis.close()
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Cloud ML Inference Platform",
        description="Production-grade ML inference with queue-backed batching and autoscaling",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus instrumentation
    Instrumentator(
        should_group_status_codes=True,
        should_group_untemplated=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/prom-metrics")

    # Routes
    app.include_router(predict_router, prefix="/api/v1", tags=["Inference"])
    app.include_router(health_router, tags=["Health"])
    app.include_router(metrics_router, tags=["Metrics"])

    return app


app = create_app()
