"""Inference Worker — consumes from Redis Stream, batches, runs model."""

import json
import os
import signal
import sys
import time
from typing import Optional

import redis
import structlog

from model.loader import InferenceEngine
from worker.batcher import DynamicBatcher, BatchRequest
from api.services.metrics import (
    REQUESTS_TOTAL,
    BATCH_SIZE,
    BATCHES_PROCESSED,
    INFERENCE_LATENCY,
    QUEUE_WAIT_TIME,
    REQUEST_LATENCY,
    ACTIVE_WORKERS,
    ERRORS_TOTAL,
    REQUESTS_IN_QUEUE,
)

logger = structlog.get_logger()


class InferenceWorker:
    """
    Main worker process.

    Loop:
    1. Read messages from Redis Stream (consumer group)
    2. Add to batcher
    3. When batch is ready (full or timeout), run inference
    4. Store results back in Redis
    5. Acknowledge processed messages
    """

    def __init__(
        self,
        worker_id: str = "worker-0",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        stream_key: str = "inference:stream",
        group_name: str = "inference-workers",
        result_prefix: str = "inference:result:",
        max_batch_size: int = 16,
        batch_timeout_ms: float = 50.0,
        poll_interval_ms: float = 10.0,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = "cpu",
    ):
        self.worker_id = worker_id
        self.stream_key = stream_key
        self.group_name = group_name
        self.result_prefix = result_prefix
        self.poll_interval_ms = poll_interval_ms

        # Redis connection (sync for worker)
        self.redis = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )

        # Batching engine
        self.batcher = DynamicBatcher(
            max_batch_size=max_batch_size,
            batch_timeout_ms=batch_timeout_ms,
        )

        # Model
        self.engine = InferenceEngine(model_name=model_name, device=device)

        # State
        self._running = False
        self._message_ids: dict[str, str] = {}  # request_id → stream message_id

    def setup(self):
        """Initialize worker: load model, create consumer group."""
        logger.info("worker_setup", worker_id=self.worker_id)

        # Load model
        self.engine.load()

        # Ensure consumer group exists
        try:
            self.redis.xgroup_create(
                self.stream_key, self.group_name, id="0", mkstream=True
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        ACTIVE_WORKERS.inc()
        logger.info("worker_ready", worker_id=self.worker_id)

    def run(self):
        """Main worker loop."""
        self._running = True
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

        self.setup()
        logger.info("worker_started", worker_id=self.worker_id)

        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error("worker_tick_error", error=str(e), worker_id=self.worker_id)
                ERRORS_TOTAL.labels(error_type="worker_tick").inc()
                time.sleep(0.1)

        ACTIVE_WORKERS.dec()
        logger.info("worker_stopped", worker_id=self.worker_id)

    def _tick(self):
        """One iteration of the worker loop."""
        # 1. Read new messages from stream
        messages = self.redis.xreadgroup(
            self.group_name,
            self.worker_id,
            {self.stream_key: ">"},
            count=self.batcher.max_batch_size,
            block=int(self.poll_interval_ms),
        )

        # 2. Add messages to batcher
        if messages:
            for stream_name, entries in messages:
                for msg_id, fields in entries:
                    try:
                        data = json.loads(fields["data"])
                        request = BatchRequest(
                            request_id=data["request_id"],
                            text=data["text"],
                            enqueued_at=data["enqueued_at"],
                        )
                        self.batcher.add(request)
                        self._message_ids[data["request_id"]] = msg_id
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error("invalid_message", msg_id=msg_id, error=str(e))
                        # Acknowledge bad message to prevent reprocessing
                        self.redis.xack(self.stream_key, self.group_name, msg_id)

        # 3. Check if batch should be flushed
        if self.batcher.should_flush():
            batch = self.batcher.flush()
            if batch:
                self._process_batch(batch)

    def _process_batch(self, batch):
        """Run inference on a batch and store results."""
        batch_start = time.time()

        BATCH_SIZE.observe(batch.size)
        BATCHES_PROCESSED.inc()

        try:
            # Run inference
            predictions, inference_time = self.engine.predict_batch(batch.texts)
            INFERENCE_LATENCY.observe(inference_time)

            # Store results and acknowledge
            for i, request in enumerate(batch.requests):
                now = time.time()
                total_latency = now - request.enqueued_at
                queue_wait = request.added_to_batch_at - request.enqueued_at

                result = {
                    "request_id": request.request_id,
                    "status": "completed",
                    "label": predictions[i]["label"],
                    "score": predictions[i]["score"],
                    "latency_ms": round(total_latency * 1000, 2),
                    "batch_size": batch.size,
                }

                # Store result
                self.redis.set(
                    f"{self.result_prefix}{request.request_id}",
                    json.dumps(result),
                    ex=3600,
                )

                # Metrics
                REQUEST_LATENCY.observe(total_latency)
                QUEUE_WAIT_TIME.observe(queue_wait)
                REQUESTS_TOTAL.labels(status="completed").inc()

                # Acknowledge message
                msg_id = self._message_ids.pop(request.request_id, None)
                if msg_id:
                    self.redis.xack(self.stream_key, self.group_name, msg_id)

            logger.info(
                "batch_completed",
                batch_size=batch.size,
                inference_ms=round(inference_time * 1000, 2),
                total_ms=round((time.time() - batch_start) * 1000, 2),
                worker_id=self.worker_id,
            )

        except Exception as e:
            logger.error("batch_failed", error=str(e), batch_size=batch.size)
            ERRORS_TOTAL.labels(error_type="inference").inc()

            # Mark all requests as failed
            for request in batch.requests:
                result = {
                    "request_id": request.request_id,
                    "status": "failed",
                    "error": str(e),
                }
                self.redis.set(
                    f"{self.result_prefix}{request.request_id}",
                    json.dumps(result),
                    ex=3600,
                )
                REQUESTS_TOTAL.labels(status="failed").inc()

                msg_id = self._message_ids.pop(request.request_id, None)
                if msg_id:
                    self.redis.xack(self.stream_key, self.group_name, msg_id)

    def _shutdown(self, signum, frame):
        """Graceful shutdown."""
        logger.info("worker_shutting_down", worker_id=self.worker_id, signal=signum)
        self._running = False


def main():
    """Entry point for worker process."""
    worker_id = os.getenv("WORKER_ID", "worker-0")
    worker = InferenceWorker(
        worker_id=worker_id,
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", 6379)),
        stream_key=os.getenv("REDIS_STREAM_KEY", "inference:stream"),
        group_name=os.getenv("REDIS_CONSUMER_GROUP", "inference-workers"),
        max_batch_size=int(os.getenv("BATCH_MAX_SIZE", 16)),
        batch_timeout_ms=float(os.getenv("BATCH_TIMEOUT_MS", 50)),
        poll_interval_ms=float(os.getenv("WORKER_POLL_INTERVAL_MS", 10)),
        model_name=os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"),
        device=os.getenv("MODEL_DEVICE", "cpu"),
    )
    worker.run()


if __name__ == "__main__":
    main()
