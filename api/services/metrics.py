"""Application metrics using Prometheus client."""

from prometheus_client import Counter, Histogram, Gauge, Summary

# Request metrics
REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["status"],  # queued, completed, failed
)

REQUESTS_IN_QUEUE = Gauge(
    "inference_queue_depth",
    "Current number of requests in the queue",
)

# Latency metrics
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "End-to-end request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

INFERENCE_LATENCY = Histogram(
    "inference_model_latency_seconds",
    "Model inference latency in seconds (excludes queue time)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

QUEUE_WAIT_TIME = Histogram(
    "inference_queue_wait_seconds",
    "Time spent waiting in queue",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# Batch metrics
BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Number of requests per batch",
    buckets=[1, 2, 4, 8, 16, 32, 64],
)

BATCHES_PROCESSED = Counter(
    "inference_batches_total",
    "Total number of batches processed",
)

# Worker metrics
ACTIVE_WORKERS = Gauge(
    "inference_active_workers",
    "Number of active worker instances",
)

WORKER_UTILIZATION = Gauge(
    "inference_worker_utilization",
    "Worker utilization (0-1)",
    ["worker_id"],
)

# Error metrics
ERRORS_TOTAL = Counter(
    "inference_errors_total",
    "Total inference errors",
    ["error_type"],
)
