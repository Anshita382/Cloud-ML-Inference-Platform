"""Dynamic Batching Engine.

Collects incoming requests over a time window and merges them
into a single batch inference call. This is the core optimization
that turns per-request overhead into amortized batch throughput.

Key parameters:
- max_batch_size: Maximum requests per batch (default 16)
- batch_timeout_ms: Max time to wait for a full batch (default 50ms)

Tradeoff:
- Larger batches → higher throughput, higher per-request latency
- Smaller batches → lower latency, lower throughput
- The timeout prevents starvation (don't wait forever for a full batch)
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger()


@dataclass
class BatchRequest:
    """A single request waiting to be batched."""
    request_id: str
    text: str
    enqueued_at: float
    added_to_batch_at: float = field(default_factory=time.time)


@dataclass
class Batch:
    """A batch of requests ready for inference."""
    requests: list[BatchRequest]
    created_at: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def texts(self) -> list[str]:
        return [r.text for r in self.requests]

    @property
    def request_ids(self) -> list[str]:
        return [r.request_id for r in self.requests]


class DynamicBatcher:
    """
    Aggregates individual requests into batches.

    Flush conditions (whichever comes first):
    1. Batch reaches max_batch_size
    2. Oldest request in batch has waited batch_timeout_ms
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        batch_timeout_ms: float = 50.0,
    ):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._pending: list[BatchRequest] = []
        self._batch_start: Optional[float] = None

        # Stats
        self.total_batches = 0
        self.total_requests = 0

    def add(self, request: BatchRequest):
        """Add a request to the current batch."""
        self._pending.append(request)
        if self._batch_start is None:
            self._batch_start = time.time()

    def should_flush(self) -> bool:
        """Check if the current batch should be flushed."""
        if not self._pending:
            return False

        # Condition 1: batch is full
        if len(self._pending) >= self.max_batch_size:
            return True

        # Condition 2: timeout reached
        if self._batch_start is not None:
            elapsed_ms = (time.time() - self._batch_start) * 1000
            if elapsed_ms >= self.batch_timeout_ms:
                return True

        return False

    def flush(self) -> Optional[Batch]:
        """Flush the current batch and return it."""
        if not self._pending:
            return None

        batch = Batch(requests=list(self._pending))

        # Stats
        self.total_batches += 1
        self.total_requests += batch.size

        logger.info(
            "batch_flushed",
            batch_size=batch.size,
            max_batch_size=self.max_batch_size,
            wait_ms=round((time.time() - self._batch_start) * 1000, 2)
            if self._batch_start
            else 0,
        )

        # Reset
        self._pending = []
        self._batch_start = None

        return batch

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def avg_batch_size(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return self.total_requests / self.total_batches
