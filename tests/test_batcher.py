"""Tests for the dynamic batching engine."""

import time
import pytest
from worker.batcher import DynamicBatcher, BatchRequest


def _make_request(request_id: str = "test-1", text: str = "hello") -> BatchRequest:
    return BatchRequest(request_id=request_id, text=text, enqueued_at=time.time())


class TestDynamicBatcher:
    def test_empty_batcher_should_not_flush(self):
        batcher = DynamicBatcher(max_batch_size=4, batch_timeout_ms=50)
        assert not batcher.should_flush()
        assert batcher.flush() is None

    def test_flush_when_batch_full(self):
        batcher = DynamicBatcher(max_batch_size=3, batch_timeout_ms=5000)
        batcher.add(_make_request("r1", "text 1"))
        batcher.add(_make_request("r2", "text 2"))
        assert not batcher.should_flush()

        batcher.add(_make_request("r3", "text 3"))
        assert batcher.should_flush()

        batch = batcher.flush()
        assert batch is not None
        assert batch.size == 3
        assert batch.texts == ["text 1", "text 2", "text 3"]
        assert batch.request_ids == ["r1", "r2", "r3"]

    def test_flush_on_timeout(self):
        batcher = DynamicBatcher(max_batch_size=100, batch_timeout_ms=20)
        batcher.add(_make_request("r1"))
        assert not batcher.should_flush()

        time.sleep(0.025)  # 25ms > 20ms timeout
        assert batcher.should_flush()

        batch = batcher.flush()
        assert batch.size == 1

    def test_stats_tracking(self):
        batcher = DynamicBatcher(max_batch_size=2, batch_timeout_ms=5000)

        batcher.add(_make_request("r1"))
        batcher.add(_make_request("r2"))
        batcher.flush()

        batcher.add(_make_request("r3"))
        batcher.add(_make_request("r4"))
        batcher.flush()

        assert batcher.total_batches == 2
        assert batcher.total_requests == 4
        assert batcher.avg_batch_size == 2.0

    def test_pending_count(self):
        batcher = DynamicBatcher(max_batch_size=10, batch_timeout_ms=5000)
        assert batcher.pending_count == 0

        batcher.add(_make_request("r1"))
        batcher.add(_make_request("r2"))
        assert batcher.pending_count == 2

        batcher.flush()
        assert batcher.pending_count == 0

    def test_batch_resets_after_flush(self):
        batcher = DynamicBatcher(max_batch_size=2, batch_timeout_ms=5000)
        batcher.add(_make_request("r1"))
        batcher.add(_make_request("r2"))
        batcher.flush()

        assert not batcher.should_flush()
        assert batcher.pending_count == 0
