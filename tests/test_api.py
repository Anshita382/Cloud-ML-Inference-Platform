"""Tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.xadd = AsyncMock(return_value="1-0")
    redis.xgroup_create = AsyncMock()
    redis.xinfo_groups = AsyncMock(return_value=[{"name": "inference-workers", "pending": 0}])
    redis.xlen = AsyncMock(return_value=0)
    redis.set = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.close = AsyncMock()
    return redis


@pytest.fixture
def client(mock_redis):
    """Create test client with mocked Redis."""
    from api.main import create_app
    from api.services.queue import QueueService

    app = create_app()

    # Inject mocks
    app.state.redis = mock_redis
    app.state.queue_service = QueueService(redis=mock_redis)

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client, mock_redis):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["redis_connected"] is True

    def test_health_degraded_when_redis_down(self, client, mock_redis):
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection refused"))
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"


class TestPredictEndpoint:
    def test_predict_returns_request_id(self, client, mock_redis):
        response = client.post(
            "/api/v1/predict",
            json={"text": "This is a great product!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert data["status"] == "queued"

    def test_predict_empty_text_rejected(self, client):
        response = client.post(
            "/api/v1/predict",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_predict_missing_text_rejected(self, client):
        response = client.post(
            "/api/v1/predict",
            json={},
        )
        assert response.status_code == 422


class TestResultEndpoint:
    def test_result_not_found(self, client, mock_redis):
        mock_redis.get = AsyncMock(return_value=None)
        response = client.get("/api/v1/result/nonexistent-id")
        assert response.status_code == 404

    def test_result_found(self, client, mock_redis):
        import json
        result = {
            "request_id": "test-123",
            "status": "completed",
            "label": "POSITIVE",
            "score": 0.98,
            "latency_ms": 45.2,
            "batch_size": 8,
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(result))

        response = client.get("/api/v1/result/test-123")
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "POSITIVE"
        assert data["score"] == 0.98


class TestBatchEndpoint:
    def test_batch_predict(self, client, mock_redis):
        response = client.post(
            "/api/v1/predict/batch",
            json={"texts": ["text one", "text two", "text three"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert len(data["request_ids"]) == 3
