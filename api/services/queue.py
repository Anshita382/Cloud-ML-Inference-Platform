"""Redis Streams queue service."""

import json
import time
import uuid
from typing import Optional

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger()


class QueueService:
    """Manages request queue using Redis Streams."""

    def __init__(
        self,
        redis: aioredis.Redis,
        stream_key: str = "inference:stream",
        group_name: str = "inference-workers",
        result_prefix: str = "inference:result:",
        result_ttl: int = 3600,
    ):
        self.redis = redis
        self.stream_key = stream_key
        self.group_name = group_name
        self.result_prefix = result_prefix
        self.result_ttl = result_ttl

    async def initialize(self):
        """Create consumer group if it doesn't exist."""
        try:
            await self.redis.xgroup_create(
                self.stream_key, self.group_name, id="0", mkstream=True
            )
            logger.info("consumer_group_created", group=self.group_name)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info("consumer_group_exists", group=self.group_name)
            else:
                raise

    async def enqueue(self, text: str) -> str:
        """Add a prediction request to the queue. Returns request_id."""
        request_id = str(uuid.uuid4())
        payload = json.dumps({
            "request_id": request_id,
            "text": text,
            "enqueued_at": time.time(),
        })

        await self.redis.xadd(self.stream_key, {"data": payload})

        # Set initial status
        await self.redis.set(
            f"{self.result_prefix}{request_id}",
            json.dumps({"status": "queued", "request_id": request_id}),
            ex=self.result_ttl,
        )

        logger.info("request_enqueued", request_id=request_id)
        return request_id

    async def get_result(self, request_id: str) -> Optional[dict]:
        """Get the result for a request."""
        data = await self.redis.get(f"{self.result_prefix}{request_id}")
        if data:
            return json.loads(data)
        return None

    async def store_result(self, request_id: str, result: dict):
        """Store inference result."""
        await self.redis.set(
            f"{self.result_prefix}{request_id}",
            json.dumps(result),
            ex=self.result_ttl,
        )

    async def get_queue_depth(self) -> int:
        """Get current number of pending messages."""
        try:
            info = await self.redis.xinfo_groups(self.stream_key)
            for group in info:
                if group.get("name") == self.group_name:
                    return group.get("pending", 0)
            # If group not found, count total stream length
            return await self.redis.xlen(self.stream_key)
        except Exception:
            return 0

    async def get_stream_length(self) -> int:
        """Get total stream length."""
        try:
            return await self.redis.xlen(self.stream_key)
        except Exception:
            return 0
