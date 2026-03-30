#!/usr/bin/env python3
"""
Autoscaling Simulator — scales worker processes based on Redis queue depth.

This simulates what KEDA/HPA does on Kubernetes, but locally using
subprocess management. Useful for demos and benchmarks without a cluster.

Usage:
    python scripts/autoscaler.py
    python scripts/autoscaler.py --min-workers 1 --max-workers 8 --scale-threshold 20

How it works:
    1. Polls Redis stream pending count every N seconds
    2. If pending > threshold * current_workers → scale up
    3. If pending < threshold * 0.25 * current_workers → scale down
    4. Respects min/max bounds and cooldown period
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import redis


class AutoscalerSimulator:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        stream_key: str = "inference:stream",
        group_name: str = "inference-workers",
        min_workers: int = 1,
        max_workers: int = 8,
        scale_threshold: int = 20,
        poll_interval: int = 5,
        cooldown: int = 15,
    ):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.stream_key = stream_key
        self.group_name = group_name
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_threshold = scale_threshold
        self.poll_interval = poll_interval
        self.cooldown = cooldown

        self.workers: dict[str, subprocess.Popen] = {}
        self.last_scale_time = 0
        self._running = False

    def get_pending_count(self) -> int:
        """Get number of pending messages in the consumer group."""
        try:
            info = self.redis.xinfo_groups(self.stream_key)
            for group in info:
                if group.get("name") == self.group_name:
                    return group.get("pending", 0)
        except Exception:
            pass
        return 0

    def get_stream_length(self) -> int:
        try:
            return self.redis.xlen(self.stream_key)
        except Exception:
            return 0

    def spawn_worker(self, worker_id: str):
        """Spawn a new worker process."""
        env = os.environ.copy()
        env["WORKER_ID"] = worker_id
        env["REDIS_HOST"] = "localhost"
        env["REDIS_PORT"] = "6379"

        proc = subprocess.Popen(
            [sys.executable, "-m", "worker.consumer"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.workers[worker_id] = proc
        self._log(f"SCALE UP: Spawned {worker_id} (pid={proc.pid}). Total: {len(self.workers)}")

    def kill_worker(self, worker_id: str):
        """Kill a worker process."""
        proc = self.workers.pop(worker_id, None)
        if proc:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            self._log(f"SCALE DOWN: Killed {worker_id}. Total: {len(self.workers)}")

    def desired_workers(self, pending: int) -> int:
        """Calculate desired worker count based on pending messages."""
        if pending == 0:
            return self.min_workers

        # 1 worker per `scale_threshold` pending messages
        desired = max(self.min_workers, pending // self.scale_threshold + 1)
        return min(desired, self.max_workers)

    def scale(self):
        """Check and adjust worker count."""
        now = time.time()
        if now - self.last_scale_time < self.cooldown:
            return

        pending = self.get_pending_count()
        stream_len = self.get_stream_length()
        current = len(self.workers)
        desired = self.desired_workers(pending + stream_len)

        self._log(
            f"CHECK: pending={pending}, stream_len={stream_len}, "
            f"current_workers={current}, desired={desired}"
        )

        if desired > current:
            # Scale up
            for i in range(desired - current):
                wid = f"autoscale-worker-{current + i}"
                self.spawn_worker(wid)
            self.last_scale_time = now

        elif desired < current:
            # Scale down (remove most recently added)
            workers_to_remove = sorted(self.workers.keys(), reverse=True)
            for wid in workers_to_remove[: current - desired]:
                self.kill_worker(wid)
            self.last_scale_time = now

    def run(self):
        """Main autoscaler loop."""
        self._running = True
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

        self._log(f"Autoscaler started. Min={self.min_workers}, Max={self.max_workers}, "
                  f"Threshold={self.scale_threshold}")

        # Spawn minimum workers
        for i in range(self.min_workers):
            self.spawn_worker(f"autoscale-worker-{i}")

        while self._running:
            # Clean up dead workers
            dead = [wid for wid, proc in self.workers.items() if proc.poll() is not None]
            for wid in dead:
                self._log(f"Worker {wid} died, removing from pool")
                del self.workers[wid]

            self.scale()
            time.sleep(self.poll_interval)

        # Cleanup
        self._log("Shutting down all workers...")
        for wid in list(self.workers.keys()):
            self.kill_worker(wid)

    def _shutdown(self, signum, frame):
        self._running = False

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [autoscaler] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Worker Autoscaler Simulator")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--scale-threshold", type=int, default=20,
                        help="Pending messages per worker before scaling up")
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--cooldown", type=int, default=15)
    args = parser.parse_args()

    scaler = AutoscalerSimulator(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        scale_threshold=args.scale_threshold,
        poll_interval=args.poll_interval,
        cooldown=args.cooldown,
    )
    scaler.run()


if __name__ == "__main__":
    main()
