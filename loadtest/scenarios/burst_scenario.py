"""Burst traffic scenario — simulates sudden spike then steady state.

Usage:
    locust -f loadtest/scenarios/burst_scenario.py --host http://localhost:8000 \
        --users 500 --spawn-rate 100 --run-time 90s --headless
"""

import random
import time
from locust import HttpUser, task, between, LoadTestShape

SAMPLE_TEXTS = [
    "This product is absolutely amazing, I love it!",
    "Terrible experience, would not recommend to anyone.",
    "Outstanding customer service, they went above and beyond.",
    "I'm very disappointed with the quality of this item.",
    "Best purchase I've made this year, highly recommend!",
    "The food was mediocre at best, overpriced for what you get.",
    "Incredible performance, exceeded all my expectations.",
    "Waste of money, broke after just two days of use.",
]


class BurstTrafficUser(HttpUser):
    """Fast-firing user for burst testing."""
    wait_time = between(0.01, 0.1)

    @task
    def predict(self):
        text = random.choice(SAMPLE_TEXTS)
        self.client.post(
            "/api/v1/predict/sync",
            json={"text": text},
            name="/predict/sync (burst)",
        )


class BurstShape(LoadTestShape):
    """
    Burst traffic pattern:
    - 0-10s:   ramp to 50 users (warm up)
    - 10-20s:  hold 50 users (baseline)
    - 20-30s:  spike to 500 users (burst!)
    - 30-50s:  hold 500 users (sustained burst)
    - 50-60s:  drop to 100 users (cool down)
    - 60-90s:  hold 100 users (recovery)
    """

    stages = [
        {"duration": 10, "users": 50, "spawn_rate": 10},
        {"duration": 20, "users": 50, "spawn_rate": 10},
        {"duration": 30, "users": 500, "spawn_rate": 100},
        {"duration": 50, "users": 500, "spawn_rate": 100},
        {"duration": 60, "users": 100, "spawn_rate": 50},
        {"duration": 90, "users": 100, "spawn_rate": 10},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])

        return None  # Stop test
