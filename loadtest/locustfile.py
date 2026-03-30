"""Locust load test for ML Inference Platform.

Usage:
    # Web UI (recommended)
    locust -f loadtest/locustfile.py --host http://localhost:8000

    # Headless — 100 users, 60 seconds
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --users 100 --spawn-rate 10 --run-time 60s --headless

    # Heavy load — 1000 users
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --users 1000 --spawn-rate 50 --run-time 120s --headless
"""

import random
import time
from locust import HttpUser, task, between, events

# Sample texts for classification (mix of positive/negative sentiments)
SAMPLE_TEXTS = [
    "This product is absolutely amazing, I love it!",
    "Terrible experience, would not recommend to anyone.",
    "The movie was okay, nothing special but not bad either.",
    "Outstanding customer service, they went above and beyond.",
    "I'm very disappointed with the quality of this item.",
    "Best purchase I've made this year, highly recommend!",
    "The food was mediocre at best, overpriced for what you get.",
    "Incredible performance, exceeded all my expectations.",
    "Waste of money, broke after just two days of use.",
    "A decent product for the price, gets the job done.",
    "The team delivered exceptional results this quarter.",
    "This software has too many bugs to be useful.",
    "What a wonderful experience at the restaurant last night!",
    "I regret buying this, completely useless product.",
    "Solid build quality and great design, very satisfied.",
    "The worst customer support I have ever dealt with.",
    "Fantastic book, couldn't put it down once I started.",
    "Not worth the hype, very average in every way.",
    "This course completely changed my career trajectory.",
    "Extremely slow shipping and the package arrived damaged.",
]


class InferenceUser(HttpUser):
    """Simulates a client sending prediction requests."""

    wait_time = between(0.1, 0.5)

    @task(7)
    def predict_sync(self):
        """Send a synchronous prediction (queue + poll)."""
        text = random.choice(SAMPLE_TEXTS)
        with self.client.post(
            "/api/v1/predict/sync",
            json={"text": text},
            catch_response=True,
            name="/predict/sync",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "completed":
                    response.success()
                else:
                    response.failure(f"Unexpected status: {data.get('status')}")
            elif response.status_code == 408:
                response.failure("Timeout")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def predict_async(self):
        """Send an async prediction and poll for result."""
        text = random.choice(SAMPLE_TEXTS)

        # Submit
        submit_resp = self.client.post(
            "/api/v1/predict",
            json={"text": text},
            name="/predict (submit)",
        )

        if submit_resp.status_code != 200:
            return

        request_id = submit_resp.json().get("request_id")
        if not request_id:
            return

        # Poll for result (max 10 attempts)
        for _ in range(10):
            time.sleep(0.1)
            with self.client.get(
                f"/api/v1/result/{request_id}",
                catch_response=True,
                name="/result/{id} (poll)",
            ) as poll_resp:
                if poll_resp.status_code == 200:
                    data = poll_resp.json()
                    if data.get("status") == "completed":
                        poll_resp.success()
                        return
                    elif data.get("status") == "failed":
                        poll_resp.failure("Inference failed")
                        return
                    # Still queued/processing — keep polling
                    poll_resp.success()

    @task(1)
    def health_check(self):
        """Periodic health check."""
        self.client.get("/health", name="/health")


class BurstUser(HttpUser):
    """Simulates burst traffic — sends requests as fast as possible."""

    wait_time = between(0.01, 0.05)

    @task
    def predict_burst(self):
        text = random.choice(SAMPLE_TEXTS)
        self.client.post(
            "/api/v1/predict/sync",
            json={"text": text},
            name="/predict/sync (burst)",
        )
