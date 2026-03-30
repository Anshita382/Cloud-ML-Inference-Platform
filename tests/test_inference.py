"""Tests for model inference engine.

These tests download the model on first run (~250MB).
Skip with: pytest -m "not slow"
"""

import pytest
import os

# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine():
    """Load the inference engine once for all tests."""
    from model.loader import InferenceEngine

    eng = InferenceEngine(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        device="cpu",
    )
    eng.load()
    return eng


class TestInferenceEngine:
    def test_model_loads(self, engine):
        assert engine.is_loaded

    def test_single_prediction_positive(self, engine):
        result, latency = engine.predict_single("This is absolutely wonderful!")
        assert result["label"] == "POSITIVE"
        assert result["score"] > 0.9
        assert latency > 0

    def test_single_prediction_negative(self, engine):
        result, latency = engine.predict_single("This is terrible and awful.")
        assert result["label"] == "NEGATIVE"
        assert result["score"] > 0.9

    def test_batch_prediction(self, engine):
        texts = [
            "Great product, love it!",
            "Terrible service, never again.",
            "It was okay, nothing special.",
        ]
        results, latency = engine.predict_batch(texts)
        assert len(results) == 3
        assert results[0]["label"] == "POSITIVE"
        assert results[1]["label"] == "NEGATIVE"
        assert latency > 0

    def test_batch_size_one(self, engine):
        results, _ = engine.predict_batch(["Single input"])
        assert len(results) == 1

    def test_long_text_truncated(self, engine):
        long_text = "This is great! " * 500  # Very long text
        result, _ = engine.predict_single(long_text)
        assert result["label"] in ("POSITIVE", "NEGATIVE")
        assert 0 <= result["score"] <= 1

    def test_batch_latency_vs_single(self, engine):
        """Batch inference should be more efficient per-item than single."""
        import time

        text = "This product is amazing, highly recommended!"
        texts = [text] * 16

        # Single predictions
        start = time.time()
        for t in texts:
            engine.predict_single(t)
        single_time = time.time() - start

        # Batch prediction
        start = time.time()
        engine.predict_batch(texts)
        batch_time = time.time() - start

        # Batch should be faster (or at least not slower by much)
        print(f"Single (16x): {single_time:.3f}s, Batch (16): {batch_time:.3f}s")
        print(f"Speedup: {single_time / batch_time:.1f}x")
        # Don't assert exact speedup — depends on hardware
