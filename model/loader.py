"""Model loading and inference."""

import os
import time
from typing import Optional

import structlog
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = structlog.get_logger()


class InferenceEngine:
    """Manages model loading and inference."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", "./model/cache")
        self.model = None
        self.tokenizer = None
        self.labels = None
        self._loaded = False

    def load(self):
        """Load model and tokenizer."""
        logger.info("loading_model", model=self.model_name, device=self.device)
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model.to(self.device)
        self.model.eval()

        # Get label mapping
        self.labels = self.model.config.id2label

        elapsed = time.time() - start
        logger.info("model_loaded", elapsed_s=round(elapsed, 2), labels=self.labels)
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Run inference on a batch of texts.

        Returns list of {label, score} dicts.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start = time.time()

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process outputs
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = []

        for i in range(len(texts)):
            score, idx = torch.max(probs[i], dim=0)
            predictions.append({
                "label": self.labels[idx.item()],
                "score": round(score.item(), 4),
            })

        inference_time = time.time() - start
        logger.debug(
            "batch_inference",
            batch_size=len(texts),
            inference_ms=round(inference_time * 1000, 2),
        )

        return predictions, inference_time

    def predict_single(self, text: str) -> dict:
        """Run inference on a single text."""
        results, inference_time = self.predict_batch([text])
        return results[0], inference_time
