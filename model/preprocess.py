"""Input preprocessing utilities."""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """Basic text cleaning for inference input."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove null bytes
    text = text.replace("\x00", "")
    # Truncate extremely long inputs (tokenizer handles this too, but saves memory)
    if len(text) > 10000:
        text = text[:10000]
    return text


def validate_text(text: str) -> Optional[str]:
    """Validate text input. Returns error message or None if valid."""
    if not text or not text.strip():
        return "Text cannot be empty"
    if len(text) > 50000:
        return "Text exceeds maximum length (50000 characters)"
    return None
