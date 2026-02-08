"""Embedding configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the embedding layer.

    Attributes:
        enabled: Whether embedding is active
        provider: Provider name ("sentence_transformer" or "openai")
        model: Model name/identifier
        similarity_threshold: Minimum cosine similarity for anchor matching
        activation_boost: Boost applied to embedding-matched anchors
    """

    enabled: bool = False
    provider: str = "sentence_transformer"
    model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    activation_boost: float = 0.15
