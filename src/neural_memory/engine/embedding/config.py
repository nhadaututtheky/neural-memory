"""Embedding configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

_VALID_PROVIDERS = ("sentence_transformer", "openai", "")


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

    VALID_PROVIDERS: ClassVar[tuple[str, ...]] = _VALID_PROVIDERS

    enabled: bool = False
    provider: str = "sentence_transformer"
    model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    activation_boost: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0.0, 1.0], got {self.similarity_threshold}"
            )
        if self.activation_boost < 0.0:
            raise ValueError(f"activation_boost must be >= 0.0, got {self.activation_boost}")
        if self.provider not in _VALID_PROVIDERS:
            raise ValueError(f"provider must be one of {_VALID_PROVIDERS}, got {self.provider!r}")
