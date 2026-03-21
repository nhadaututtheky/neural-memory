"""Configuration for LLM-powered deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DedupConfig:
    """Configuration for the 3-tier dedup pipeline.

    All off by default to preserve zero-LLM core.

    Attributes:
        enabled: Master switch for dedup during encoding.
        simhash_threshold: Max Hamming distance for SimHash match (Tier 1).
        embedding_threshold: Cosine similarity above which = definite duplicate (Tier 2).
        embedding_ambiguous_low: Cosine similarity below which = not duplicate (Tier 2).
        llm_enabled: Whether to use LLM for borderline cases (Tier 3).
        llm_provider: LLM provider for Tier 3 ("openai" | "anthropic" | "none").
        llm_model: Model name for Tier 3.
        llm_max_pairs_per_encode: Max LLM calls per encode operation.
        merge_strategy: How to handle duplicates ("keep_newer" | "keep_older" | "merge_metadata").
        max_candidates: Max candidates to fetch from storage for comparison.
    """

    enabled: bool = True
    simhash_threshold: int = 7  # tighter: ~89% similarity (was 10 / ~85%)
    embedding_threshold: float = 0.85
    embedding_ambiguous_low: float = 0.75
    llm_enabled: bool = False
    llm_provider: str = "none"
    llm_model: str = ""
    llm_max_pairs_per_encode: int = 3
    merge_strategy: str = "keep_newer"
    max_candidates: int = 30  # wider search (was 10)

    _VALID_PROVIDERS: tuple[str, ...] = ("none", "openai", "anthropic")
    _VALID_STRATEGIES: tuple[str, ...] = ("keep_newer", "keep_older", "merge_metadata")

    def __post_init__(self) -> None:
        if not 0.0 <= self.embedding_ambiguous_low <= 1.0:
            raise ValueError(
                f"embedding_ambiguous_low must be in [0.0, 1.0], got {self.embedding_ambiguous_low}"
            )
        if not 0.0 <= self.embedding_threshold <= 1.0:
            raise ValueError(
                f"embedding_threshold must be in [0.0, 1.0], got {self.embedding_threshold}"
            )
        if self.embedding_ambiguous_low > self.embedding_threshold:
            raise ValueError(
                f"embedding_ambiguous_low ({self.embedding_ambiguous_low}) "
                f"must be <= embedding_threshold ({self.embedding_threshold})"
            )
        if not 0 <= self.simhash_threshold <= 64:
            raise ValueError(f"simhash_threshold must be in [0, 64], got {self.simhash_threshold}")
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {self.max_candidates}")
        if self.llm_max_pairs_per_encode < 1:
            raise ValueError(
                f"llm_max_pairs_per_encode must be >= 1, got {self.llm_max_pairs_per_encode}"
            )
        if self.llm_provider not in self._VALID_PROVIDERS:
            raise ValueError(
                f"llm_provider must be one of {self._VALID_PROVIDERS}, got '{self.llm_provider}'"
            )
        if self.merge_strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"merge_strategy must be one of {self._VALID_STRATEGIES}, "
                f"got '{self.merge_strategy}'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "simhash_threshold": self.simhash_threshold,
            "embedding_threshold": self.embedding_threshold,
            "embedding_ambiguous_low": self.embedding_ambiguous_low,
            "llm_enabled": self.llm_enabled,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_max_pairs_per_encode": self.llm_max_pairs_per_encode,
            "merge_strategy": self.merge_strategy,
            "max_candidates": self.max_candidates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DedupConfig:
        try:
            return cls(
                enabled=bool(data.get("enabled", True)),
                simhash_threshold=int(data.get("simhash_threshold", 7)),
                embedding_threshold=float(data.get("embedding_threshold", 0.85)),
                embedding_ambiguous_low=float(data.get("embedding_ambiguous_low", 0.75)),
                llm_enabled=bool(data.get("llm_enabled", False)),
                llm_provider=str(data.get("llm_provider", "none")),
                llm_model=str(data.get("llm_model", "")),
                llm_max_pairs_per_encode=int(data.get("llm_max_pairs_per_encode", 3)),
                merge_strategy=str(data.get("merge_strategy", "keep_newer")),
                max_candidates=int(data.get("max_candidates", 30)),
            )
        except (ValueError, TypeError):
            return cls()  # Fall back to safe defaults
