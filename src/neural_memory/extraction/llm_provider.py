"""Optional LLM extraction provider for enhanced relation/entity extraction.

OFF by default — regex stays primary. LLM deduplicates against regex results
and failure falls back silently.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for LLM-based extraction."""

    enabled: bool = False
    provider: str = "none"  # "openai" | "anthropic" | "none"
    model: str = ""
    fallback_to_regex: bool = True

    def __post_init__(self) -> None:
        if self.enabled:
            if self.provider == "none":
                raise ValueError("provider must not be 'none' when extraction is enabled")
            if not self.model:
                raise ValueError("model must not be empty when extraction is enabled")


@dataclass(frozen=True)
class RelationCandidate:
    """A relation extracted by LLM."""

    source: str
    relation_type: str
    target: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ExtractionProvider(ABC):
    """Abstract base for LLM extraction providers."""

    @abstractmethod
    async def extract_relations(self, text: str, language: str = "auto") -> list[RelationCandidate]:
        """Extract relations from text using LLM.

        Args:
            text: Input text to analyze
            language: Language hint

        Returns:
            List of extracted relation candidates
        """
        ...

    @abstractmethod
    async def extract_entities(self, text: str, language: str = "auto") -> list[dict[str, Any]]:
        """Extract entities from text using LLM.

        Args:
            text: Input text to analyze
            language: Language hint

        Returns:
            List of entity dicts with keys: text, type, confidence
        """
        ...


def deduplicate_relations(
    regex_relations: list[dict[str, Any]],
    llm_relations: list[RelationCandidate],
) -> list[RelationCandidate]:
    """Deduplicate LLM relations against regex results.

    Returns only LLM relations that don't overlap with regex results.
    Overlap is determined by matching source+target (case-insensitive).
    """
    existing_pairs: set[tuple[str, str]] = set()
    for rel in regex_relations:
        src = rel.get("source", "").lower().strip()
        tgt = rel.get("target", "").lower().strip()
        if src and tgt:
            existing_pairs.add((src, tgt))

    new_relations: list[RelationCandidate] = []
    for llm_rel in llm_relations:
        pair = (llm_rel.source.lower().strip(), llm_rel.target.lower().strip())
        if pair not in existing_pairs:
            new_relations.append(llm_rel)
            existing_pairs.add(pair)

    return new_relations
