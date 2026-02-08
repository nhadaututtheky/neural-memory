"""Tests for LLM extraction provider."""

from __future__ import annotations

from typing import Any

import pytest

from neural_memory.extraction.llm_provider import (
    ExtractionConfig,
    ExtractionProvider,
    RelationCandidate,
    deduplicate_relations,
)


# ── Mock provider ────────────────────────────────────────────────


class MockExtractionProvider(ExtractionProvider):
    """Mock LLM extraction provider for testing."""

    def __init__(self, relations: list[RelationCandidate] | None = None) -> None:
        self._relations = relations or []

    async def extract_relations(
        self,
        text: str,
        language: str = "auto",
    ) -> list[RelationCandidate]:
        return self._relations

    async def extract_entities(
        self,
        text: str,
        language: str = "auto",
    ) -> list[dict[str, Any]]:
        return [{"text": "mock-entity", "type": "ENTITY", "confidence": 0.9}]


# ── ExtractionConfig tests ──────────────────────────────────────


class TestExtractionConfig:
    """Test ExtractionConfig dataclass."""

    def test_defaults(self) -> None:
        """Config should be disabled by default."""
        config = ExtractionConfig()
        assert config.enabled is False
        assert config.provider == "none"
        assert config.model == ""
        assert config.fallback_to_regex is True

    def test_frozen(self) -> None:
        """Config should be immutable."""
        config = ExtractionConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore[misc]

    def test_custom_config(self) -> None:
        """Should support custom configuration."""
        config = ExtractionConfig(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            fallback_to_regex=False,
        )
        assert config.enabled is True
        assert config.provider == "openai"
        assert config.fallback_to_regex is False


# ── RelationCandidate tests ─────────────────────────────────────


class TestRelationCandidate:
    """Test RelationCandidate dataclass."""

    def test_creation(self) -> None:
        """Should create with required fields."""
        rel = RelationCandidate(source="Redis", relation_type="is_a", target="database")
        assert rel.source == "Redis"
        assert rel.relation_type == "is_a"
        assert rel.target == "database"
        assert rel.confidence == 1.0

    def test_frozen(self) -> None:
        """Should be immutable."""
        rel = RelationCandidate(source="A", relation_type="r", target="B")
        with pytest.raises(AttributeError):
            rel.confidence = 0.5  # type: ignore[misc]

    def test_with_metadata(self) -> None:
        """Should support metadata dict."""
        rel = RelationCandidate(
            source="A",
            relation_type="r",
            target="B",
            metadata={"llm_model": "gpt-4"},
        )
        assert rel.metadata["llm_model"] == "gpt-4"


# ── ExtractionProvider tests ────────────────────────────────────


class TestExtractionProvider:
    """Test ExtractionProvider protocol."""

    @pytest.mark.asyncio
    async def test_extract_relations(self) -> None:
        """Should return configured relations."""
        relations = [
            RelationCandidate(source="Redis", relation_type="is_a", target="cache"),
        ]
        provider = MockExtractionProvider(relations=relations)
        result = await provider.extract_relations("Redis is a cache")
        assert len(result) == 1
        assert result[0].source == "Redis"

    @pytest.mark.asyncio
    async def test_extract_entities(self) -> None:
        """Should return entities."""
        provider = MockExtractionProvider()
        result = await provider.extract_entities("Test text")
        assert len(result) == 1
        assert result[0]["text"] == "mock-entity"

    @pytest.mark.asyncio
    async def test_empty_relations(self) -> None:
        """Should return empty list when no relations found."""
        provider = MockExtractionProvider(relations=[])
        result = await provider.extract_relations("No relations here")
        assert result == []


# ── Deduplication tests ─────────────────────────────────────────


class TestDeduplication:
    """Test relation deduplication logic."""

    def test_no_overlap(self) -> None:
        """Non-overlapping relations should all be kept."""
        regex = [{"source": "A", "target": "B"}]
        llm = [RelationCandidate(source="C", relation_type="r", target="D")]
        result = deduplicate_relations(regex, llm)
        assert len(result) == 1

    def test_exact_overlap(self) -> None:
        """Exact overlap should be removed."""
        regex = [{"source": "Redis", "target": "cache"}]
        llm = [RelationCandidate(source="Redis", relation_type="is_a", target="cache")]
        result = deduplicate_relations(regex, llm)
        assert len(result) == 0

    def test_case_insensitive_overlap(self) -> None:
        """Overlap should be case-insensitive."""
        regex = [{"source": "redis", "target": "Cache"}]
        llm = [RelationCandidate(source="Redis", relation_type="is_a", target="cache")]
        result = deduplicate_relations(regex, llm)
        assert len(result) == 0

    def test_partial_overlap(self) -> None:
        """Only overlapping relations should be removed."""
        regex = [{"source": "A", "target": "B"}]
        llm = [
            RelationCandidate(source="A", relation_type="r", target="B"),
            RelationCandidate(source="C", relation_type="r", target="D"),
        ]
        result = deduplicate_relations(regex, llm)
        assert len(result) == 1
        assert result[0].source == "C"

    def test_empty_regex(self) -> None:
        """Empty regex list should keep all LLM relations."""
        llm = [
            RelationCandidate(source="A", relation_type="r", target="B"),
            RelationCandidate(source="C", relation_type="r", target="D"),
        ]
        result = deduplicate_relations([], llm)
        assert len(result) == 2

    def test_empty_llm(self) -> None:
        """Empty LLM list should return empty."""
        regex = [{"source": "A", "target": "B"}]
        result = deduplicate_relations(regex, [])
        assert len(result) == 0

    def test_whitespace_handling(self) -> None:
        """Whitespace should be stripped before comparison."""
        regex = [{"source": " Redis ", "target": " cache "}]
        llm = [RelationCandidate(source="Redis", relation_type="r", target="cache")]
        result = deduplicate_relations(regex, llm)
        assert len(result) == 0

    def test_duplicate_llm_entries(self) -> None:
        """Duplicate LLM entries should be deduplicated among themselves."""
        llm = [
            RelationCandidate(source="A", relation_type="r1", target="B"),
            RelationCandidate(source="A", relation_type="r2", target="B"),
        ]
        result = deduplicate_relations([], llm)
        assert len(result) == 1
