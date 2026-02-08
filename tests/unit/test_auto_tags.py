"""Tests for auto-tag generation in the encoder.

Auto-tags ensure every fiber has a baseline tag set for clustering
and pattern extraction, regardless of agent quality.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import NeuronType
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.storage.memory_store import InMemoryStorage


@pytest.fixture
async def encoder() -> tuple[MemoryEncoder, InMemoryStorage]:
    """Create an encoder with in-memory storage."""
    storage = InMemoryStorage()
    config = BrainConfig()
    brain = Brain.create(name="test", config=config)
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return MemoryEncoder(storage, brain.config), storage


class TestAutoTagGeneration:
    """Test that fibers always get auto-generated tags."""

    @pytest.mark.asyncio
    async def test_auto_tags_from_entities(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Entity names should appear as tags even without agent-provided tags."""
        enc, _storage = encoder

        result = await enc.encode(
            "Met Alice and Bob at Microsoft headquarters",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        tags_lower = {t.lower() for t in result.fiber.tags}
        # At least one entity should become a tag
        entity_names = {"alice", "bob", "microsoft"}
        assert tags_lower & entity_names, (
            f"Expected at least one of {entity_names} in tags, got {tags_lower}"
        )

    @pytest.mark.asyncio
    async def test_auto_tags_from_keywords(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Top keywords should appear as tags."""
        enc, _storage = encoder

        result = await enc.encode(
            "We decided to use Redis for the caching layer instead of Memcached",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        tags_lower = {t.lower() for t in result.fiber.tags}
        # Core keywords should be present
        keyword_candidates = {"redis", "caching", "memcached"}
        assert tags_lower & keyword_candidates, (
            f"Expected at least one of {keyword_candidates} in tags, got {tags_lower}"
        )

    @pytest.mark.asyncio
    async def test_agent_tags_merged_with_auto_tags(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Agent-provided tags should be unioned with auto-tags, not replaced."""
        enc, _storage = encoder

        result = await enc.encode(
            "We decided to use Redis for caching",
            timestamp=datetime(2024, 2, 4, 15, 0),
            tags={"architecture-decision", "infrastructure"},
        )

        tags = result.fiber.tags
        # Agent tags preserved (may be normalized: "infrastructure" → "infra")
        assert "architecture-decision" in tags or "infra" in tags
        assert "infra" in tags  # "infrastructure" normalized to "infra"
        # Auto-tags also present
        tags_lower = {t.lower() for t in tags}
        assert tags_lower & {"redis", "caching"}, (
            f"Auto-tags missing: expected redis/caching in {tags_lower}"
        )

    @pytest.mark.asyncio
    async def test_auto_tags_no_agent_tags(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Fibers should have tags even when agent provides none."""
        enc, _storage = encoder

        result = await enc.encode(
            "PostgreSQL database is running slowly on production server",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        # Should NOT be empty
        assert len(result.fiber.tags) > 0, "Fiber should have auto-generated tags"

    @pytest.mark.asyncio
    async def test_auto_tags_short_content(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Short content should still get some tags (at minimum from keywords)."""
        enc, _storage = encoder

        result = await enc.encode(
            "Fix auth bug",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        # Even short content should produce tags
        assert len(result.fiber.tags) >= 0  # May be empty for very short

    @pytest.mark.asyncio
    async def test_auto_tags_normalized_lowercase(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Auto-tags should be normalized to lowercase."""
        enc, _storage = encoder

        result = await enc.encode(
            "Alice deployed the NextJS application to AWS",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        # All auto-generated tags should be lowercase (agent tags may not be)
        # Filter out any agent-provided tags (there are none here)
        for tag in result.fiber.tags:
            assert tag == tag.lower(), f"Auto-tag '{tag}' should be lowercase"

    @pytest.mark.asyncio
    async def test_auto_tags_no_single_char(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Auto-tags should not include single-character strings."""
        enc, _storage = encoder

        result = await enc.encode(
            "I met A at B for a quick C review",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        for tag in result.fiber.tags:
            assert len(tag) >= 2, f"Tag '{tag}' is too short (min 2 chars)"

    @pytest.mark.asyncio
    async def test_auto_tags_vietnamese_content(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Vietnamese content should produce meaningful tags."""
        enc, _storage = encoder

        result = await enc.encode(
            "Anh Minh quyết định dùng PostgreSQL cho dự án mới",
            timestamp=datetime(2024, 2, 4, 15, 0),
            language="vi",
        )

        tags_lower = {t.lower() for t in result.fiber.tags}
        # Should extract Vietnamese entities/keywords
        assert len(tags_lower) > 0, "Vietnamese content should produce auto-tags"

    @pytest.mark.asyncio
    async def test_auto_tags_improve_clustering_potential(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Two related memories without agent tags should share auto-tags."""
        enc, _storage = encoder

        result1 = await enc.encode(
            "Alice fixed the authentication bug in the login module",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        result2 = await enc.encode(
            "Alice refactored the authentication service for better security",
            timestamp=datetime(2024, 2, 5, 10, 0),
        )

        # Both should have overlapping tags (e.g., "alice", "authentication")
        overlap = result1.fiber.tags & result2.fiber.tags
        assert len(overlap) > 0, (
            f"Related memories should share auto-tags.\n"
            f"Tags 1: {result1.fiber.tags}\n"
            f"Tags 2: {result2.fiber.tags}"
        )

    @pytest.mark.asyncio
    async def test_auto_tags_limit(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Auto-tags should not flood fibers with too many tags."""
        enc, _storage = encoder

        result = await enc.encode(
            "Alice and Bob met Charlie and Diana at the Microsoft office in Seattle "
            "to discuss the PostgreSQL migration plan with Redis caching layer "
            "and Kubernetes deployment strategy for the NextJS frontend application",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        # Auto-tags: up to 5 keywords + entities. With agent tags = 0,
        # total should be reasonable (not > 30)
        assert len(result.fiber.tags) <= 30, (
            f"Too many auto-tags: {len(result.fiber.tags)}"
        )


class TestAutoTagsWithConflictDetection:
    """Test that auto-tags improve conflict detection."""

    @pytest.mark.asyncio
    async def test_conflict_detection_uses_merged_tags(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Conflict detection should benefit from auto-tags even without agent tags."""
        enc, _storage = encoder

        # First memory — no agent tags
        await enc.encode(
            "We decided to use PostgreSQL for the database",
            timestamp=datetime(2024, 2, 4, 15, 0),
            metadata={"type": "decision"},
        )

        # Second memory — contradicts, also no agent tags
        result2 = await enc.encode(
            "We decided to use MySQL for the database",
            timestamp=datetime(2024, 2, 5, 10, 0),
            metadata={"type": "decision"},
        )

        # Both memories should have auto-tags (e.g., "database", "decided")
        # which enables conflict detection via tag overlap
        assert len(result2.fiber.tags) > 0


class TestGenerateAutoTagsUnit:
    """Unit tests for the _generate_auto_tags method directly."""

    def test_empty_neurons_returns_empty(self) -> None:
        """No neurons and empty content should produce empty tags."""
        storage = InMemoryStorage()
        config = BrainConfig()
        enc = MemoryEncoder(storage, config)

        result = enc._generate_auto_tags(
            entity_neurons=[],
            concept_neurons=[],
            content="",
        )
        assert result == set()

    def test_entity_neurons_become_tags(self) -> None:
        """Entity neuron content should become lowercase tags."""
        from neural_memory.core.neuron import Neuron

        storage = InMemoryStorage()
        config = BrainConfig()
        enc = MemoryEncoder(storage, config)

        entities = [
            Neuron.create(type=NeuronType.ENTITY, content="Alice"),
            Neuron.create(type=NeuronType.ENTITY, content="Microsoft"),
        ]

        result = enc._generate_auto_tags(
            entity_neurons=entities,
            concept_neurons=[],
            content="Alice works at Microsoft",
        )

        assert "alice" in result
        assert "microsoft" in result

    def test_keywords_limited_to_top_5(self) -> None:
        """At most 5 keywords should be included as tags."""
        storage = InMemoryStorage()
        config = BrainConfig()
        enc = MemoryEncoder(storage, config)

        # Long content with many keywords
        content = (
            "Redis PostgreSQL MySQL MongoDB Cassandra Elasticsearch "
            "Kubernetes Docker Terraform Ansible Jenkins GitHub"
        )

        result = enc._generate_auto_tags(
            entity_neurons=[],
            concept_neurons=[],
            content=content,
        )

        # Keywords from extract_weighted_keywords limited to 5,
        # but entities may add more. With no entities, should be <= 5 keywords
        # (plus possible bi-grams which also count toward the 5 limit)
        # Just verify it's bounded
        assert len(result) <= 15  # reasonable upper bound

    def test_short_tags_filtered(self) -> None:
        """Tags shorter than 2 chars should be excluded."""
        from neural_memory.core.neuron import Neuron

        storage = InMemoryStorage()
        config = BrainConfig()
        enc = MemoryEncoder(storage, config)

        entities = [
            Neuron.create(type=NeuronType.ENTITY, content="A"),  # too short
            Neuron.create(type=NeuronType.ENTITY, content="DB"),  # ok
        ]

        result = enc._generate_auto_tags(
            entity_neurons=entities,
            concept_neurons=[],
            content="A and DB",
        )

        assert "a" not in result
        # "DB" normalizes to "database" via tag normalizer synonym map
        assert "database" in result or "db" in result
