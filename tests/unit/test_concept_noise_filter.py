"""Tests for concept neuron noise filtering.

Verifies that the ExtractConceptNeuronsStep avoids creating low-signal
concept neurons from short common words, entity duplicates, and
3-char noise that survives keyword extraction.
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


class TestConceptNoiseFilter:
    """Concept neurons should exclude short noise words."""

    @pytest.mark.asyncio
    async def test_no_concept_from_short_noise_words(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """3-char words like 'AI', 'OS', 'the' should not become concept neurons."""
        enc, _storage = encoder

        result = await enc.encode(
            "The AI system has a new OS running",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        concept_contents = {
            n.content.lower() for n in result.neurons_created if n.type == NeuronType.CONCEPT
        }
        # "AI" and "OS" are 2-3 chars and should be filtered out
        assert "ai" not in concept_contents
        assert "os" not in concept_contents

    @pytest.mark.asyncio
    async def test_short_content_gets_fewer_concepts(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Short text (<50 chars) should produce at most 3 concept neurons."""
        enc, _storage = encoder

        result = await enc.encode(
            "Fixed a bug",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        concept_count = sum(1 for n in result.neurons_created if n.type == NeuronType.CONCEPT)
        assert concept_count <= 3, f"Short text produced {concept_count} concepts, expected <= 3"

    @pytest.mark.asyncio
    async def test_entity_not_duplicated_as_concept(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Words already captured as entity neurons should not also become concepts."""
        enc, _storage = encoder

        result = await enc.encode(
            "Met Alice at the Redis deployment meeting",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        entity_contents = {
            n.content.lower()
            for n in result.neurons_created
            if n.type in (NeuronType.ENTITY, NeuronType.CONCEPT) and n.type != NeuronType.CONCEPT
        }
        concept_contents = {
            n.content.lower() for n in result.neurons_created if n.type == NeuronType.CONCEPT
        }
        # No overlap between entities and concepts
        overlap = entity_contents & concept_contents
        assert not overlap, f"Entities duplicated as concepts: {overlap}"

    @pytest.mark.asyncio
    async def test_noise_concept_set_filtered(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Known noise words ('use', 'run', 'new', etc.) should not become concepts."""
        enc, _storage = encoder

        result = await enc.encode(
            "We use the new tool to run the test",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        concept_contents = {
            n.content.lower() for n in result.neurons_created if n.type == NeuronType.CONCEPT
        }
        for noise_word in ("use", "run", "new"):
            assert noise_word not in concept_contents, (
                f"Noise word '{noise_word}' became a concept neuron"
            )

    @pytest.mark.asyncio
    async def test_meaningful_concepts_still_created(
        self, encoder: tuple[MemoryEncoder, InMemoryStorage]
    ) -> None:
        """Meaningful multi-char concepts should still be created normally."""
        enc, _storage = encoder

        result = await enc.encode(
            "Decided to use PostgreSQL for the caching layer instead of Redis",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        concept_contents = {
            n.content.lower() for n in result.neurons_created if n.type == NeuronType.CONCEPT
        }
        # Keywords may be bigrams — check that meaningful terms appear as
        # substrings in concept content rather than exact matches.
        meaningful = ["caching", "postgresql", "redis"]
        found = [term for term in meaningful if any(term in c for c in concept_contents)]
        assert found, (
            f"Expected at least one meaningful concept from {meaningful}, got {concept_contents}"
        )
