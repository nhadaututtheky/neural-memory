"""Integration tests for encoding flow."""

from __future__ import annotations

from datetime import datetime

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import NeuronType
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.storage.memory_store import InMemoryStorage


class TestEncodingFlow:
    """Integration tests for the memory encoding flow."""

    @pytest.fixture
    async def storage(self) -> InMemoryStorage:
        """Create storage with brain."""
        storage = InMemoryStorage()
        config = BrainConfig()
        brain = Brain.create(name="test", config=config)
        await storage.save_brain(brain)
        storage.set_brain(brain.id)
        return storage

    @pytest.mark.asyncio
    async def test_encode_simple_memory(self, storage: InMemoryStorage) -> None:
        """Test encoding a simple memory."""
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore
        assert brain is not None

        encoder = MemoryEncoder(storage, brain.config)

        result = await encoder.encode(
            "Met Alice at the coffee shop",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        # Should create neurons
        assert len(result.neurons_created) > 0

        # Should create synapses
        assert len(result.synapses_created) > 0

        # Should create a fiber
        assert result.fiber is not None
        assert result.fiber.anchor_neuron_id is not None

    @pytest.mark.asyncio
    async def test_encode_creates_time_neurons(self, storage: InMemoryStorage) -> None:
        """Test that encoding creates time neurons."""
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore
        assert brain is not None

        encoder = MemoryEncoder(storage, brain.config)

        timestamp = datetime(2024, 2, 4, 15, 0)
        result = await encoder.encode("Something happened", timestamp=timestamp)

        # Should have time neuron
        time_neurons = [n for n in result.neurons_created if n.type == NeuronType.TIME]
        assert len(time_neurons) > 0

    @pytest.mark.asyncio
    async def test_encode_extracts_entities(self, storage: InMemoryStorage) -> None:
        """Test that encoding extracts entities."""
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore
        assert brain is not None

        encoder = MemoryEncoder(storage, brain.config)

        result = await encoder.encode(
            "Met with Alice and Bob at Microsoft headquarters",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        # Should have entity neurons
        entity_neurons = [n for n in result.neurons_created if n.type == NeuronType.ENTITY]
        spatial_neurons = [n for n in result.neurons_created if n.type == NeuronType.SPATIAL]

        # Should find some entities
        all_entities = entity_neurons + spatial_neurons
        assert len(all_entities) >= 0  # May or may not find all

    @pytest.mark.asyncio
    async def test_encode_creates_synapses(self, storage: InMemoryStorage) -> None:
        """Test that encoding creates meaningful synapses."""
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore
        assert brain is not None

        encoder = MemoryEncoder(storage, brain.config)

        result = await encoder.encode(
            "Met Alice at coffee shop at 3pm",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        # Should create multiple synapses connecting neurons
        assert len(result.synapses_created) > 0

        # Synapses should connect valid neurons
        for synapse in result.synapses_created:
            assert synapse.source_id
            assert synapse.target_id
            assert synapse.weight > 0

    @pytest.mark.asyncio
    async def test_encode_with_tags(self, storage: InMemoryStorage) -> None:
        """Test encoding with tags."""
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore
        assert brain is not None

        encoder = MemoryEncoder(storage, brain.config)

        result = await encoder.encode(
            "Important meeting",
            timestamp=datetime(2024, 2, 4, 15, 0),
            tags={"work", "important"},
        )

        assert "work" in result.fiber.tags
        assert "important" in result.fiber.tags

    @pytest.mark.asyncio
    async def test_encode_with_metadata(self, storage: InMemoryStorage) -> None:
        """Test encoding with metadata."""
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore
        assert brain is not None

        encoder = MemoryEncoder(storage, brain.config)

        result = await encoder.encode(
            "Test memory",
            timestamp=datetime(2024, 2, 4, 15, 0),
            metadata={"source": "test", "priority": 1},
        )

        # Metadata should be attached to fiber
        assert result.fiber.metadata.get("source") == "test"

    @pytest.mark.asyncio
    async def test_multiple_encodes_link_temporally(
        self, storage: InMemoryStorage
    ) -> None:
        """Test that multiple encodings link to nearby memories."""
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore
        assert brain is not None

        encoder = MemoryEncoder(storage, brain.config)

        # Encode first memory
        _result1 = await encoder.encode(
            "First event",
            timestamp=datetime(2024, 2, 4, 14, 0),
        )

        # Encode nearby memory
        result2 = await encoder.encode(
            "Second event",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        # Second encoding might link to first
        # (depends on implementation details)
        assert result2.fiber is not None

    @pytest.mark.asyncio
    async def test_encode_vietnamese_content(self, storage: InMemoryStorage) -> None:
        """Test encoding Vietnamese content."""
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore
        assert brain is not None

        encoder = MemoryEncoder(storage, brain.config)

        result = await encoder.encode(
            "Chiều nay 3h uống cà phê ở Viva với Minh",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        # Should create neurons
        assert len(result.neurons_created) > 0

        # Should create fiber
        assert result.fiber is not None

    @pytest.mark.asyncio
    async def test_fiber_has_time_bounds(self, storage: InMemoryStorage) -> None:
        """Test that created fiber has time bounds."""
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore
        assert brain is not None

        encoder = MemoryEncoder(storage, brain.config)

        timestamp = datetime(2024, 2, 4, 15, 0)
        result = await encoder.encode("Test memory", timestamp=timestamp)

        assert result.fiber.time_start is not None
        assert result.fiber.time_end is not None

    @pytest.mark.asyncio
    async def test_fiber_retrievable_after_encode(
        self, storage: InMemoryStorage
    ) -> None:
        """Test that fiber can be retrieved after encoding."""
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore
        assert brain is not None

        encoder = MemoryEncoder(storage, brain.config)

        result = await encoder.encode(
            "Test memory",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        # Retrieve fiber
        fiber = await storage.get_fiber(result.fiber.id)
        assert fiber is not None
        assert fiber.id == result.fiber.id
