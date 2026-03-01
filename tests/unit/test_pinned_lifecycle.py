"""Tests for pinned (KB) memory lifecycle bypass."""

from __future__ import annotations

from dataclasses import replace as dc_replace
from datetime import timedelta

import pytest

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.utils.timeutils import utcnow


class TestFiberPinned:
    """Test Fiber pinned field."""

    def test_default_not_pinned(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        assert fiber.pinned is False

    def test_create_pinned_via_replace(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        pinned = dc_replace(fiber, pinned=True)
        assert pinned.pinned is True
        # Original unchanged (immutable)
        assert fiber.pinned is False

    def test_pinned_field_exists(self) -> None:
        """Pinned field is part of the dataclass."""
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        assert hasattr(fiber, "pinned")


class TestPinnedDecayBypass:
    """Test that pinned neurons skip decay."""

    @pytest.fixture()
    def decay_manager(self):
        from neural_memory.engine.lifecycle import DecayManager

        return DecayManager(decay_rate=0.1, prune_threshold=0.01, min_age_days=0)

    @pytest.fixture()
    def old_time(self):
        return utcnow() - timedelta(days=30)

    async def test_pinned_neurons_skip_decay(self, decay_manager, old_time, tmp_path):
        """Pinned neurons should not have their activation reduced."""
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        # Create brain
        from neural_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create a neuron that belongs to a pinned fiber
        neuron = Neuron.create(type=NeuronType.ENTITY, content="React hooks")
        await storage.add_neuron(neuron)

        # Create initial neuron state with old activation
        state = NeuronState(
            neuron_id=neuron.id,
            activation_level=1.0,
            access_frequency=1,
            last_activated=old_time,
            decay_rate=0.1,
        )
        await storage.update_neuron_state(state)

        # Create a pinned fiber containing this neuron
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
        )
        pinned_fiber = dc_replace(fiber, pinned=True)
        await storage.add_fiber(pinned_fiber)

        # Apply decay
        report = await decay_manager.apply_decay(storage)

        # Neuron should NOT be decayed (belongs to pinned fiber)
        updated_state = await storage.get_neuron_state(neuron.id)
        assert updated_state is not None
        assert updated_state.activation_level == 1.0

        await storage.close()

    async def test_unpinned_neurons_still_decay(self, decay_manager, old_time, tmp_path):
        """Non-pinned neurons should decay normally."""
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        from neural_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        neuron = Neuron.create(type=NeuronType.ENTITY, content="Temporary note")
        await storage.add_neuron(neuron)

        state = NeuronState(
            neuron_id=neuron.id,
            activation_level=1.0,
            access_frequency=1,
            last_activated=old_time,
            decay_rate=0.1,
        )
        await storage.update_neuron_state(state)

        # Create a NON-pinned fiber
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
        )
        await storage.add_fiber(fiber)
        assert fiber.pinned is False

        report = await decay_manager.apply_decay(storage)

        # Neuron SHOULD be decayed
        updated_state = await storage.get_neuron_state(neuron.id)
        assert updated_state is not None
        assert updated_state.activation_level < 1.0

        await storage.close()


class TestPinnedCompressionBypass:
    """Test that pinned fibers skip compression."""

    async def test_pinned_fiber_skips_compression(self, tmp_path):
        """Pinned fiber should remain at tier 0."""
        from neural_memory.engine.compression import CompressionEngine
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        from neural_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create an old neuron + pinned fiber
        old_time = utcnow() - timedelta(days=200)
        neuron = Neuron.create(type=NeuronType.ENTITY, content="Permanent KB content")
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
        )
        pinned_fiber = dc_replace(fiber, pinned=True, created_at=old_time)
        await storage.add_fiber(pinned_fiber)

        # Run compression
        engine = CompressionEngine(storage)
        report = await engine.run()

        # Pinned fiber should be skipped
        assert report.fibers_skipped >= 1

        # Verify tier unchanged
        updated = await storage.get_fiber(pinned_fiber.id)
        assert updated is not None
        assert updated.compression_tier == 0

        await storage.close()


class TestPinnedPruneBypass:
    """Test that synapses to pinned neurons skip pruning."""

    async def test_pinned_synapse_skips_prune(self, tmp_path):
        """Synapses connected to pinned neurons should not be pruned."""
        from neural_memory.engine.consolidation import (
            ConsolidationEngine,
            ConsolidationStrategy,
        )
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        from neural_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create two neurons
        n1 = Neuron.create(type=NeuronType.ENTITY, content="Pinned KB concept")
        n2 = Neuron.create(type=NeuronType.ENTITY, content="Related concept")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Create a weak, old synapse (normally would be pruned)
        old_time = utcnow() - timedelta(days=30)
        synapse = Synapse.create(
            source_id=n1.id,
            target_id=n2.id,
            type=SynapseType.RELATED_TO,
            weight=0.01,  # Below prune threshold
        )
        synapse = dc_replace(synapse, created_at=old_time, last_activated=old_time)
        await storage.add_synapse(synapse)

        # Pin the fiber containing n1
        fiber = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids={synapse.id},
            anchor_neuron_id=n1.id,
        )
        pinned_fiber = dc_replace(fiber, pinned=True)
        await storage.add_fiber(pinned_fiber)

        # Run pruning
        engine = ConsolidationEngine(storage)
        report = await engine.run(strategies=[ConsolidationStrategy.PRUNE])

        # Synapse should NOT be pruned (connected to pinned neuron)
        remaining = await storage.get_synapse(synapse.id)
        assert remaining is not None, "Synapse connected to pinned neuron should survive pruning"

        await storage.close()


class TestPinUnpin:
    """Test pin/unpin operations."""

    async def test_pin_fibers(self, tmp_path):
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        from neural_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        neuron = Neuron.create(type=NeuronType.ENTITY, content="test")
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
        )
        await storage.add_fiber(fiber)
        assert fiber.pinned is False

        # Pin
        count = await storage.pin_fibers([fiber.id], pinned=True)
        assert count == 1

        updated = await storage.get_fiber(fiber.id)
        assert updated is not None
        assert updated.pinned is True

        # Unpin
        count = await storage.pin_fibers([fiber.id], pinned=False)
        assert count == 1

        updated = await storage.get_fiber(fiber.id)
        assert updated is not None
        assert updated.pinned is False

        await storage.close()

    async def test_get_pinned_neuron_ids(self, tmp_path):
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        from neural_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        n1 = Neuron.create(type=NeuronType.ENTITY, content="pinned")
        n2 = Neuron.create(type=NeuronType.ENTITY, content="not pinned")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Pinned fiber
        f1 = Fiber.create(neuron_ids={n1.id}, synapse_ids=set(), anchor_neuron_id=n1.id)
        f1 = dc_replace(f1, pinned=True)
        await storage.add_fiber(f1)

        # Not pinned fiber
        f2 = Fiber.create(neuron_ids={n2.id}, synapse_ids=set(), anchor_neuron_id=n2.id)
        await storage.add_fiber(f2)

        pinned_ids = await storage.get_pinned_neuron_ids()
        assert n1.id in pinned_ids
        assert n2.id not in pinned_ids

        await storage.close()
