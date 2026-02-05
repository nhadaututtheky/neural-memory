"""Tests for memory consolidation — high-frequency fibers boost synapses."""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.lifecycle import DecayManager
from neural_memory.storage.memory_store import InMemoryStorage


@pytest_asyncio.fixture
async def consolidation_storage() -> InMemoryStorage:
    """Storage with fibers at different frequencies."""
    store = InMemoryStorage()
    brain = Brain.create(name="consolidation_test", brain_id="cons-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    # Create neurons
    n1 = Neuron.create(type=NeuronType.ENTITY, content="alpha", neuron_id="n-1")
    n2 = Neuron.create(type=NeuronType.ENTITY, content="beta", neuron_id="n-2")
    n3 = Neuron.create(type=NeuronType.ENTITY, content="gamma", neuron_id="n-3")
    for n in [n1, n2, n3]:
        await store.add_neuron(n)

    # Synapses for high-frequency fiber
    s1 = Synapse.create(
        source_id="n-1",
        target_id="n-2",
        type=SynapseType.RELATED_TO,
        weight=0.5,
        synapse_id="syn-hi-1",
    )
    # Synapse for low-frequency fiber
    s2 = Synapse.create(
        source_id="n-2",
        target_id="n-3",
        type=SynapseType.RELATED_TO,
        weight=0.5,
        synapse_id="syn-lo-1",
    )
    await store.add_synapse(s1)
    await store.add_synapse(s2)

    # High-frequency fiber (frequency=10)
    hi_fiber = Fiber(
        id="fiber-hi",
        neuron_ids={"n-1", "n-2"},
        synapse_ids={"syn-hi-1"},
        anchor_neuron_id="n-1",
        pathway=["n-1", "n-2"],
        frequency=10,
    )
    # Low-frequency fiber (frequency=2)
    lo_fiber = Fiber(
        id="fiber-lo",
        neuron_ids={"n-2", "n-3"},
        synapse_ids={"syn-lo-1"},
        anchor_neuron_id="n-2",
        pathway=["n-2", "n-3"],
        frequency=2,
    )
    await store.add_fiber(hi_fiber)
    await store.add_fiber(lo_fiber)

    return store


@pytest.mark.asyncio
async def test_high_frequency_fiber_consolidated(
    consolidation_storage: InMemoryStorage,
) -> None:
    """Synapses in high-freq fiber boosted by boost_delta."""
    manager = DecayManager()
    await manager.consolidate(
        consolidation_storage,
        frequency_threshold=5,
        boost_delta=0.03,
    )

    synapse = await consolidation_storage.get_synapse("syn-hi-1")
    assert synapse is not None
    assert synapse.weight == pytest.approx(0.53, abs=1e-9)


@pytest.mark.asyncio
async def test_low_frequency_fiber_unchanged(
    consolidation_storage: InMemoryStorage,
) -> None:
    """Synapses in low-freq fiber untouched."""
    manager = DecayManager()
    await manager.consolidate(
        consolidation_storage,
        frequency_threshold=5,
        boost_delta=0.03,
    )

    synapse = await consolidation_storage.get_synapse("syn-lo-1")
    assert synapse is not None
    assert synapse.weight == pytest.approx(0.5, abs=1e-9)


@pytest.mark.asyncio
async def test_returns_consolidated_count(
    consolidation_storage: InMemoryStorage,
) -> None:
    """Return value matches number of synapses updated."""
    manager = DecayManager()
    count = await manager.consolidate(
        consolidation_storage,
        frequency_threshold=5,
        boost_delta=0.03,
    )

    # Only the high-frequency fiber's 1 synapse should be consolidated
    assert count == 1
