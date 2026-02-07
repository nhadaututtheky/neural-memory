"""Tests for Hebbian plasticity — co-activated neurons strengthening."""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.reflex_activation import CoActivation
from neural_memory.engine.retrieval import ReflexPipeline
from neural_memory.storage.memory_store import InMemoryStorage


@pytest.fixture
def hebbian_config() -> BrainConfig:
    """Config with explicit Hebbian parameters."""
    return BrainConfig(
        hebbian_delta=0.03,
        hebbian_threshold=0.5,
        hebbian_initial_weight=0.2,
    )


@pytest_asyncio.fixture
async def hebbian_storage() -> InMemoryStorage:
    """Storage with two neurons and an optional synapse."""
    from neural_memory.core.brain import Brain

    store = InMemoryStorage()
    brain = Brain.create(name="hebbian_test", brain_id="hebb-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    n1 = Neuron.create(type=NeuronType.ENTITY, content="alpha", neuron_id="n-a")
    n2 = Neuron.create(type=NeuronType.ENTITY, content="beta", neuron_id="n-b")
    n3 = Neuron.create(type=NeuronType.ENTITY, content="gamma", neuron_id="n-c")
    await store.add_neuron(n1)
    await store.add_neuron(n2)
    await store.add_neuron(n3)

    return store


@pytest.mark.asyncio
async def test_strengthen_existing_synapse(
    hebbian_config: BrainConfig,
    hebbian_storage: InMemoryStorage,
) -> None:
    """Existing synapse weight increases by hebbian_delta."""
    synapse = Synapse.create(
        source_id="n-a",
        target_id="n-b",
        type=SynapseType.RELATED_TO,
        weight=0.5,
        synapse_id="syn-ab",
    )
    await hebbian_storage.add_synapse(synapse)

    pipeline = ReflexPipeline(hebbian_storage, hebbian_config)

    co = CoActivation(
        neuron_ids=frozenset(["n-a", "n-b"]),
        temporal_window_ms=500,
        binding_strength=0.8,
    )
    await pipeline._defer_co_activated([co])
    await pipeline._write_queue.flush(hebbian_storage)

    updated = await hebbian_storage.get_synapse("syn-ab")
    assert updated is not None
    assert updated.weight == pytest.approx(0.53, abs=1e-9)


@pytest.mark.asyncio
async def test_create_new_synapse(
    hebbian_config: BrainConfig,
    hebbian_storage: InMemoryStorage,
) -> None:
    """New RELATED_TO synapse created when none exists."""
    pipeline = ReflexPipeline(hebbian_storage, hebbian_config)

    co = CoActivation(
        neuron_ids=frozenset(["n-a", "n-b"]),
        temporal_window_ms=500,
        binding_strength=0.8,
    )
    await pipeline._defer_co_activated([co])
    await pipeline._write_queue.flush(hebbian_storage)

    # Should have created a new synapse
    forward = await hebbian_storage.get_synapses(source_id="n-a", target_id="n-b")
    reverse = await hebbian_storage.get_synapses(source_id="n-b", target_id="n-a")
    created = forward or reverse
    assert len(created) == 1
    assert created[0].type == SynapseType.RELATED_TO
    assert created[0].weight == pytest.approx(0.2, abs=1e-9)


@pytest.mark.asyncio
async def test_below_threshold_ignored(
    hebbian_config: BrainConfig,
    hebbian_storage: InMemoryStorage,
) -> None:
    """Weak co-activations (binding_strength < threshold) do nothing."""
    synapse = Synapse.create(
        source_id="n-a",
        target_id="n-b",
        type=SynapseType.RELATED_TO,
        weight=0.5,
        synapse_id="syn-ab",
    )
    await hebbian_storage.add_synapse(synapse)

    pipeline = ReflexPipeline(hebbian_storage, hebbian_config)

    co = CoActivation(
        neuron_ids=frozenset(["n-a", "n-b"]),
        temporal_window_ms=500,
        binding_strength=0.3,  # Below threshold of 0.5
    )
    await pipeline._defer_co_activated([co])
    await pipeline._write_queue.flush(hebbian_storage)

    unchanged = await hebbian_storage.get_synapse("syn-ab")
    assert unchanged is not None
    assert unchanged.weight == pytest.approx(0.5, abs=1e-9)


@pytest.mark.asyncio
async def test_reverse_direction_found(
    hebbian_config: BrainConfig,
    hebbian_storage: InMemoryStorage,
) -> None:
    """Synapse B->A is found and strengthened when co-activating A,B."""
    # Synapse stored as B->A
    synapse = Synapse.create(
        source_id="n-b",
        target_id="n-a",
        type=SynapseType.RELATED_TO,
        weight=0.4,
        synapse_id="syn-ba",
    )
    await hebbian_storage.add_synapse(synapse)

    pipeline = ReflexPipeline(hebbian_storage, hebbian_config)

    co = CoActivation(
        neuron_ids=frozenset(["n-a", "n-b"]),
        temporal_window_ms=500,
        binding_strength=0.8,
    )
    await pipeline._defer_co_activated([co])
    await pipeline._write_queue.flush(hebbian_storage)

    updated = await hebbian_storage.get_synapse("syn-ba")
    assert updated is not None
    assert updated.weight == pytest.approx(0.43, abs=1e-9)

    # No new synapse should be created
    forward = await hebbian_storage.get_synapses(source_id="n-a", target_id="n-b")
    assert len(forward) == 0


@pytest.mark.asyncio
async def test_single_neuron_skipped(
    hebbian_config: BrainConfig,
    hebbian_storage: InMemoryStorage,
) -> None:
    """CoActivation with 1 neuron_id is skipped (no pair to connect)."""
    pipeline = ReflexPipeline(hebbian_storage, hebbian_config)

    co = CoActivation(
        neuron_ids=frozenset(["n-a"]),
        temporal_window_ms=500,
        binding_strength=0.9,
    )
    await pipeline._defer_co_activated([co])
    await pipeline._write_queue.flush(hebbian_storage)

    # No synapses should be created
    all_synapses = await hebbian_storage.get_all_synapses()
    assert len(all_synapses) == 0
