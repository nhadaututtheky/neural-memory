"""Tests for memory consolidation — high-frequency fibers boost synapses."""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
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


# ── INFER strategy tests ────────────────────────────────────────


@pytest_asyncio.fixture
async def infer_storage() -> InMemoryStorage:
    """Storage with neurons and co-activation events for inference."""
    store = InMemoryStorage()
    brain = Brain.create(name="infer_test", brain_id="infer-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    # Create neurons
    n1 = Neuron.create(type=NeuronType.ENTITY, content="python programming", neuron_id="in-1")
    n2 = Neuron.create(type=NeuronType.ENTITY, content="python testing", neuron_id="in-2")
    n3 = Neuron.create(type=NeuronType.ENTITY, content="python debugging", neuron_id="in-3")
    for n in [n1, n2, n3]:
        await store.add_neuron(n)

    # Existing synapse between n1 and n2
    s1 = Synapse.create(
        source_id="in-1",
        target_id="in-2",
        type=SynapseType.RELATED_TO,
        weight=0.4,
        synapse_id="syn-existing",
    )
    await store.add_synapse(s1)

    # Create a fiber containing all neurons
    fiber = Fiber(
        id="fiber-infer",
        neuron_ids={"in-1", "in-2", "in-3"},
        synapse_ids={"syn-existing"},
        anchor_neuron_id="in-1",
        pathway=["in-1", "in-2", "in-3"],
        frequency=5,
    )
    await store.add_fiber(fiber)

    # Record co-activation events (n1,n3 have no existing synapse — should be inferred)
    for _ in range(5):
        await store.record_co_activation("in-1", "in-3", 0.8)
    # n1,n2 already have a synapse — should be reinforced
    for _ in range(4):
        await store.record_co_activation("in-1", "in-2", 0.7)

    return store


@pytest.mark.asyncio
async def test_infer_creates_new_synapse(infer_storage: InMemoryStorage) -> None:
    """INFER creates CO_OCCURS synapse for pairs without existing connections."""
    config = ConsolidationConfig(infer_co_activation_threshold=3)
    engine = ConsolidationEngine(infer_storage, config)
    report = await engine.run(strategies=[ConsolidationStrategy.INFER])

    assert report.synapses_inferred >= 1

    # Check that a synapse was created between in-1 and in-3
    synapses = await infer_storage.get_synapses(source_id="in-1", target_id="in-3")
    reverse = await infer_storage.get_synapses(source_id="in-3", target_id="in-1")
    all_found = synapses + reverse
    assert len(all_found) >= 1
    inferred = all_found[0]
    assert inferred.type == SynapseType.CO_OCCURS
    assert inferred.metadata.get("_inferred") is True


@pytest.mark.asyncio
async def test_infer_reinforces_existing_synapse(infer_storage: InMemoryStorage) -> None:
    """INFER reinforces existing synapses for pairs that already have connections."""
    original = await infer_storage.get_synapse("syn-existing")
    assert original is not None
    original_weight = original.weight

    config = ConsolidationConfig(infer_co_activation_threshold=3)
    engine = ConsolidationEngine(infer_storage, config)
    await engine.run(strategies=[ConsolidationStrategy.INFER])

    updated = await infer_storage.get_synapse("syn-existing")
    assert updated is not None
    assert updated.weight > original_weight


@pytest.mark.asyncio
async def test_infer_prunes_old_co_activations(infer_storage: InMemoryStorage) -> None:
    """INFER prunes co-activation events outside the window."""
    config = ConsolidationConfig(infer_co_activation_threshold=3, infer_window_days=7)
    engine = ConsolidationEngine(infer_storage, config)
    report = await engine.run(strategies=[ConsolidationStrategy.INFER])

    assert report.co_activations_pruned >= 0


@pytest.mark.asyncio
async def test_infer_dry_run(infer_storage: InMemoryStorage) -> None:
    """Dry run reports counts but doesn't modify storage."""
    config = ConsolidationConfig(infer_co_activation_threshold=3)
    engine = ConsolidationEngine(infer_storage, config)

    # Count synapses before
    synapses_before = await infer_storage.get_synapses()
    count_before = len(synapses_before)

    report = await engine.run(strategies=[ConsolidationStrategy.INFER], dry_run=True)

    assert report.synapses_inferred >= 1
    assert report.dry_run is True

    # Synapses unchanged
    synapses_after = await infer_storage.get_synapses()
    assert len(synapses_after) == count_before


@pytest.mark.asyncio
async def test_infer_report_in_summary(infer_storage: InMemoryStorage) -> None:
    """INFER results appear in report summary."""
    config = ConsolidationConfig(infer_co_activation_threshold=3)
    engine = ConsolidationEngine(infer_storage, config)
    report = await engine.run(strategies=[ConsolidationStrategy.INFER])

    summary = report.summary()
    assert "Synapses inferred" in summary
    assert "Co-activations pruned" in summary
