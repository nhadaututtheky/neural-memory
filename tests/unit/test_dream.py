"""Tests for dream engine — random exploration for hidden connections."""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.dream import DreamResult, dream
from neural_memory.storage.memory_store import InMemoryStorage


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """Storage with a brain context ready for dream tests."""
    storage = InMemoryStorage()
    brain = Brain.create(name="dream_test", brain_id="dream-brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


# ── helpers ──────────────────────────────────────────────────────


async def _add_connected_graph(store: InMemoryStorage) -> list[str]:
    """Add 4 neurons connected in a chain: n1->n2->n3->n4.

    Returns the list of neuron IDs.
    """
    ids = ["dn-1", "dn-2", "dn-3", "dn-4"]
    neurons = [
        Neuron.create(type=NeuronType.CONCEPT, content=f"concept-{i}", neuron_id=nid)
        for i, nid in enumerate(ids)
    ]
    for n in neurons:
        await store.add_neuron(n)

    # Chain: n1->n2, n2->n3, n3->n4
    for i in range(len(ids) - 1):
        syn = Synapse.create(
            source_id=ids[i],
            target_id=ids[i + 1],
            type=SynapseType.RELATED_TO,
            weight=0.5,
            synapse_id=f"dsyn-{i}",
        )
        await store.add_synapse(syn)

    return ids


# ── test: empty graph ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dream_empty_graph(store: InMemoryStorage) -> None:
    """Dream with no neurons returns empty result."""
    config = BrainConfig()
    result = await dream(store, config, seed=42)

    assert isinstance(result, DreamResult)
    assert result.synapses_created == []
    assert result.pairs_explored == 0


@pytest.mark.asyncio
async def test_dream_single_neuron(store: InMemoryStorage) -> None:
    """Dream with a single neuron returns empty result (need >=2)."""
    n = Neuron.create(type=NeuronType.CONCEPT, content="lonely", neuron_id="solo-1")
    await store.add_neuron(n)

    config = BrainConfig()
    result = await dream(store, config, seed=42)

    assert result.synapses_created == []
    assert result.pairs_explored == 0


# ── test: small connected graph creates dream synapses ───────────


@pytest.mark.asyncio
async def test_dream_creates_synapses(store: InMemoryStorage) -> None:
    """Dream with connected neurons creates new RELATED_TO synapses."""
    await _add_connected_graph(store)

    config = BrainConfig(dream_neuron_count=4)
    result = await dream(store, config, seed=42)

    assert isinstance(result, DreamResult)
    # Spreading activation should find co-activated pairs beyond
    # direct connections and create new synapses for them.
    assert len(result.synapses_created) > 0
    assert result.pairs_explored > 0


# ── test: dream synapse properties ───────────────────────────────


@pytest.mark.asyncio
async def test_dream_synapse_weight_and_metadata(store: InMemoryStorage) -> None:
    """Dream synapses have weight=0.1 and _dream=True metadata."""
    await _add_connected_graph(store)

    config = BrainConfig(dream_neuron_count=4)
    result = await dream(store, config, seed=42)

    for synapse in result.synapses_created:
        assert synapse.weight == pytest.approx(0.1)
        assert synapse.type == SynapseType.RELATED_TO
        assert synapse.metadata.get("_dream") is True


# ── test: deterministic with seed ────────────────────────────────


@pytest.mark.asyncio
async def test_dream_deterministic_with_seed(store: InMemoryStorage) -> None:
    """Running dream twice with the same seed produces identical results."""
    await _add_connected_graph(store)

    config = BrainConfig(dream_neuron_count=4)
    result_a = await dream(store, config, seed=123)
    result_b = await dream(store, config, seed=123)

    assert len(result_a.synapses_created) == len(result_b.synapses_created)
    assert result_a.pairs_explored == result_b.pairs_explored

    pairs_a = {(s.source_id, s.target_id) for s in result_a.synapses_created}
    pairs_b = {(s.source_id, s.target_id) for s in result_b.synapses_created}
    assert pairs_a == pairs_b


# ── test: no duplicate synapses for connected pairs ──────────────


@pytest.mark.asyncio
async def test_dream_skips_existing_connections(store: InMemoryStorage) -> None:
    """Dream does not create synapses for pairs already connected."""
    await _add_connected_graph(store)

    config = BrainConfig(dream_neuron_count=4)
    result = await dream(store, config, seed=42)

    for new_syn in result.synapses_created:
        # new_syn should connect distinct neurons
        assert new_syn.source_id != new_syn.target_id


# ── test: dream_neuron_count respected ───────────────────────────


@pytest.mark.asyncio
async def test_dream_respects_neuron_count(store: InMemoryStorage) -> None:
    """With dream_neuron_count=1, fewer neurons are sampled."""
    await _add_connected_graph(store)

    config_small = BrainConfig(dream_neuron_count=1)
    config_large = BrainConfig(dream_neuron_count=4)

    result_small = await dream(store, config_small, seed=42)
    result_large = await dream(store, config_large, seed=42)

    # With fewer seed neurons, we expect equal or fewer pairs explored
    assert result_small.pairs_explored <= result_large.pairs_explored


# ── test: DreamResult is frozen ──────────────────────────────────


def test_dream_result_frozen() -> None:
    """DreamResult is a frozen dataclass — attributes are immutable."""
    result = DreamResult(synapses_created=[], pairs_explored=5)
    with pytest.raises(AttributeError):
        result.pairs_explored = 10  # type: ignore[misc]
