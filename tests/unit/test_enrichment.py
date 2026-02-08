"""Tests for enrichment engine — transitive closure and cross-cluster linking."""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.enrichment import (
    EnrichmentResult,
    enrich,
    find_cross_cluster_links,
    find_transitive_closures,
)
from neural_memory.storage.memory_store import InMemoryStorage

# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """Create an InMemoryStorage with a brain context."""
    s = InMemoryStorage()
    brain = Brain.create(name="enrichment-test", config=BrainConfig(), owner_id="test")
    await s.save_brain(brain)
    s.set_brain(brain.id)
    return s


# ── Transitive Closure Tests ─────────────────────────────────────


class TestFindTransitiveClosures:
    """Tests for find_transitive_closures function."""

    async def test_chain_creates_transitive_synapse(self, store: InMemoryStorage) -> None:
        """A->B->C chain creates A->C with correct weight and metadata."""
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content="deploy failed", neuron_id="n-a"),
            Neuron.create(type=NeuronType.CONCEPT, content="config error", neuron_id="n-b"),
            Neuron.create(type=NeuronType.CONCEPT, content="missing env", neuron_id="n-c"),
        ]
        for n in neurons:
            await store.add_neuron(n)

        synapses = [
            Synapse.create(
                source_id="n-a", target_id="n-b", type=SynapseType.CAUSED_BY, weight=0.8
            ),
            Synapse.create(
                source_id="n-b", target_id="n-c", type=SynapseType.CAUSED_BY, weight=0.9
            ),
        ]
        for s in synapses:
            await store.add_synapse(s)

        result = await find_transitive_closures(store)
        assert len(result) == 1
        syn = result[0]
        assert syn.source_id == "n-a"
        assert syn.target_id == "n-c"
        assert syn.type == SynapseType.CAUSED_BY
        assert syn.metadata["_enriched"] is True
        assert syn.metadata["_chain"] == ["n-a", "n-b", "n-c"]

    async def test_weight_calculation(self, store: InMemoryStorage) -> None:
        """Weight = 0.5 * min(w_AB, w_BC)."""
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content="a", neuron_id="n-a"),
            Neuron.create(type=NeuronType.CONCEPT, content="b", neuron_id="n-b"),
            Neuron.create(type=NeuronType.CONCEPT, content="c", neuron_id="n-c"),
        ]
        for n in neurons:
            await store.add_neuron(n)

        await store.add_synapse(
            Synapse.create(source_id="n-a", target_id="n-b", type=SynapseType.CAUSED_BY, weight=0.6)
        )
        await store.add_synapse(
            Synapse.create(source_id="n-b", target_id="n-c", type=SynapseType.CAUSED_BY, weight=0.4)
        )

        result = await find_transitive_closures(store)
        assert len(result) == 1
        expected_weight = 0.5 * min(0.6, 0.4)
        assert abs(result[0].weight - expected_weight) < 1e-9

    async def test_no_duplicate_when_direct_exists(self, store: InMemoryStorage) -> None:
        """No transitive synapse when A->C already exists."""
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content="a", neuron_id="n-a"),
            Neuron.create(type=NeuronType.CONCEPT, content="b", neuron_id="n-b"),
            Neuron.create(type=NeuronType.CONCEPT, content="c", neuron_id="n-c"),
        ]
        for n in neurons:
            await store.add_neuron(n)

        synapses = [
            Synapse.create(
                source_id="n-a", target_id="n-b", type=SynapseType.CAUSED_BY, weight=0.8
            ),
            Synapse.create(
                source_id="n-b", target_id="n-c", type=SynapseType.CAUSED_BY, weight=0.9
            ),
            Synapse.create(
                source_id="n-a", target_id="n-c", type=SynapseType.CAUSED_BY, weight=0.7
            ),
        ]
        for s in synapses:
            await store.add_synapse(s)

        result = await find_transitive_closures(store)
        assert len(result) == 0

    async def test_skip_self_loops(self, store: InMemoryStorage) -> None:
        """A->B->A should not create A->A self-loop."""
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content="a", neuron_id="n-a"),
            Neuron.create(type=NeuronType.CONCEPT, content="b", neuron_id="n-b"),
        ]
        for n in neurons:
            await store.add_neuron(n)

        synapses = [
            Synapse.create(
                source_id="n-a", target_id="n-b", type=SynapseType.CAUSED_BY, weight=0.8
            ),
            Synapse.create(
                source_id="n-b", target_id="n-a", type=SynapseType.CAUSED_BY, weight=0.7
            ),
        ]
        for s in synapses:
            await store.add_synapse(s)

        result = await find_transitive_closures(store)
        assert len(result) == 0

    async def test_respect_max_synapses(self, store: InMemoryStorage) -> None:
        """Should stop creating synapses at max_synapses limit."""
        # Build 5 independent chains: X_i -> Y_i -> Z_i
        for i in range(5):
            neurons = [
                Neuron.create(type=NeuronType.CONCEPT, content=f"x{i}", neuron_id=f"x-{i}"),
                Neuron.create(type=NeuronType.CONCEPT, content=f"y{i}", neuron_id=f"y-{i}"),
                Neuron.create(type=NeuronType.CONCEPT, content=f"z{i}", neuron_id=f"z-{i}"),
            ]
            for n in neurons:
                await store.add_neuron(n)
            await store.add_synapse(
                Synapse.create(
                    source_id=f"x-{i}",
                    target_id=f"y-{i}",
                    type=SynapseType.CAUSED_BY,
                    weight=0.8,
                )
            )
            await store.add_synapse(
                Synapse.create(
                    source_id=f"y-{i}",
                    target_id=f"z-{i}",
                    type=SynapseType.CAUSED_BY,
                    weight=0.9,
                )
            )

        result = await find_transitive_closures(store, max_synapses=3)
        assert len(result) == 3

    async def test_empty_graph(self, store: InMemoryStorage) -> None:
        """Empty graph returns empty list."""
        result = await find_transitive_closures(store)
        assert result == []

    async def test_no_causal_synapses(self, store: InMemoryStorage) -> None:
        """Non-CAUSED_BY synapses are ignored."""
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content="a", neuron_id="n-a"),
            Neuron.create(type=NeuronType.CONCEPT, content="b", neuron_id="n-b"),
            Neuron.create(type=NeuronType.CONCEPT, content="c", neuron_id="n-c"),
        ]
        for n in neurons:
            await store.add_neuron(n)

        await store.add_synapse(
            Synapse.create(
                source_id="n-a", target_id="n-b", type=SynapseType.RELATED_TO, weight=0.8
            )
        )
        await store.add_synapse(
            Synapse.create(
                source_id="n-b", target_id="n-c", type=SynapseType.RELATED_TO, weight=0.9
            )
        )

        result = await find_transitive_closures(store)
        assert result == []


# ── Cross-Cluster Link Tests ─────────────────────────────────────


class TestFindCrossClusterLinks:
    """Tests for find_cross_cluster_links function."""

    async def test_shared_entity_creates_link(self, store: InMemoryStorage) -> None:
        """Two clusters sharing an entity neuron create a RELATED_TO synapse."""
        shared_neuron = Neuron.create(
            type=NeuronType.ENTITY, content="shared", neuron_id="n-shared"
        )
        anchor_a = Neuron.create(
            type=NeuronType.CONCEPT, content="anchor-a", neuron_id="n-anchor-a"
        )
        anchor_b = Neuron.create(
            type=NeuronType.CONCEPT, content="anchor-b", neuron_id="n-anchor-b"
        )
        for n in [shared_neuron, anchor_a, anchor_b]:
            await store.add_neuron(n)

        # Cluster A: tags {"python", "api"}, includes shared neuron
        fiber_a = Fiber.create(
            neuron_ids={"n-anchor-a", "n-shared"},
            synapse_ids=set(),
            anchor_neuron_id="n-anchor-a",
            tags={"python", "api"},
            fiber_id="f-a",
        )
        # Cluster B: tags {"rust", "cli"}, includes shared neuron
        fiber_b = Fiber.create(
            neuron_ids={"n-anchor-b", "n-shared"},
            synapse_ids=set(),
            anchor_neuron_id="n-anchor-b",
            tags={"rust", "cli"},
            fiber_id="f-b",
        )
        await store.add_fiber(fiber_a)
        await store.add_fiber(fiber_b)

        result = await find_cross_cluster_links(store)
        assert len(result) == 1
        syn = result[0]
        assert syn.type == SynapseType.RELATED_TO
        assert syn.weight == pytest.approx(0.3)
        assert syn.metadata["_enriched"] is True
        assert syn.metadata["_cross_cluster"] is True

    async def test_no_link_without_shared_neurons(self, store: InMemoryStorage) -> None:
        """Two clusters with no shared neurons produce no links."""
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content="a1", neuron_id="n-a1"),
            Neuron.create(type=NeuronType.CONCEPT, content="b1", neuron_id="n-b1"),
        ]
        for n in neurons:
            await store.add_neuron(n)

        fiber_a = Fiber.create(
            neuron_ids={"n-a1"},
            synapse_ids=set(),
            anchor_neuron_id="n-a1",
            tags={"python", "api"},
            fiber_id="f-a",
        )
        fiber_b = Fiber.create(
            neuron_ids={"n-b1"},
            synapse_ids=set(),
            anchor_neuron_id="n-b1",
            tags={"rust", "cli"},
            fiber_id="f-b",
        )
        await store.add_fiber(fiber_a)
        await store.add_fiber(fiber_b)

        result = await find_cross_cluster_links(store)
        assert len(result) == 0

    async def test_single_cluster_no_links(self, store: InMemoryStorage) -> None:
        """Fibers that cluster together (high tag overlap) produce no cross-cluster links."""
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content="a", neuron_id="n-a"),
            Neuron.create(type=NeuronType.CONCEPT, content="b", neuron_id="n-b"),
        ]
        for n in neurons:
            await store.add_neuron(n)

        # High tag overlap -> same cluster
        fiber_a = Fiber.create(
            neuron_ids={"n-a"},
            synapse_ids=set(),
            anchor_neuron_id="n-a",
            tags={"python", "api", "web"},
            fiber_id="f-a",
        )
        fiber_b = Fiber.create(
            neuron_ids={"n-b"},
            synapse_ids=set(),
            anchor_neuron_id="n-b",
            tags={"python", "api", "web"},
            fiber_id="f-b",
        )
        await store.add_fiber(fiber_a)
        await store.add_fiber(fiber_b)

        result = await find_cross_cluster_links(store)
        assert len(result) == 0

    async def test_no_fibers(self, store: InMemoryStorage) -> None:
        """Empty storage returns empty list."""
        result = await find_cross_cluster_links(store)
        assert result == []

    async def test_fibers_without_tags_ignored(self, store: InMemoryStorage) -> None:
        """Fibers with no tags are not considered for clustering."""
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content="a", neuron_id="n-a"),
            Neuron.create(type=NeuronType.CONCEPT, content="b", neuron_id="n-b"),
        ]
        for n in neurons:
            await store.add_neuron(n)

        fiber_a = Fiber.create(
            neuron_ids={"n-a"},
            synapse_ids=set(),
            anchor_neuron_id="n-a",
            fiber_id="f-a",
        )
        fiber_b = Fiber.create(
            neuron_ids={"n-b"},
            synapse_ids=set(),
            anchor_neuron_id="n-b",
            fiber_id="f-b",
        )
        await store.add_fiber(fiber_a)
        await store.add_fiber(fiber_b)

        result = await find_cross_cluster_links(store)
        assert result == []


# ── Full Enrich Tests ─────────────────────────────────────────────


class TestEnrich:
    """Tests for the combined enrich() function."""

    async def test_combines_both_strategies(self, store: InMemoryStorage) -> None:
        """enrich() returns results from transitive closure and cross-cluster linking."""
        # Set up transitive closure: A->B->C
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content="a", neuron_id="n-a"),
            Neuron.create(type=NeuronType.CONCEPT, content="b", neuron_id="n-b"),
            Neuron.create(type=NeuronType.CONCEPT, content="c", neuron_id="n-c"),
            Neuron.create(type=NeuronType.ENTITY, content="shared", neuron_id="n-shared"),
            Neuron.create(type=NeuronType.CONCEPT, content="anchor-x", neuron_id="n-anchor-x"),
            Neuron.create(type=NeuronType.CONCEPT, content="anchor-y", neuron_id="n-anchor-y"),
        ]
        for n in neurons:
            await store.add_neuron(n)

        # Causal chain for transitive closure
        await store.add_synapse(
            Synapse.create(source_id="n-a", target_id="n-b", type=SynapseType.CAUSED_BY, weight=0.8)
        )
        await store.add_synapse(
            Synapse.create(source_id="n-b", target_id="n-c", type=SynapseType.CAUSED_BY, weight=0.9)
        )

        # Two clusters sharing n-shared for cross-cluster
        fiber_x = Fiber.create(
            neuron_ids={"n-anchor-x", "n-shared"},
            synapse_ids=set(),
            anchor_neuron_id="n-anchor-x",
            tags={"python", "api"},
            fiber_id="f-x",
        )
        fiber_y = Fiber.create(
            neuron_ids={"n-anchor-y", "n-shared"},
            synapse_ids=set(),
            anchor_neuron_id="n-anchor-y",
            tags={"rust", "cli"},
            fiber_id="f-y",
        )
        await store.add_fiber(fiber_x)
        await store.add_fiber(fiber_y)

        result = await enrich(store)
        assert isinstance(result, EnrichmentResult)
        assert len(result.transitive_synapses) == 1
        assert len(result.cross_cluster_synapses) == 1
        assert result.total_synapses == 2

    async def test_empty_graph_returns_empty_result(self, store: InMemoryStorage) -> None:
        """Empty storage returns EnrichmentResult with zero synapses."""
        result = await enrich(store)
        assert isinstance(result, EnrichmentResult)
        assert result.transitive_synapses == []
        assert result.cross_cluster_synapses == []
        assert result.total_synapses == 0


# ── EnrichmentResult Tests ────────────────────────────────────────


class TestEnrichmentResult:
    """Tests for EnrichmentResult data structure."""

    def test_frozen(self) -> None:
        """EnrichmentResult should be immutable."""
        result = EnrichmentResult()
        with pytest.raises(AttributeError):
            result.transitive_synapses = []  # type: ignore[misc]

    def test_total_synapses(self) -> None:
        """total_synapses returns sum of both lists."""
        s1 = Synapse.create(source_id="a", target_id="b", type=SynapseType.CAUSED_BY)
        s2 = Synapse.create(source_id="c", target_id="d", type=SynapseType.RELATED_TO)
        s3 = Synapse.create(source_id="e", target_id="f", type=SynapseType.RELATED_TO)
        result = EnrichmentResult(transitive_synapses=[s1], cross_cluster_synapses=[s2, s3])
        assert result.total_synapses == 3

    def test_defaults_empty(self) -> None:
        """Default EnrichmentResult has empty lists."""
        result = EnrichmentResult()
        assert result.transitive_synapses == []
        assert result.cross_cluster_synapses == []
        assert result.total_synapses == 0
