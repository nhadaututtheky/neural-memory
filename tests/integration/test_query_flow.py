"""Integration tests for query flow."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


class TestQueryFlow:
    """Integration tests for the full query flow."""

    @pytest.fixture
    async def storage_with_memories(self) -> InMemoryStorage:
        """Create storage populated with test memories."""
        storage = InMemoryStorage()

        config = BrainConfig(
            activation_threshold=0.1,
            max_spread_hops=4,
        )
        brain = Brain.create(name="test_brain", config=config)
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        encoder = MemoryEncoder(storage, config)

        # Encode several memories
        await encoder.encode(
            "Met with Alice at the coffee shop to discuss API design",
            timestamp=datetime(2024, 2, 3, 15, 0),
        )

        await encoder.encode(
            "Alice suggested adding rate limiting to the API",
            timestamp=datetime(2024, 2, 3, 15, 30),
        )

        await encoder.encode(
            "Completed the authentication module",
            timestamp=datetime(2024, 2, 4, 10, 0),
        )

        return storage

    @pytest.mark.asyncio
    async def test_basic_query(self, storage_with_memories: InMemoryStorage) -> None:
        """Test a basic query returns results."""
        brain = await storage_with_memories.get_brain(
            storage_with_memories._current_brain_id  # type: ignore
        )
        assert brain is not None

        pipeline = ReflexPipeline(storage_with_memories, brain.config)

        result = await pipeline.query(
            "What did Alice suggest?",
            reference_time=datetime(2024, 2, 4, 16, 0),
        )

        assert result.neurons_activated > 0
        assert result.latency_ms >= 0
        assert result.context  # Should have some context

    @pytest.mark.asyncio
    async def test_query_with_time_constraint(self, storage_with_memories: InMemoryStorage) -> None:
        """Test query with temporal constraint."""
        brain = await storage_with_memories.get_brain(
            storage_with_memories._current_brain_id  # type: ignore
        )
        assert brain is not None

        pipeline = ReflexPipeline(storage_with_memories, brain.config)

        result = await pipeline.query(
            "What happened yesterday afternoon?",
            reference_time=datetime(2024, 2, 4, 16, 0),
        )

        # Query should complete without error and return valid structure
        assert result.confidence >= 0
        assert result.latency_ms >= 0
        assert result.depth_used is not None

    @pytest.mark.asyncio
    async def test_query_with_entity(self, storage_with_memories: InMemoryStorage) -> None:
        """Test query mentioning a specific entity."""
        brain = await storage_with_memories.get_brain(
            storage_with_memories._current_brain_id  # type: ignore
        )
        assert brain is not None

        pipeline = ReflexPipeline(storage_with_memories, brain.config)

        result = await pipeline.query(
            "Tell me about Alice",
            reference_time=datetime(2024, 2, 4, 16, 0),
        )

        assert result.neurons_activated > 0

    @pytest.mark.asyncio
    async def test_query_depth_levels(self, storage_with_memories: InMemoryStorage) -> None:
        """Test different depth levels."""
        brain = await storage_with_memories.get_brain(
            storage_with_memories._current_brain_id  # type: ignore
        )
        assert brain is not None

        pipeline = ReflexPipeline(storage_with_memories, brain.config)

        # Instant (shallow)
        instant = await pipeline.query(
            "Who?", depth=DepthLevel.INSTANT, reference_time=datetime(2024, 2, 4, 16, 0)
        )
        assert instant.depth_used == DepthLevel.INSTANT

        # Deep
        deep = await pipeline.query(
            "Why?", depth=DepthLevel.DEEP, reference_time=datetime(2024, 2, 4, 16, 0)
        )
        assert deep.depth_used == DepthLevel.DEEP

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Sufficiency gate may reject thin signal in InMemoryStorage")
    async def test_query_returns_context(self, storage_with_memories: InMemoryStorage) -> None:
        """Test that query returns formatted context."""
        brain = await storage_with_memories.get_brain(
            storage_with_memories._current_brain_id  # type: ignore
        )
        assert brain is not None

        pipeline = ReflexPipeline(storage_with_memories, brain.config)

        result = await pipeline.query(
            "What happened?",
            max_tokens=1000,
            reference_time=datetime(2024, 2, 4, 16, 0),
        )

        assert isinstance(result.context, str)
        # Context should have some structure
        assert len(result.context) > 0

    @pytest.mark.asyncio
    async def test_query_subgraph_extraction(self, storage_with_memories: InMemoryStorage) -> None:
        """Test that query extracts relevant subgraph."""
        brain = await storage_with_memories.get_brain(
            storage_with_memories._current_brain_id  # type: ignore
        )
        assert brain is not None

        pipeline = ReflexPipeline(storage_with_memories, brain.config)

        result = await pipeline.query(
            "Coffee shop meeting",
            reference_time=datetime(2024, 2, 4, 16, 0),
        )

        assert result.subgraph is not None
        assert isinstance(result.subgraph.neuron_ids, list)
        assert isinstance(result.subgraph.synapse_ids, list)

    @pytest.mark.asyncio
    async def test_empty_query(self, storage_with_memories: InMemoryStorage) -> None:
        """Test query with minimal content."""
        brain = await storage_with_memories.get_brain(
            storage_with_memories._current_brain_id  # type: ignore
        )
        assert brain is not None

        pipeline = ReflexPipeline(storage_with_memories, brain.config)

        result = await pipeline.query(
            "xyz123",  # Unlikely to match anything
            reference_time=datetime(2024, 2, 4, 16, 0),
        )

        # Should still return a valid result structure
        assert result.confidence >= 0
        assert result.context is not None


class TestHybridFallback:
    """Tests for hybrid reflex + classic fallback behavior."""

    @pytest.fixture
    async def storage_with_partial_fibers(self) -> InMemoryStorage:
        """
        Create a graph where some neurons are in fibers and some are only
        reachable via synapses (not in any fiber pathway).

        Graph structure:
            [Alice] --fiber--> [meeting] --fiber--> [JWT]    (in fiber pathway)
            [Alice] --synapse-> [deploy] --synapse-> [staging]  (outside any fiber)

        Querying "Alice" should find both [JWT] (via fiber)
        and [deploy]/[staging] (via classic BFS discovery).
        """
        config = BrainConfig(activation_threshold=0.05, max_spread_hops=4)
        storage = InMemoryStorage()
        brain = Brain.create(name="hybrid_test", config=config)
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        neurons = [
            Neuron.create(type=NeuronType.ENTITY, content="Alice", neuron_id="nA"),
            Neuron.create(type=NeuronType.CONCEPT, content="meeting", neuron_id="nB"),
            Neuron.create(type=NeuronType.CONCEPT, content="JWT", neuron_id="nC"),
            Neuron.create(type=NeuronType.ACTION, content="deploy", neuron_id="nD"),
            Neuron.create(type=NeuronType.STATE, content="staging", neuron_id="nE"),
        ]
        for n in neurons:
            await storage.add_neuron(n)

        synapses = [
            Synapse.create("nA", "nB", SynapseType.RELATED_TO, weight=0.9, synapse_id="sAB"),
            Synapse.create("nB", "nC", SynapseType.RELATED_TO, weight=0.8, synapse_id="sBC"),
            Synapse.create("nA", "nD", SynapseType.INVOLVES, weight=0.8, synapse_id="sAD"),
            Synapse.create("nD", "nE", SynapseType.LEADS_TO, weight=0.7, synapse_id="sDE"),
        ]
        for s in synapses:
            await storage.add_synapse(s)

        # Fiber only covers Alice -> meeting -> JWT
        fiber = Fiber.create(
            neuron_ids={"nA", "nB", "nC"},
            synapse_ids={"sAB", "sBC"},
            anchor_neuron_id="nA",
            pathway=["nA", "nB", "nC"],
            fiber_id="f1",
        )
        fiber = fiber.conduct(
            conducted_at=utcnow() - timedelta(hours=1),
            reinforce=False,
        )
        await storage.add_fiber(fiber)

        return storage

    @pytest.mark.asyncio
    async def test_hybrid_finds_neurons_outside_fibers(
        self, storage_with_partial_fibers: InMemoryStorage
    ) -> None:
        """Reflex mode with hybrid fallback discovers neurons outside fiber pathways."""
        brain = await storage_with_partial_fibers.get_brain(
            storage_with_partial_fibers._current_brain_id  # type: ignore
        )
        assert brain is not None

        # Test the activation engine directly to avoid parser dependency
        from neural_memory.engine.activation import SpreadingActivation
        from neural_memory.engine.reflex_activation import ReflexActivation

        reflex = ReflexActivation(storage_with_partial_fibers, brain.config)
        classic = SpreadingActivation(storage_with_partial_fibers, brain.config)

        fibers = await storage_with_partial_fibers.find_fibers(contains_neuron="nA")

        # Pure reflex: only fiber pathway neurons
        reflex_results = await reflex.activate_trail(
            anchor_neurons=["nA"],
            fibers=fibers,
        )
        reflex_ids = set(reflex_results.keys())

        # Classic BFS: discovers all connected neurons
        classic_results = await classic.activate(["nA"], max_hops=4)
        classic_ids = set(classic_results.keys())

        # Reflex should find fiber neurons
        assert "nA" in reflex_ids
        assert "nB" in reflex_ids or "nC" in reflex_ids

        # Classic should find neurons outside the fiber
        assert "nD" in classic_ids, "Classic should discover nD (outside fiber)"

        # Reflex alone should NOT find nD (it's not in any fiber pathway)
        assert "nD" not in reflex_ids, "Pure reflex should not find nD"

        # Now test the hybrid pipeline method directly
        pipeline = ReflexPipeline(
            storage_with_partial_fibers,
            brain.config,
            use_reflex=True,
        )
        hybrid_activations, intersections, co_acts = await pipeline._reflex_query(
            anchor_sets=[["nA"]],
            reference_time=utcnow(),
        )
        hybrid_ids = set(hybrid_activations.keys())

        # Hybrid should find BOTH fiber neurons AND discovery neurons
        assert "nA" in hybrid_ids
        assert "nD" in hybrid_ids, (
            f"Hybrid should discover nD (outside fiber via classic BFS), got: {hybrid_ids}"
        )

    @pytest.mark.asyncio
    async def test_hybrid_reflex_neurons_rank_higher(
        self, storage_with_partial_fibers: InMemoryStorage
    ) -> None:
        """Fiber-pathway neurons get higher activation than BFS-discovered neurons."""
        brain = await storage_with_partial_fibers.get_brain(
            storage_with_partial_fibers._current_brain_id  # type: ignore
        )
        assert brain is not None

        pipeline = ReflexPipeline(
            storage_with_partial_fibers,
            brain.config,
            use_reflex=True,
        )
        activations, _, _ = await pipeline._reflex_query(
            anchor_sets=[["nA"]],
            reference_time=utcnow(),
        )

        # nB is in fiber pathway (reflex primary)
        # nD is discovered via classic BFS (dampened)
        if "nB" in activations and "nD" in activations:
            assert activations["nB"].activation_level >= activations["nD"].activation_level, (
                f"Fiber neuron nB ({activations['nB'].activation_level:.3f}) should rank "
                f">= discovery neuron nD ({activations['nD'].activation_level:.3f})"
            )

    @pytest.mark.asyncio
    async def test_hybrid_no_fibers_falls_back_to_classic(
        self, storage_with_partial_fibers: InMemoryStorage
    ) -> None:
        """When no fibers contain anchor neurons, hybrid falls back entirely to classic."""
        brain = await storage_with_partial_fibers.get_brain(
            storage_with_partial_fibers._current_brain_id  # type: ignore
        )
        assert brain is not None

        pipeline = ReflexPipeline(
            storage_with_partial_fibers,
            brain.config,
            use_reflex=True,
        )

        # nD is not in any fiber -- should fall back to pure classic
        activations, intersections, co_acts = await pipeline._reflex_query(
            anchor_sets=[["nD"]],
            reference_time=utcnow(),
        )

        # Should find nD and its neighbor nE via classic activation
        assert "nD" in activations
        assert "nE" in activations
        # No co-activations in pure classic fallback
        assert len(co_acts) == 0
