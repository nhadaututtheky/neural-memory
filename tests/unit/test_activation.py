"""Unit tests for spreading activation."""

from __future__ import annotations

from datetime import timedelta

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.activation import ActivationResult, SpreadingActivation
from neural_memory.engine.reflex_activation import CoActivation, ReflexActivation
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


class TestSpreadingActivation:
    """Tests for SpreadingActivation class."""

    @pytest.fixture
    def config(self) -> BrainConfig:
        """Create test config."""
        return BrainConfig(
            activation_threshold=0.1,
            max_spread_hops=3,
        )

    @pytest.fixture
    async def storage_with_graph(self, config: BrainConfig) -> InMemoryStorage:
        """Create storage with a simple graph for testing."""
        from neural_memory.core.brain import Brain

        storage = InMemoryStorage()
        brain = Brain.create(name="test", config=config)
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create a simple graph:
        # A -> B -> C -> D
        # |         |
        # +----E----+
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content="A", neuron_id="a"),
            Neuron.create(type=NeuronType.CONCEPT, content="B", neuron_id="b"),
            Neuron.create(type=NeuronType.CONCEPT, content="C", neuron_id="c"),
            Neuron.create(type=NeuronType.CONCEPT, content="D", neuron_id="d"),
            Neuron.create(type=NeuronType.CONCEPT, content="E", neuron_id="e"),
        ]

        for n in neurons:
            await storage.add_neuron(n)

        synapses = [
            Synapse.create("a", "b", SynapseType.RELATED_TO, weight=0.8, synapse_id="ab"),
            Synapse.create("b", "c", SynapseType.RELATED_TO, weight=0.8, synapse_id="bc"),
            Synapse.create("c", "d", SynapseType.RELATED_TO, weight=0.8, synapse_id="cd"),
            Synapse.create("a", "e", SynapseType.RELATED_TO, weight=0.5, synapse_id="ae"),
            Synapse.create("e", "c", SynapseType.RELATED_TO, weight=0.5, synapse_id="ec"),
        ]

        for s in synapses:
            await storage.add_synapse(s)

        return storage

    @pytest.mark.asyncio
    async def test_activate_single_anchor(
        self, storage_with_graph: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test activation from a single anchor."""
        activator = SpreadingActivation(storage_with_graph, config)

        results, _trace = await activator.activate(["a"])

        # Anchor should have full activation
        assert "a" in results
        assert results["a"].activation_level == 1.0
        assert results["a"].hop_distance == 0

        # Direct neighbor should have decayed activation
        assert "b" in results
        assert results["b"].activation_level < 1.0
        assert results["b"].hop_distance == 1

    @pytest.mark.asyncio
    async def test_activation_decays_with_distance(
        self, storage_with_graph: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test that activation decays with distance."""
        activator = SpreadingActivation(storage_with_graph, config)

        results, _trace = await activator.activate(["a"], max_hops=4)

        # Each hop should have lower activation
        if "b" in results and "c" in results:
            assert results["b"].activation_level > results["c"].activation_level

        if "c" in results and "d" in results:
            assert results["c"].activation_level > results["d"].activation_level

    @pytest.mark.asyncio
    async def test_activate_respects_max_hops(
        self, storage_with_graph: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test that activation respects max_hops."""
        activator = SpreadingActivation(storage_with_graph, config)

        results, _trace = await activator.activate(["a"], max_hops=1)

        # Should only reach 1-hop neighbors
        assert "a" in results
        assert "b" in results or "e" in results

        # D is 3 hops away, should not be reached
        assert (
            "d" not in results
            or results.get("d", ActivationResult("", 0, 0, [], "")).hop_distance <= 1
        )

    @pytest.mark.asyncio
    async def test_activate_multiple_anchors(
        self, storage_with_graph: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test activation from multiple anchors."""
        activator = SpreadingActivation(storage_with_graph, config)

        results, _trace = await activator.activate(["a", "d"])

        # Both anchors should be activated
        assert "a" in results
        assert "d" in results

        # C should be reachable from both
        assert "c" in results

    @pytest.mark.asyncio
    async def test_activate_from_multiple_sets_finds_intersection(
        self, storage_with_graph: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test that activating from multiple sets finds intersections."""
        activator = SpreadingActivation(storage_with_graph, config)

        results, intersections = await activator.activate_from_multiple([["a"], ["d"]], max_hops=4)

        # C is reachable from both A and D
        assert "c" in intersections or len(intersections) > 0

    @pytest.mark.asyncio
    async def test_get_activated_subgraph(
        self, storage_with_graph: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test extracting subgraph from activations."""
        activator = SpreadingActivation(storage_with_graph, config)

        activations, _trace = await activator.activate(["a"])
        neuron_ids, synapse_ids = await activator.get_activated_subgraph(
            activations, min_activation=0.1
        )

        assert len(neuron_ids) > 0
        assert "a" in neuron_ids

    @pytest.mark.asyncio
    async def test_empty_anchors(
        self, storage_with_graph: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test activation with empty anchor list."""
        activator = SpreadingActivation(storage_with_graph, config)

        results, _trace = await activator.activate([])

        assert results == {}

    @pytest.mark.asyncio
    async def test_nonexistent_anchor(
        self, storage_with_graph: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test activation with nonexistent anchor."""
        activator = SpreadingActivation(storage_with_graph, config)

        results, _trace = await activator.activate(["nonexistent"])

        assert results == {}


class TestReflexActivation:
    """Tests for ReflexActivation class."""

    @pytest.fixture
    def config(self) -> BrainConfig:
        """Create test config."""
        return BrainConfig(
            activation_threshold=0.1,
            max_spread_hops=3,
        )

    @pytest.fixture
    async def storage_with_fibers(self, config: BrainConfig) -> InMemoryStorage:
        """Create storage with fibers for testing reflex activation."""
        from neural_memory.core.brain import Brain

        storage = InMemoryStorage()
        brain = Brain.create(name="test", config=config)
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create neurons
        neurons = [
            Neuron.create(type=NeuronType.TIME, content="morning", neuron_id="time1"),
            Neuron.create(type=NeuronType.ENTITY, content="coffee", neuron_id="entity1"),
            Neuron.create(type=NeuronType.ACTION, content="drink", neuron_id="action1"),
            Neuron.create(type=NeuronType.SPATIAL, content="cafe", neuron_id="spatial1"),
            Neuron.create(type=NeuronType.CONCEPT, content="happiness", neuron_id="concept1"),
        ]

        for n in neurons:
            await storage.add_neuron(n)

        # Create synapses
        synapses = [
            Synapse.create(
                "time1", "entity1", SynapseType.HAPPENED_AT, weight=0.9, synapse_id="s1"
            ),
            Synapse.create("entity1", "action1", SynapseType.INVOLVES, weight=0.8, synapse_id="s2"),
            Synapse.create(
                "action1", "spatial1", SynapseType.AT_LOCATION, weight=0.7, synapse_id="s3"
            ),
            Synapse.create("spatial1", "concept1", SynapseType.EVOKES, weight=0.6, synapse_id="s4"),
        ]

        for s in synapses:
            await storage.add_synapse(s)

        # Create a fiber with pathway
        fiber = Fiber.create(
            neuron_ids={"time1", "entity1", "action1", "spatial1", "concept1"},
            synapse_ids={"s1", "s2", "s3", "s4"},
            anchor_neuron_id="time1",
            pathway=["time1", "entity1", "action1", "spatial1", "concept1"],
            fiber_id="fiber1",
        )
        # Set conductivity and last_conducted
        fiber = fiber.conduct(conducted_at=utcnow() - timedelta(hours=1))
        await storage.add_fiber(fiber)

        return storage

    @pytest.mark.asyncio
    async def test_activate_trail_basic(
        self, storage_with_fibers: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test basic trail activation through fiber pathway."""
        fibers = await storage_with_fibers.find_fibers()
        reflex = ReflexActivation(storage_with_fibers, config)

        results, _trace = await reflex.activate_trail(
            anchor_neurons=["time1"],
            fibers=fibers,
        )

        # Anchor should have full activation
        assert "time1" in results
        assert results["time1"].activation_level == 1.0

        # Neurons in pathway should be activated
        assert "entity1" in results
        assert "action1" in results

        # Activation should decay along the pathway
        assert results["entity1"].activation_level > results["action1"].activation_level

    @pytest.mark.asyncio
    async def test_trail_decay_uses_fiber_conductivity(
        self, storage_with_fibers: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test that trail decay considers fiber conductivity."""
        # Get fiber and reduce conductivity
        fibers = await storage_with_fibers.find_fibers()
        low_conductivity_fiber = fibers[0].with_conductivity(0.5)
        await storage_with_fibers.update_fiber(low_conductivity_fiber)

        reflex = ReflexActivation(storage_with_fibers, config)
        updated_fibers = await storage_with_fibers.find_fibers()

        results, _trace = await reflex.activate_trail(
            anchor_neurons=["time1"],
            fibers=updated_fibers,
        )

        # With lower conductivity, activation should decay faster
        # entity1 should have lower activation than with full conductivity
        assert results["entity1"].activation_level < 0.9  # Would be ~0.85 with full conductivity

    @pytest.mark.asyncio
    async def test_time_factor_affects_activation(
        self, storage_with_fibers: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test that time factor affects activation levels."""
        reflex = ReflexActivation(storage_with_fibers, config)

        # Get fiber and set it as recently conducted
        fibers = await storage_with_fibers.find_fibers()
        recent_fiber = fibers[0].conduct(conducted_at=utcnow())
        await storage_with_fibers.update_fiber(recent_fiber)

        recent_fibers = await storage_with_fibers.find_fibers()

        # Activate with recent fiber
        recent_results, _trace = await reflex.activate_trail(
            anchor_neurons=["time1"],
            fibers=recent_fibers,
            reference_time=utcnow(),
        )

        # Set fiber as old
        old_fiber = fibers[0].conduct(conducted_at=utcnow() - timedelta(days=5))
        await storage_with_fibers.update_fiber(old_fiber)
        old_fibers = await storage_with_fibers.find_fibers()

        # Activate with old fiber
        old_results, _trace = await reflex.activate_trail(
            anchor_neurons=["time1"],
            fibers=old_fibers,
            reference_time=utcnow(),
        )

        # Recent fiber should result in higher activation
        if "entity1" in recent_results and "entity1" in old_results:
            assert (
                recent_results["entity1"].activation_level
                >= old_results["entity1"].activation_level
            )


class TestCoActivation:
    """Tests for CoActivation and co-activation detection."""

    def test_co_activation_creation(self) -> None:
        """Test CoActivation dataclass creation."""
        co_act = CoActivation(
            neuron_ids=frozenset(["a", "b"]),
            temporal_window_ms=500,
            co_fire_count=2,
            binding_strength=0.8,
        )

        assert "a" in co_act.neuron_ids
        assert "b" in co_act.neuron_ids
        assert co_act.temporal_window_ms == 500
        assert co_act.binding_strength == 0.8

    @pytest.fixture
    def config(self) -> BrainConfig:
        return BrainConfig(activation_threshold=0.1)

    @pytest.fixture
    async def storage(self, config: BrainConfig) -> InMemoryStorage:
        from neural_memory.core.brain import Brain

        storage = InMemoryStorage()
        brain = Brain.create(name="test", config=config)
        await storage.save_brain(brain)
        storage.set_brain(brain.id)
        return storage

    @pytest.mark.asyncio
    async def test_find_co_activated(self, storage: InMemoryStorage, config: BrainConfig) -> None:
        """Test finding co-activated neurons."""
        reflex = ReflexActivation(storage, config)

        # Create activation sets where some neurons are activated by multiple anchors
        set1: dict[str, ActivationResult] = {
            "n1": ActivationResult("n1", 0.8, 1, ["a", "n1"], "a"),
            "n2": ActivationResult("n2", 0.6, 2, ["a", "b", "n2"], "a"),
            "n3": ActivationResult("n3", 0.4, 3, ["a", "b", "c", "n3"], "a"),
        }
        set2: dict[str, ActivationResult] = {
            "n2": ActivationResult("n2", 0.7, 1, ["b", "n2"], "b"),
            "n3": ActivationResult("n3", 0.5, 2, ["b", "c", "n3"], "b"),
            "n4": ActivationResult("n4", 0.3, 3, ["b", "c", "d", "n4"], "b"),
        }

        co_activations = reflex.find_co_activated([set1, set2])

        # n2 and n3 should be co-activated (appear in both sets)
        co_activated_ids = {neuron_id for co in co_activations for neuron_id in co.neuron_ids}
        assert "n2" in co_activated_ids
        assert "n3" in co_activated_ids
        assert "n1" not in co_activated_ids  # Only in set1
        assert "n4" not in co_activated_ids  # Only in set2

    @pytest.mark.asyncio
    async def test_binding_strength_calculation(
        self, storage: InMemoryStorage, config: BrainConfig
    ) -> None:
        """Test that binding strength is calculated correctly."""
        reflex = ReflexActivation(storage, config)

        # Neuron activated by all 3 anchor sets
        set1: dict[str, ActivationResult] = {
            "common": ActivationResult("common", 0.8, 1, ["a", "common"], "a"),
        }
        set2: dict[str, ActivationResult] = {
            "common": ActivationResult("common", 0.7, 1, ["b", "common"], "b"),
        }
        set3: dict[str, ActivationResult] = {
            "common": ActivationResult("common", 0.6, 1, ["c", "common"], "c"),
        }

        co_activations = reflex.find_co_activated([set1, set2, set3])

        # common neuron: weighted binding = (0.8 + 0.7 + 0.6) / 3 = 0.7
        assert len(co_activations) == 1
        assert abs(co_activations[0].binding_strength - 0.7) < 0.01
        assert co_activations[0].co_fire_count == 3
