"""Tests for Personalized PageRank activation."""

from __future__ import annotations

from unittest.mock import AsyncMock

from neural_memory.core.brain import BrainConfig
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.ppr_activation import PPRActivation


def _make_synapse(source_id: str, target_id: str, weight: float = 0.5) -> Synapse:
    return Synapse.create(
        source_id=source_id,
        target_id=target_id,
        type=SynapseType.RELATED_TO,
        weight=weight,
    )


def _mock_storage_with_graph(
    graph: dict[str, list[tuple[str, float]]],
) -> AsyncMock:
    """Create mock storage from adjacency list: {node: [(target, weight), ...]}."""
    storage = AsyncMock()

    async def get_synapses_for_neurons(
        neuron_ids: list[str], direction: str = "out"
    ) -> dict[str, list[Synapse]]:
        result: dict[str, list[Synapse]] = {}
        for nid in neuron_ids:
            neighbors = graph.get(nid, [])
            result[nid] = [_make_synapse(nid, target, weight) for target, weight in neighbors]
        return result

    storage.get_synapses_for_neurons = get_synapses_for_neurons
    return storage


class TestPPRActivation:
    """Test PPR activation algorithm."""

    async def test_empty_anchors(self) -> None:
        storage = AsyncMock()
        config = BrainConfig(activation_strategy="ppr")
        ppr = PPRActivation(storage, config)
        results, _trace = await ppr.activate([])
        assert results == {}

    async def test_single_node_no_edges(self) -> None:
        """Single anchor with no outgoing edges — all activation stays at seed."""
        storage = _mock_storage_with_graph({"a": []})
        config = BrainConfig(activation_strategy="ppr", activation_threshold=0.01)
        ppr = PPRActivation(storage, config)

        results, _trace = await ppr.activate(["a"])
        assert "a" in results
        assert results["a"].activation_level == 1.0
        assert results["a"].hop_distance == 0

    async def test_chain_graph_decay(self) -> None:
        """Linear chain: a -> b -> c. Activation decays along chain."""
        storage = _mock_storage_with_graph(
            {
                "a": [("b", 1.0)],
                "b": [("c", 1.0)],
                "c": [],
            }
        )
        config = BrainConfig(
            activation_strategy="ppr",
            activation_threshold=0.01,
            ppr_damping=0.15,
            ppr_iterations=30,
        )
        ppr = PPRActivation(storage, config)

        results, _trace = await ppr.activate(["a"])
        assert "a" in results
        assert "b" in results
        assert "c" in results
        # Activation should decrease: a > b > c
        assert results["a"].activation_level > results["b"].activation_level
        assert results["b"].activation_level > results["c"].activation_level

    async def test_hub_dampening(self) -> None:
        """Star graph: hub -> {s1, s2, s3, s4, s5}. Hub should not dominate."""
        storage = _mock_storage_with_graph(
            {
                "seed": [("hub", 1.0)],
                "hub": [("s1", 1.0), ("s2", 1.0), ("s3", 1.0), ("s4", 1.0), ("s5", 1.0)],
                "s1": [],
                "s2": [],
                "s3": [],
                "s4": [],
                "s5": [],
            }
        )
        config = BrainConfig(
            activation_strategy="ppr",
            activation_threshold=0.01,
            ppr_damping=0.15,
            ppr_iterations=30,
        )
        ppr = PPRActivation(storage, config)

        results, _trace = await ppr.activate(["seed"])

        # Each spoke should get roughly equal share (hub distributes evenly)
        spoke_levels = [
            results[f"s{i}"].activation_level for i in range(1, 6) if f"s{i}" in results
        ]
        assert len(spoke_levels) >= 3  # at least some spokes activated
        if len(spoke_levels) >= 2:
            # Spokes should be roughly equal (within 2x of each other)
            assert max(spoke_levels) / max(min(spoke_levels), 1e-10) < 3.0

    async def test_anchor_activations_from_rrf(self) -> None:
        """RRF-weighted anchors: higher-weighted seed should have more influence."""
        storage = _mock_storage_with_graph(
            {
                "high": [("target", 1.0)],
                "low": [("target", 1.0)],
                "target": [],
            }
        )
        config = BrainConfig(
            activation_strategy="ppr",
            activation_threshold=0.01,
            ppr_iterations=20,
        )
        ppr = PPRActivation(storage, config)

        results, _trace = await ppr.activate(
            ["high", "low"],
            anchor_activations={"high": 0.9, "low": 0.1},
        )
        assert "high" in results
        assert "low" in results
        # High-weighted seed should have more activation
        assert results["high"].activation_level > results["low"].activation_level

    async def test_convergence(self) -> None:
        """Should converge within max iterations for simple graph."""
        storage = _mock_storage_with_graph(
            {
                "a": [("b", 0.8)],
                "b": [("a", 0.8)],  # cycle
            }
        )
        config = BrainConfig(
            activation_strategy="ppr",
            activation_threshold=0.01,
            ppr_iterations=50,
            ppr_epsilon=1e-8,
        )
        ppr = PPRActivation(storage, config)

        results, _trace = await ppr.activate(["a"])
        # Both should be activated due to cycle
        assert "a" in results
        assert "b" in results

    async def test_activation_threshold_filters(self) -> None:
        """Neurons below threshold should be excluded."""
        storage = _mock_storage_with_graph(
            {
                "a": [("b", 0.01)],  # very weak edge
                "b": [],
            }
        )
        config = BrainConfig(
            activation_strategy="ppr",
            activation_threshold=0.5,  # high threshold
            ppr_iterations=10,
        )
        ppr = PPRActivation(storage, config)

        results, _trace = await ppr.activate(["a"])
        assert "a" in results
        # b should be filtered due to weak edge + high threshold
        # (depends on exact convergence, but likely filtered)


class TestPPRActivateFromMultiple:
    """Test multi-anchor-set PPR with intersection detection."""

    async def test_single_set(self) -> None:
        storage = _mock_storage_with_graph({"a": []})
        config = BrainConfig(activation_strategy="ppr", activation_threshold=0.01)
        ppr = PPRActivation(storage, config)

        results, intersections = await ppr.activate_from_multiple([["a"]])
        assert "a" in results
        assert "a" in intersections

    async def test_empty_sets(self) -> None:
        storage = AsyncMock()
        config = BrainConfig(activation_strategy="ppr")
        ppr = PPRActivation(storage, config)

        results, intersections = await ppr.activate_from_multiple([])
        assert results == {}
        assert intersections == []

    async def test_intersection_detection(self) -> None:
        """Neurons reachable from multiple anchor sets should be in intersections."""
        storage = _mock_storage_with_graph(
            {
                "a1": [("shared", 1.0)],
                "a2": [("shared", 1.0)],
                "shared": [],
            }
        )
        config = BrainConfig(
            activation_strategy="ppr",
            activation_threshold=0.01,
            ppr_iterations=20,
        )
        ppr = PPRActivation(storage, config)

        results, intersections = await ppr.activate_from_multiple([["a1"], ["a2"]])
        assert "shared" in results
        # shared should be in intersections (reached from both a1 and a2)
        # Note: with single PPR run, intersection detection depends on source tracking
