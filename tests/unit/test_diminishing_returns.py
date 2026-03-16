"""Tests for the diminishing returns gate in spreading activation.

Phase 5 of v4.0 Brain Intelligence: stop spreading early when new hops
add insufficient signal.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.engine.activation import (
    ActivationTrace,
    SpreadingActivation,
    should_stop_spreading,
)

# ---------------------------------------------------------------------------
# Pure unit tests for ActivationTrace
# ---------------------------------------------------------------------------


class TestActivationTrace:
    """Test ActivationTrace dataclass."""

    def test_empty_trace(self) -> None:
        trace = ActivationTrace()
        assert trace.total_neurons_activated == 0
        assert trace.max_hop_used == 0
        assert trace.max_hop_allowed == 0
        assert trace.stopped_early is False
        assert trace.stop_reason == ""

    def test_total_neurons_activated(self) -> None:
        trace = ActivationTrace()
        trace.new_neurons_per_hop[0] = 3
        trace.new_neurons_per_hop[1] = 5
        trace.new_neurons_per_hop[2] = 1
        assert trace.total_neurons_activated == 9

    def test_trace_records_max_hop(self) -> None:
        trace = ActivationTrace(max_hop_allowed=4)
        trace.max_hop_used = 2
        assert trace.max_hop_allowed == 4
        assert trace.max_hop_used == 2

    def test_activation_gain_tracking(self) -> None:
        trace = ActivationTrace()
        trace.activation_gain_per_hop[0] = 1.0
        trace.activation_gain_per_hop[1] = 0.5
        assert trace.activation_gain_per_hop[0] == 1.0
        assert trace.activation_gain_per_hop[1] == 0.5


# ---------------------------------------------------------------------------
# Pure unit tests for should_stop_spreading
# ---------------------------------------------------------------------------


class TestShouldStopSpreading:
    """Test the diminishing returns detector function."""

    def _make_trace(self, neurons_per_hop: dict[int, int]) -> ActivationTrace:
        trace = ActivationTrace()
        for hop, count in neurons_per_hop.items():
            trace.new_neurons_per_hop[hop] = count
        return trace

    def test_never_stop_at_grace_hops(self) -> None:
        """Should never stop during grace hops regardless of signal."""
        trace = self._make_trace({0: 1, 1: 0})
        stop, reason = should_stop_spreading(trace, current_hop=1, grace_hops=1)
        assert stop is False

    def test_stop_absolute_too_few_neurons(self) -> None:
        """Stop when previous hop added fewer than min_new_neurons."""
        trace = self._make_trace({0: 10, 1: 1})
        stop, reason = should_stop_spreading(trace, current_hop=2, min_new_neurons=2, grace_hops=1)
        assert stop is True
        assert "added only 1" in reason

    def test_no_stop_when_enough_neurons(self) -> None:
        """Don't stop when previous hop added enough neurons."""
        trace = self._make_trace({0: 10, 1: 5})
        stop, reason = should_stop_spreading(trace, current_hop=2, min_new_neurons=2, grace_hops=1)
        assert stop is False

    def test_stop_relative_gain_ratio(self) -> None:
        """Stop when gain ratio drops below threshold."""
        trace = self._make_trace({0: 20, 1: 10, 2: 1})
        # gain_ratio = 1/10 = 0.1 < 0.15
        stop, reason = should_stop_spreading(
            trace, current_hop=3, threshold=0.15, min_new_neurons=1, grace_hops=1
        )
        assert stop is True
        assert "gain ratio" in reason

    def test_no_stop_good_gain_ratio(self) -> None:
        """Don't stop when gain ratio is healthy."""
        trace = self._make_trace({0: 10, 1: 8, 2: 6})
        # gain_ratio = 6/8 = 0.75 > 0.15
        stop, reason = should_stop_spreading(
            trace, current_hop=3, threshold=0.15, min_new_neurons=2, grace_hops=1
        )
        assert stop is False

    def test_no_stop_hop_1_zero_division_safe(self) -> None:
        """When hop before previous had 0 neurons, skip ratio check."""
        trace = self._make_trace({0: 0, 1: 5})
        stop, reason = should_stop_spreading(
            trace, current_hop=2, threshold=0.15, min_new_neurons=2, grace_hops=0
        )
        assert stop is False  # 5 >= min_new=2, ratio check skipped (0 in denominator)

    def test_grace_hops_zero(self) -> None:
        """With grace_hops=0, gating starts at hop 1."""
        trace = self._make_trace({0: 10})
        stop, reason = should_stop_spreading(
            trace, current_hop=1, min_new_neurons=2, grace_hops=0
        )
        # hop 0 had 10 neurons, checking at hop 1 should not stop
        assert stop is False  # 10 >= 2

    def test_grace_hops_2(self) -> None:
        """With grace_hops=2, never stop at hop 1 or 2."""
        trace = self._make_trace({0: 10, 1: 0})
        stop, _ = should_stop_spreading(trace, current_hop=2, grace_hops=2)
        assert stop is False

    def test_exact_threshold_boundary(self) -> None:
        """At exactly the threshold, should NOT stop (< not <=)."""
        trace = self._make_trace({0: 100, 1: 20, 2: 3})
        # gain_ratio = 3/20 = 0.15 == threshold → not less than, so no stop
        stop, _ = should_stop_spreading(
            trace, current_hop=3, threshold=0.15, min_new_neurons=2, grace_hops=1
        )
        assert stop is False


# ---------------------------------------------------------------------------
# Integration tests: BFS engine with diminishing returns
# ---------------------------------------------------------------------------


def _make_storage_mock(
    neurons: dict[str, Any],
    neighbors: dict[str, list[tuple[str, float]]],
) -> AsyncMock:
    """Create a mock storage with configurable graph topology."""
    storage = AsyncMock()

    # get_neurons_batch
    async def mock_get_neurons_batch(ids: list[str]) -> dict[str, Any]:
        return {nid: SimpleNamespace(id=nid) for nid in ids if nid in neurons}

    storage.get_neurons_batch = mock_get_neurons_batch

    # get_neighbors: returns list of (neuron, synapse) tuples
    async def mock_get_neighbors(neuron_id: str, direction: str = "both", min_weight: float = 0.1) -> list:
        result = []
        for target_id, weight in neighbors.get(neuron_id, []):
            neuron_obj = SimpleNamespace(id=target_id)
            synapse_obj = SimpleNamespace(weight=weight, target_id=target_id)
            result.append((neuron_obj, synapse_obj))
        return result

    storage.get_neighbors = mock_get_neighbors

    # get_neuron_states_batch: return empty states (no frequency, no refractory)
    async def mock_get_neuron_states_batch(ids: list[str]) -> dict:
        return {nid: SimpleNamespace(access_frequency=0, in_refractory=False) for nid in ids}

    storage.get_neuron_states_batch = mock_get_neuron_states_batch

    return storage


@pytest.mark.asyncio
class TestBFSDiminishingReturns:
    """Test diminishing returns gate in BFS spreading activation."""

    async def test_early_stop_sparse_graph(self) -> None:
        """BFS stops early when hop 2 adds too few neurons."""
        # Graph: a -> b -> c (linear chain, hop 2 adds only 1 neuron)
        neurons = {"a": True, "b": True, "c": True}
        neighbors = {
            "a": [("b", 0.8)],
            "b": [("c", 0.8)],
            "c": [],
        }
        storage = _make_storage_mock(neurons, neighbors)
        config = BrainConfig(
            diminishing_returns_enabled=True,
            diminishing_returns_min_neurons=2,  # need 2+ per hop
            diminishing_returns_grace_hops=0,  # no grace
            max_spread_hops=4,
        )
        activator = SpreadingActivation(storage, config)
        results, trace = await activator.activate(["a"], max_hops=4)

        # hop 0: a (1 neuron), hop 1: b (1 neuron) — below min=2
        # Gate should trigger at hop 2 check
        assert trace.stopped_early is True
        assert "added only 1" in trace.stop_reason
        # c should NOT be in results (stopped before hop 2)
        assert "c" not in results

    async def test_no_early_stop_dense_graph(self) -> None:
        """BFS doesn't stop when each hop adds enough neurons."""
        # Graph: a -> {b1,b2,b3} -> {c1,c2,c3}
        neurons = dict.fromkeys(["a", "b1", "b2", "b3", "c1", "c2", "c3"], True)
        neighbors = {
            "a": [("b1", 0.8), ("b2", 0.8), ("b3", 0.8)],
            "b1": [("c1", 0.8)],
            "b2": [("c2", 0.8)],
            "b3": [("c3", 0.8)],
            "c1": [], "c2": [], "c3": [],
        }
        storage = _make_storage_mock(neurons, neighbors)
        config = BrainConfig(
            diminishing_returns_enabled=True,
            diminishing_returns_min_neurons=2,
            diminishing_returns_grace_hops=1,
            activation_threshold=0.05,  # Low threshold so hop 2 activations pass
            max_spread_hops=4,
        )
        activator = SpreadingActivation(storage, config)
        results, trace = await activator.activate(["a"], max_hops=4)

        assert trace.stopped_early is False
        # All nodes should be reached
        assert "c1" in results
        assert "c2" in results
        assert "c3" in results

    async def test_disabled_gate(self) -> None:
        """When disabled, never stops early."""
        neurons = {"a": True, "b": True, "c": True}
        neighbors = {
            "a": [("b", 0.8)],
            "b": [("c", 0.8)],
            "c": [],
        }
        storage = _make_storage_mock(neurons, neighbors)
        config = BrainConfig(
            diminishing_returns_enabled=False,
            activation_threshold=0.05,
            max_spread_hops=4,
        )
        activator = SpreadingActivation(storage, config)
        results, trace = await activator.activate(["a"], max_hops=4)

        assert trace.stopped_early is False
        assert "c" in results  # All nodes reached

    async def test_trace_records_metrics(self) -> None:
        """Trace correctly records per-hop neuron counts."""
        neurons = {"a": True, "b1": True, "b2": True, "c1": True}
        neighbors = {
            "a": [("b1", 0.8), ("b2", 0.8)],
            "b1": [("c1", 0.8)],
            "b2": [],
            "c1": [],
        }
        storage = _make_storage_mock(neurons, neighbors)
        config = BrainConfig(
            diminishing_returns_enabled=False,  # Disable to see all hops
            activation_threshold=0.05,
            max_spread_hops=4,
        )
        activator = SpreadingActivation(storage, config)
        results, trace = await activator.activate(["a"], max_hops=4)

        assert trace.new_neurons_per_hop[0] == 1  # anchor: a
        assert trace.new_neurons_per_hop[1] == 2  # b1, b2
        assert trace.new_neurons_per_hop[2] == 1  # c1
        assert trace.total_neurons_activated == 4
        assert trace.max_hop_used == 2

    async def test_grace_hops_respected(self) -> None:
        """Grace hops allow sparse initial exploration."""
        neurons = {"a": True, "b": True, "c": True, "d": True}
        neighbors = {
            "a": [("b", 0.8)],
            "b": [("c", 0.8)],
            "c": [("d", 0.8)],
            "d": [],
        }
        storage = _make_storage_mock(neurons, neighbors)
        config = BrainConfig(
            diminishing_returns_enabled=True,
            diminishing_returns_min_neurons=2,
            diminishing_returns_grace_hops=2,  # Allow hops 0-2 freely
            activation_threshold=0.05,
            max_spread_hops=4,
        )
        activator = SpreadingActivation(storage, config)
        results, trace = await activator.activate(["a"], max_hops=4)

        # With grace_hops=2, gate starts checking at hop 3
        # hop 2 added 1 neuron (c) < min=2 → stop at hop 3
        assert trace.stopped_early is True
        assert "b" in results  # hop 1
        assert "c" in results  # hop 2 (grace)
        assert "d" not in results  # hop 3 blocked

    async def test_empty_anchors(self) -> None:
        """Empty anchor list returns empty results and clean trace."""
        storage = _make_storage_mock({}, {})
        config = BrainConfig()
        activator = SpreadingActivation(storage, config)
        results, trace = await activator.activate([])

        assert results == {}
        assert trace.total_neurons_activated == 0
        assert trace.stopped_early is False

    async def test_single_hop_query(self) -> None:
        """With max_hops=1, gate doesn't interfere."""
        neurons = {"a": True, "b": True}
        neighbors = {"a": [("b", 0.8)], "b": []}
        storage = _make_storage_mock(neurons, neighbors)
        config = BrainConfig(
            diminishing_returns_enabled=True,
            max_spread_hops=1,
        )
        activator = SpreadingActivation(storage, config)
        results, trace = await activator.activate(["a"], max_hops=1)

        assert "b" in results
        assert trace.stopped_early is False


# ---------------------------------------------------------------------------
# PPR diminishing returns tests
# ---------------------------------------------------------------------------


def _make_ppr_storage_mock(
    neighbors: dict[str, list[tuple[str, float]]],
) -> AsyncMock:
    """Create storage mock for PPR tests."""
    storage = AsyncMock()

    async def mock_get_synapses_for_neurons(
        neuron_ids: list[str], direction: str = "out"
    ) -> dict[str, list]:
        result: dict[str, list] = {}
        for nid in neuron_ids:
            synapses = []
            for target_id, weight in neighbors.get(nid, []):
                synapse = SimpleNamespace(target_id=target_id, weight=weight)
                synapses.append(synapse)
            result[nid] = synapses
        return result

    storage.get_synapses_for_neurons = mock_get_synapses_for_neurons
    return storage


@pytest.mark.asyncio
class TestPPRDiminishingReturns:
    """Test diminishing returns in PPR activation."""

    async def test_ppr_returns_trace(self) -> None:
        """PPR activate() returns (results, trace) tuple."""
        from neural_memory.engine.ppr_activation import PPRActivation

        neighbors: dict[str, list[tuple[str, float]]] = {
            "a": [("b", 0.5)],
            "b": [("a", 0.5)],
        }
        storage = _make_ppr_storage_mock(neighbors)
        config = BrainConfig(
            diminishing_returns_enabled=False,
            ppr_iterations=5,
        )
        ppr = PPRActivation(storage, config)
        results, trace = await ppr.activate(["a"])

        assert isinstance(trace, ActivationTrace)
        assert trace.max_hop_allowed == 5

    async def test_ppr_early_stop_on_stale_iterations(self) -> None:
        """PPR stops when iterations add no new above-threshold nodes."""
        from neural_memory.engine.ppr_activation import PPRActivation

        # Simple graph: a -> b, no further expansion
        neighbors: dict[str, list[tuple[str, float]]] = {
            "a": [("b", 1.0)],
            "b": [],
        }
        storage = _make_ppr_storage_mock(neighbors)
        config = BrainConfig(
            diminishing_returns_enabled=True,
            diminishing_returns_min_neurons=1,
            ppr_iterations=20,
            ppr_damping=0.15,
            activation_threshold=0.1,
        )
        ppr = PPRActivation(storage, config)
        results, trace = await ppr.activate(["a"])

        # PPR should converge or hit diminishing returns well before 20 iterations
        assert trace.max_hop_used < 19  # Should stop early

    async def test_ppr_disabled_gate(self) -> None:
        """PPR doesn't stop early when gate is disabled."""
        from neural_memory.engine.ppr_activation import PPRActivation

        neighbors: dict[str, list[tuple[str, float]]] = {
            "a": [("b", 1.0)],
            "b": [],
        }
        storage = _make_ppr_storage_mock(neighbors)
        config = BrainConfig(
            diminishing_returns_enabled=False,
            ppr_iterations=5,
            activation_threshold=0.1,
        )
        ppr = PPRActivation(storage, config)
        results, trace = await ppr.activate(["a"])

        assert trace.stopped_early is False


# ---------------------------------------------------------------------------
# Reflex trace tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReflexTrace:
    """Test trace tracking in reflex activation."""

    async def test_reflex_returns_trace(self) -> None:
        """Reflex activate_trail() returns (results, trace) tuple."""
        from neural_memory.engine.reflex_activation import ReflexActivation

        storage = AsyncMock()
        config = BrainConfig()
        reflex = ReflexActivation(storage, config)

        # Create a simple fiber mock
        fiber = MagicMock()
        fiber.is_in_pathway.return_value = True
        fiber.pathway_position.return_value = 1
        fiber.pathway = ["x", "a", "y"]
        fiber.conductivity = 0.9
        fiber.last_conducted = None
        fiber.salience = 0.5

        results, trace = await reflex.activate_trail(
            anchor_neurons=["a"],
            fibers=[fiber],
        )

        assert isinstance(trace, ActivationTrace)
        assert trace.new_neurons_per_hop[0] >= 1  # at least the anchor

    async def test_reflex_no_fibers(self) -> None:
        """Reflex with no matching fibers returns clean trace."""
        from neural_memory.engine.reflex_activation import ReflexActivation

        storage = AsyncMock()
        config = BrainConfig()
        reflex = ReflexActivation(storage, config)

        fiber = MagicMock()
        fiber.is_in_pathway.return_value = False

        results, trace = await reflex.activate_trail(
            anchor_neurons=["a"],
            fibers=[fiber],
        )

        assert "a" in results  # anchor always in results
        assert trace.new_neurons_per_hop[0] == 1
        assert trace.stopped_early is False
