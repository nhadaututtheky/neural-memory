"""Tests for topology_analysis: graph structure metrics."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.topology_analysis import (
    TopologyMetrics,
    _clustering_coefficient,
    _largest_component_ratio,
    compute_topology,
)


def _mock_synapse(source_id: str, target_id: str, enriched: bool = False) -> MagicMock:
    """Create a mock synapse with source/target IDs."""
    s = MagicMock()
    s.source_id = source_id
    s.target_id = target_id
    s.metadata = {"_enriched": True} if enriched else {}
    return s


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Storage with empty brain."""
    storage = AsyncMock()
    storage.get_stats = AsyncMock(
        return_value={"neuron_count": 0, "synapse_count": 0, "fiber_count": 0}
    )
    storage.get_all_synapses = AsyncMock(return_value=[])
    storage.get_neighbors = AsyncMock(return_value=[])
    return storage


class TestTopologyMetrics:
    """Tests for TopologyMetrics dataclass."""

    def test_frozen(self) -> None:
        """TopologyMetrics is immutable."""
        tm = TopologyMetrics(
            clustering_coefficient=0.5,
            largest_component_ratio=0.8,
            density=0.1,
            knowledge_density=3.0,
            enriched_synapse_ratio=0.2,
        )
        with pytest.raises(AttributeError):
            tm.density = 0.9  # type: ignore[misc]


class TestEmptyBrain:
    """Tests for empty brain topology."""

    @pytest.mark.asyncio
    async def test_empty_brain(self, mock_storage: AsyncMock) -> None:
        """Empty brain returns all-zero metrics."""
        result = await compute_topology(mock_storage, "brain-1")
        assert result.clustering_coefficient == 0.0
        assert result.largest_component_ratio == 0.0
        assert result.density == 0.0
        assert result.knowledge_density == 0.0
        assert result.enriched_synapse_ratio == 0.0


class TestLargestComponentRatio:
    """Tests for connected component analysis."""

    def test_single_cluster(self) -> None:
        """All neurons connected → LCC ratio 1.0."""
        synapses = [
            _mock_synapse("a", "b"),
            _mock_synapse("b", "c"),
            _mock_synapse("c", "a"),
        ]
        ratio = _largest_component_ratio(synapses, neuron_count=3)
        assert ratio == 1.0

    def test_disconnected_clusters(self) -> None:
        """Two separate clusters → LCC ratio < 1.0."""
        synapses = [
            _mock_synapse("a", "b"),
            _mock_synapse("c", "d"),
        ]
        ratio = _largest_component_ratio(synapses, neuron_count=4)
        assert ratio == 0.5

    def test_isolated_neurons(self) -> None:
        """Neurons not in any synapse reduce LCC ratio."""
        synapses = [_mock_synapse("a", "b")]
        # neuron_count=5 means 3 neurons are isolated
        ratio = _largest_component_ratio(synapses, neuron_count=5)
        assert ratio == 2 / 5

    def test_empty_synapses(self) -> None:
        """No synapses → ratio 0.0."""
        ratio = _largest_component_ratio([], neuron_count=5)
        assert ratio == 0.0

    def test_zero_neurons(self) -> None:
        """Zero neuron_count → ratio 0.0."""
        ratio = _largest_component_ratio([], neuron_count=0)
        assert ratio == 0.0


class TestClusteringCoefficient:
    """Tests for clustering coefficient."""

    def test_triangle(self) -> None:
        """Perfect triangle → coefficient 1.0."""
        synapses = [
            _mock_synapse("a", "b"),
            _mock_synapse("b", "c"),
            _mock_synapse("a", "c"),
        ]
        coeff = _clustering_coefficient(synapses)
        assert coeff == 1.0

    def test_star(self) -> None:
        """Star topology (hub + leaves) → coefficient 0.0."""
        synapses = [
            _mock_synapse("hub", "a"),
            _mock_synapse("hub", "b"),
            _mock_synapse("hub", "c"),
        ]
        coeff = _clustering_coefficient(synapses)
        assert coeff == 0.0

    def test_empty(self) -> None:
        """No synapses → coefficient 0.0."""
        coeff = _clustering_coefficient([])
        assert coeff == 0.0

    def test_single_edge(self) -> None:
        """Single edge, no triangles possible → 0.0."""
        synapses = [_mock_synapse("a", "b")]
        coeff = _clustering_coefficient(synapses)
        assert coeff == 0.0

    def test_large_graph_sampling(self) -> None:
        """Graph with >200 nodes triggers sampling path."""
        # Create 250 nodes in a chain (no triangles)
        synapses = [
            _mock_synapse(f"n{i}", f"n{i+1}") for i in range(250)
        ]
        coeff = _clustering_coefficient(synapses)
        # Chain has no triangles → coefficient 0.0
        assert coeff == 0.0

    def test_large_graph_deterministic(self) -> None:
        """Sampling is deterministic (same result on repeated calls)."""
        synapses = [
            _mock_synapse(f"n{i}", f"n{i+1}") for i in range(250)
        ]
        coeff1 = _clustering_coefficient(synapses)
        coeff2 = _clustering_coefficient(synapses)
        assert coeff1 == coeff2

    def test_hub_neighbor_cap(self) -> None:
        """Hub node with many neighbors does not cause excessive computation."""
        # Hub connected to 300 leaves + some triangles among first 3
        synapses = [_mock_synapse("hub", f"leaf{i}") for i in range(300)]
        # Add triangles among first 3 leaves
        synapses.append(_mock_synapse("leaf0", "leaf1"))
        synapses.append(_mock_synapse("leaf1", "leaf2"))
        synapses.append(_mock_synapse("leaf0", "leaf2"))

        coeff = _clustering_coefficient(synapses)
        # Should complete without hanging and return a bounded value
        assert 0.0 <= coeff <= 1.0


class TestEnrichedRatio:
    """Tests for enriched synapse ratio."""

    @pytest.mark.asyncio
    async def test_enriched_synapses_counted(self, mock_storage: AsyncMock) -> None:
        """Enriched synapses are counted correctly."""
        synapses = [
            _mock_synapse("a", "b", enriched=True),
            _mock_synapse("b", "c", enriched=False),
            _mock_synapse("c", "a", enriched=True),
        ]
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 3, "synapse_count": 3, "fiber_count": 0}
        )
        mock_storage.get_all_synapses = AsyncMock(return_value=synapses)

        result = await compute_topology(mock_storage, "brain-1")
        assert abs(result.enriched_synapse_ratio - 2 / 3) < 0.01


class TestKnowledgeDensity:
    """Tests for knowledge density."""

    @pytest.mark.asyncio
    async def test_knowledge_density(self, mock_storage: AsyncMock) -> None:
        """Knowledge density = synapses / neurons."""
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 10, "synapse_count": 30, "fiber_count": 5}
        )
        mock_storage.get_all_synapses = AsyncMock(
            return_value=[_mock_synapse("a", "b") for _ in range(30)]
        )

        result = await compute_topology(mock_storage, "brain-1")
        assert result.knowledge_density == 3.0


class TestUndirectedDensity:
    """Tests for undirected density formula."""

    @pytest.mark.asyncio
    async def test_density_undirected(self, mock_storage: AsyncMock) -> None:
        """Density uses undirected formula: edges / (n*(n-1)/2)."""
        # 3 neurons, 3 synapses → max_edges = 3*2/2 = 3 → density = 1.0
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 3, "synapse_count": 3, "fiber_count": 0}
        )
        mock_storage.get_all_synapses = AsyncMock(
            return_value=[
                _mock_synapse("a", "b"),
                _mock_synapse("b", "c"),
                _mock_synapse("a", "c"),
            ]
        )

        result = await compute_topology(mock_storage, "brain-1")
        assert result.density == 1.0

    @pytest.mark.asyncio
    async def test_density_capped_at_one(self, mock_storage: AsyncMock) -> None:
        """Density is capped at 1.0 even if synapse_count exceeds max_edges."""
        # 2 neurons, max_edges = 1, but 5 synapses reported
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 2, "synapse_count": 5, "fiber_count": 0}
        )
        mock_storage.get_all_synapses = AsyncMock(
            return_value=[_mock_synapse("a", "b") for _ in range(5)]
        )

        result = await compute_topology(mock_storage, "brain-1")
        assert result.density == 1.0


class TestPreloadedSynapses:
    """Tests for _preloaded_synapses parameter."""

    @pytest.mark.asyncio
    async def test_uses_preloaded(self, mock_storage: AsyncMock) -> None:
        """Pre-loaded synapses avoid calling get_all_synapses."""
        synapses = [
            _mock_synapse("a", "b"),
            _mock_synapse("b", "c"),
        ]
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 3, "synapse_count": 2, "fiber_count": 0}
        )

        result = await compute_topology(
            mock_storage, "brain-1", _preloaded_synapses=synapses
        )

        # Should NOT have called get_all_synapses
        mock_storage.get_all_synapses.assert_not_called()
        assert result.largest_component_ratio > 0.0
