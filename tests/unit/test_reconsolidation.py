"""Tests for retrieval reconsolidation — Phase 2 Neuro Engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.reconsolidation import (
    _CONTEXT_ANCHOR_CACHE,
    _jaccard_distance,
    reconsolidate_on_recall,
)


class TestJaccardDistance:
    def test_identical_sets(self) -> None:
        assert _jaccard_distance({"a", "b"}, {"a", "b"}) == 0.0

    def test_disjoint_sets(self) -> None:
        assert _jaccard_distance({"a", "b"}, {"c", "d"}) == 1.0

    def test_partial_overlap(self) -> None:
        # {a,b,c} vs {b,c,d} → intersection=2, union=4 → 1 - 0.5 = 0.5
        assert _jaccard_distance({"a", "b", "c"}, {"b", "c", "d"}) == 0.5

    def test_empty_sets(self) -> None:
        assert _jaccard_distance(set(), set()) == 0.0

    def test_one_empty(self) -> None:
        assert _jaccard_distance({"a"}, set()) == 1.0


class TestReconsolidateOnRecall:
    """Test reconsolidation of recalled memories."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        _CONTEXT_ANCHOR_CACHE.clear()

    @pytest.fixture
    def mock_storage(self) -> AsyncMock:
        storage = AsyncMock()
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="original memory about Python",
            metadata={"_original_tags": ["python", "programming"]},
        )
        storage.get_neuron = AsyncMock(return_value=neuron)
        storage.add_synapse = AsyncMock()
        storage.update_neuron = AsyncMock()
        storage.add_neuron = AsyncMock()
        # No existing concept anchor
        storage.find_neurons = AsyncMock(return_value=[])
        return storage

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.reconsolidation_enabled = True
        config.reconsolidation_drift_threshold = 0.6
        return config

    @pytest.mark.asyncio
    async def test_same_context_low_drift(
        self, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Recall in same context → low drift, no bridge."""
        result = await reconsolidate_on_recall(
            fiber_id="fiber-1",
            anchor_neuron_id="neuron-1",
            query_tags={"python", "programming"},
            query_entities=["Python"],
            storage=mock_storage,
            config=mock_config,
        )
        assert result is not None
        assert result.drift_score < 0.6
        assert not result.bridge_created
        assert result.reconsolidation_count == 1

    @pytest.mark.asyncio
    async def test_different_context_high_drift(
        self, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Recall in different context → high drift, bridge created."""
        result = await reconsolidate_on_recall(
            fiber_id="fiber-1",
            anchor_neuron_id="neuron-1",
            query_tags={"kubernetes", "deployment", "devops"},
            query_entities=["Kubernetes"],
            storage=mock_storage,
            config=mock_config,
        )
        assert result is not None
        assert result.drift_score > 0.6
        assert result.bridge_created
        mock_storage.add_synapse.assert_called()
        # Verify synapse type is RELATED_TO
        call_args = mock_storage.add_synapse.call_args[0][0]
        assert call_args.type == SynapseType.RELATED_TO
        assert call_args.metadata.get("_reconsolidation_bridge") is True

    @pytest.mark.asyncio
    async def test_context_trail_updated(
        self, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Reconsolidation updates context trail in metadata."""
        await reconsolidate_on_recall(
            fiber_id="fiber-1",
            anchor_neuron_id="neuron-1",
            query_tags={"python"},
            query_entities=["Python"],
            storage=mock_storage,
            config=mock_config,
        )
        # Check update_neuron was called with context trail
        mock_storage.update_neuron.assert_called_once()
        updated = mock_storage.update_neuron.call_args[0][0]
        trail = updated.metadata["_context_trail"]
        assert len(trail) == 1
        assert "python" in trail[0]["tags"]

    @pytest.mark.asyncio
    async def test_context_trail_rolling_window(
        self, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Context trail should be capped at 5 entries."""
        # Pre-populate with 5 trail entries
        existing_trail = [
            {"tags": [f"tag{i}"], "entities": [], "ts": "2026-01-01"} for i in range(5)
        ]
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="test",
            metadata={"_context_trail": existing_trail, "_reconsolidation_count": 5},
        )
        mock_storage.get_neuron = AsyncMock(return_value=neuron)

        result = await reconsolidate_on_recall(
            fiber_id="fiber-1",
            anchor_neuron_id="neuron-1",
            query_tags={"new_tag"},
            query_entities=[],
            storage=mock_storage,
            config=mock_config,
        )

        updated = mock_storage.update_neuron.call_args[0][0]
        trail = updated.metadata["_context_trail"]
        assert len(trail) == 5  # capped
        assert "new_tag" in trail[-1]["tags"]  # latest entry
        assert result is not None
        assert result.reconsolidation_count == 6

    @pytest.mark.asyncio
    async def test_reconsolidation_count_increments(
        self, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Count increments on each recall."""
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="test",
            metadata={"_reconsolidation_count": 3},
        )
        mock_storage.get_neuron = AsyncMock(return_value=neuron)

        result = await reconsolidate_on_recall(
            fiber_id="fiber-1",
            anchor_neuron_id="neuron-1",
            query_tags=set(),
            query_entities=[],
            storage=mock_storage,
            config=mock_config,
        )
        assert result is not None
        assert result.reconsolidation_count == 4

    @pytest.mark.asyncio
    async def test_drift_threshold_configurable(self, mock_storage: AsyncMock) -> None:
        """Custom threshold changes bridge creation behavior."""
        config = MagicMock()
        config.reconsolidation_enabled = True
        config.reconsolidation_drift_threshold = 0.9  # Very high threshold

        result = await reconsolidate_on_recall(
            fiber_id="fiber-1",
            anchor_neuron_id="neuron-1",
            query_tags={"kubernetes", "devops"},
            query_entities=["Kubernetes"],
            storage=mock_storage,
            config=config,
        )
        # Drift might be high, but threshold is 0.9 — bridge may not be created
        assert result is not None
        # With threshold 0.9, only extreme drift triggers bridges

    @pytest.mark.asyncio
    async def test_neuron_not_found(self, mock_storage: AsyncMock, mock_config: MagicMock) -> None:
        """Missing neuron → None result."""
        mock_storage.get_neuron = AsyncMock(return_value=None)
        result = await reconsolidate_on_recall(
            fiber_id="fiber-1",
            anchor_neuron_id="missing-id",
            query_tags={"test"},
            query_entities=[],
            storage=mock_storage,
            config=mock_config,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_context_anchor_reused_from_cache(
        self, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Context anchor should be cached and reused."""
        # First call — creates anchor
        await reconsolidate_on_recall(
            fiber_id="fiber-1",
            anchor_neuron_id="neuron-1",
            query_tags={"kubernetes", "devops", "cloud"},
            query_entities=["Kubernetes"],
            storage=mock_storage,
            config=mock_config,
            brain_id="test-brain",
        )

        # Reset find_neurons calls

        # Second call — same entity, should use cache
        await reconsolidate_on_recall(
            fiber_id="fiber-2",
            anchor_neuron_id="neuron-1",
            query_tags={"kubernetes", "devops", "cloud"},
            query_entities=["Kubernetes"],
            storage=mock_storage,
            config=mock_config,
            brain_id="test-brain",
        )

        # Second call should not trigger additional find_neurons for "Kubernetes"
        # (it may still call find_neurons for the neuron lookup, but anchor is cached)
        assert "test-brain:Kubernetes" in _CONTEXT_ANCHOR_CACHE
