"""Tests for graph-based query expansion."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.query_expansion import expand_via_graph


def _make_neuron(nid: str, ntype: NeuronType = NeuronType.ENTITY) -> Neuron:
    return Neuron.create(content=f"content-{nid}", type=ntype, neuron_id=nid)


def _make_synapse(source_id: str, target_id: str, weight: float = 0.5) -> Synapse:
    return Synapse.create(
        source_id=source_id,
        target_id=target_id,
        type=SynapseType.RELATED_TO,
        weight=weight,
    )


@pytest.fixture()
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    return storage


class TestExpandViaGraph:
    """Test 1-hop graph expansion."""

    async def test_empty_seeds(self, mock_storage: AsyncMock) -> None:
        ids, ranked = await expand_via_graph(mock_storage, [])
        assert ids == []
        assert ranked == []

    async def test_non_expandable_type_skipped(self, mock_storage: AsyncMock) -> None:
        """TIME neurons should not be expanded from."""
        mock_storage.get_neurons_batch.return_value = {
            "t1": _make_neuron("t1", NeuronType.TIME),
        }
        ids, ranked = await expand_via_graph(mock_storage, ["t1"])
        assert ids == []
        # Should not even call get_synapses_for_neurons
        mock_storage.get_synapses_for_neurons.assert_not_called()

    async def test_basic_expansion(self, mock_storage: AsyncMock) -> None:
        """Entity neuron expands to its 1-hop neighbors."""
        mock_storage.get_neurons_batch.side_effect = [
            # First call: seed neurons
            {"e1": _make_neuron("e1", NeuronType.ENTITY)},
            # Second call: candidate neurons (filter TIME)
            {
                "n1": _make_neuron("n1", NeuronType.CONCEPT),
                "n2": _make_neuron("n2", NeuronType.ENTITY),
            },
        ]
        mock_storage.get_synapses_for_neurons.return_value = {
            "e1": [
                _make_synapse("e1", "n1", weight=0.8),
                _make_synapse("e1", "n2", weight=0.6),
            ],
        }

        ids, ranked = await expand_via_graph(mock_storage, ["e1"])

        assert ids == ["n1", "n2"]  # sorted by weight desc
        assert len(ranked) == 2
        assert ranked[0].neuron_id == "n1"
        assert ranked[0].rank == 1
        assert ranked[0].retriever == "graph_expansion"
        assert ranked[1].neuron_id == "n2"
        assert ranked[1].rank == 2

    async def test_filters_below_min_weight(self, mock_storage: AsyncMock) -> None:
        """Weak synapses below min_weight are excluded."""
        mock_storage.get_neurons_batch.side_effect = [
            {"e1": _make_neuron("e1", NeuronType.ENTITY)},
            {"n1": _make_neuron("n1")},
        ]
        mock_storage.get_synapses_for_neurons.return_value = {
            "e1": [
                _make_synapse("e1", "n1", weight=0.8),
                _make_synapse("e1", "n2", weight=0.1),  # below min
            ],
        }

        ids, _ = await expand_via_graph(mock_storage, ["e1"], min_synapse_weight=0.3)
        assert "n1" in ids
        assert "n2" not in ids

    async def test_excludes_seed_neurons(self, mock_storage: AsyncMock) -> None:
        """Seed neurons should not appear in expansion results."""
        mock_storage.get_neurons_batch.side_effect = [
            {"e1": _make_neuron("e1", NeuronType.ENTITY)},
            {},  # no extra neurons to fetch (self-loop filtered)
        ]
        mock_storage.get_synapses_for_neurons.return_value = {
            "e1": [_make_synapse("e1", "e1", weight=0.9)],  # self-loop
        }

        ids, _ = await expand_via_graph(mock_storage, ["e1"])
        assert ids == []

    async def test_caps_at_max_expansions(self, mock_storage: AsyncMock) -> None:
        """Expansion is capped at max_expansions."""
        # Create 20 neighbor synapses
        synapses = [_make_synapse("e1", f"n{i}", weight=0.5 + i * 0.01) for i in range(20)]
        neuron_batch = {f"n{i}": _make_neuron(f"n{i}") for i in range(20)}
        mock_storage.get_neurons_batch.side_effect = [
            {"e1": _make_neuron("e1", NeuronType.ENTITY)},
            neuron_batch,
        ]
        mock_storage.get_synapses_for_neurons.return_value = {"e1": synapses}

        ids, ranked = await expand_via_graph(mock_storage, ["e1"], max_expansions=5)
        assert len(ids) == 5
        assert len(ranked) == 5

    async def test_filters_time_neurons_from_targets(self, mock_storage: AsyncMock) -> None:
        """TIME neurons should be excluded from expansion targets."""
        mock_storage.get_neurons_batch.side_effect = [
            {"e1": _make_neuron("e1", NeuronType.ENTITY)},
            {
                "n1": _make_neuron("n1", NeuronType.CONCEPT),
                "t1": _make_neuron("t1", NeuronType.TIME),
            },
        ]
        mock_storage.get_synapses_for_neurons.return_value = {
            "e1": [
                _make_synapse("e1", "n1", weight=0.7),
                _make_synapse("e1", "t1", weight=0.9),
            ],
        }

        ids, _ = await expand_via_graph(mock_storage, ["e1"])
        assert "n1" in ids
        assert "t1" not in ids

    async def test_concept_neurons_expandable(self, mock_storage: AsyncMock) -> None:
        """CONCEPT neurons should also be expanded from."""
        mock_storage.get_neurons_batch.side_effect = [
            {"c1": _make_neuron("c1", NeuronType.CONCEPT)},
            {"n1": _make_neuron("n1")},
        ]
        mock_storage.get_synapses_for_neurons.return_value = {
            "c1": [_make_synapse("c1", "n1", weight=0.6)],
        }

        ids, _ = await expand_via_graph(mock_storage, ["c1"])
        assert ids == ["n1"]

    async def test_no_synapses_returns_empty(self, mock_storage: AsyncMock) -> None:
        mock_storage.get_neurons_batch.side_effect = [
            {"e1": _make_neuron("e1", NeuronType.ENTITY)},
        ]
        mock_storage.get_synapses_for_neurons.return_value = {"e1": []}

        ids, _ = await expand_via_graph(mock_storage, ["e1"])
        assert ids == []
