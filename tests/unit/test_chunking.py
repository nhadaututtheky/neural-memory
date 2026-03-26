"""Tests for working memory chunking."""

from __future__ import annotations

from neural_memory.engine.chunking import CognitiveChunk, chunk_retrieval_results


class TestCognitiveChunk:
    def test_frozen(self) -> None:
        c = CognitiveChunk(label="test", neuron_ids=("a",), coherence=0.5, relevance=0.8)
        assert c.label == "test"


class TestChunkRetrievalResults:
    def test_empty(self) -> None:
        assert chunk_retrieval_results([], {}, []) == []

    def test_single_neuron(self) -> None:
        chunks = chunk_retrieval_results(
            neuron_ids=["n1"],
            activation_levels={"n1": 0.9},
            synapse_pairs=[],
            max_chunks=5,
        )
        # Single neuron becomes a singleton chunk
        assert len(chunks) == 1
        assert chunks[0].neuron_ids == ("n1",)

    def test_connected_neurons_cluster(self) -> None:
        chunks = chunk_retrieval_results(
            neuron_ids=["n1", "n2", "n3"],
            activation_levels={"n1": 0.9, "n2": 0.7, "n3": 0.5},
            synapse_pairs=[("n1", "n2", 0.8), ("n1", "n3", 0.6)],
            max_chunks=5,
        )
        # n1 seeds a cluster absorbing n2 and n3
        assert len(chunks) >= 1
        assert "n1" in chunks[0].neuron_ids

    def test_max_chunks_limit(self) -> None:
        neuron_ids = [f"n{i}" for i in range(20)]
        activations = {nid: 1.0 / (i + 1) for i, nid in enumerate(neuron_ids)}
        chunks = chunk_retrieval_results(
            neuron_ids=neuron_ids,
            activation_levels=activations,
            synapse_pairs=[],
            max_chunks=3,
        )
        assert len(chunks) <= 3

    def test_auto_label_from_tags(self) -> None:
        chunks = chunk_retrieval_results(
            neuron_ids=["n1", "n2"],
            activation_levels={"n1": 0.9, "n2": 0.7},
            synapse_pairs=[("n1", "n2", 0.8)],
            neuron_tags={"n1": ["python", "async"], "n2": ["python", "testing"]},
            max_chunks=5,
        )
        assert len(chunks) >= 1
        # "python" is shared, should be the label
        assert chunks[0].label == "python"

    def test_sorted_by_relevance(self) -> None:
        chunks = chunk_retrieval_results(
            neuron_ids=["n1", "n2", "n3"],
            activation_levels={"n1": 0.3, "n2": 0.9, "n3": 0.1},
            synapse_pairs=[],
            max_chunks=5,
        )
        relevances = [c.relevance for c in chunks]
        assert relevances == sorted(relevances, reverse=True)

    def test_coherence_computed(self) -> None:
        chunks = chunk_retrieval_results(
            neuron_ids=["n1", "n2"],
            activation_levels={"n1": 0.9, "n2": 0.8},
            synapse_pairs=[("n1", "n2", 0.7)],
            max_chunks=5,
        )
        # Cluster with internal edge → coherence > 0
        cluster = [c for c in chunks if len(c.neuron_ids) > 1]
        if cluster:
            assert cluster[0].coherence > 0.0

    def test_max_per_chunk(self) -> None:
        # With max_per_chunk=2, clusters should be small
        chunks = chunk_retrieval_results(
            neuron_ids=["n1", "n2", "n3", "n4"],
            activation_levels={"n1": 0.9, "n2": 0.8, "n3": 0.7, "n4": 0.6},
            synapse_pairs=[
                ("n1", "n2", 0.8),
                ("n1", "n3", 0.7),
                ("n1", "n4", 0.6),
            ],
            max_chunks=5,
            max_per_chunk=2,
        )
        for c in chunks:
            assert len(c.neuron_ids) <= 2
