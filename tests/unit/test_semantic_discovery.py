"""Tests for semantic synapse discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from neural_memory.engine.semantic_discovery import (
    SemanticDiscoveryResult,
    _cosine_similarity,
    discover_semantic_synapses,
)
from neural_memory.storage.memory_store import InMemoryStorage


@pytest.fixture
def brain_config() -> BrainConfig:
    return BrainConfig(
        embedding_enabled=True,
        semantic_discovery_similarity_threshold=0.7,
        semantic_discovery_max_pairs=100,
    )


@pytest.fixture
def brain_config_disabled() -> BrainConfig:
    return BrainConfig(embedding_enabled=False)


@pytest.fixture
def brain(brain_config: BrainConfig) -> Brain:
    return Brain.create(name="test", config=brain_config)


@pytest.fixture
async def storage(brain: Brain) -> InMemoryStorage:
    store = InMemoryStorage()
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


class TestCosimeSimilarity:
    """Tests for cosine similarity helper."""

    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_similar_vectors(self) -> None:
        sim = _cosine_similarity([1.0, 0.5], [1.0, 0.6])
        assert sim > 0.9


class TestSemanticDiscoveryResult:
    """Tests for result dataclass."""

    def test_defaults(self) -> None:
        r = SemanticDiscoveryResult()
        assert r.neurons_embedded == 0
        assert r.pairs_evaluated == 0
        assert r.synapses_created == 0
        assert r.skipped_existing == 0
        assert r.synapses == []

    def test_frozen(self) -> None:
        r = SemanticDiscoveryResult(neurons_embedded=5)
        with pytest.raises(AttributeError):
            r.neurons_embedded = 10  # type: ignore[misc]


class TestDiscoverSemanticSynapses:
    """Tests for the main discovery function."""

    async def test_skips_when_embedding_disabled(
        self, storage: InMemoryStorage, brain_config_disabled: BrainConfig
    ) -> None:
        result = await discover_semantic_synapses(storage, brain_config_disabled)
        assert result.neurons_embedded == 0
        assert result.synapses_created == 0

    async def test_skips_when_provider_unavailable(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        with patch(
            "neural_memory.engine.semantic_discovery._create_provider",
            side_effect=ImportError("no provider"),
        ):
            result = await discover_semantic_synapses(storage, brain_config)
            assert result.neurons_embedded == 0

    async def test_skips_fewer_than_two_neurons(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="only one", neuron_id="n1")
        await storage.add_neuron(n1)

        mock_provider = AsyncMock()
        with patch(
            "neural_memory.engine.semantic_discovery._create_provider",
            return_value=mock_provider,
        ):
            result = await discover_semantic_synapses(storage, brain_config)
            assert result.neurons_embedded == 0

    async def test_discovers_similar_neurons(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """Two similar neurons should produce a SIMILAR_TO synapse."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="machine learning", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="deep learning", neuron_id="n2")
        n3 = Neuron.create(type=NeuronType.ENTITY, content="pizza recipe", neuron_id="n3")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        # Mock embeddings: n1 and n2 are very similar, n3 is different
        mock_provider = AsyncMock()
        mock_provider.embed_batch.return_value = [
            [0.9, 0.1, 0.0],  # machine learning
            [0.85, 0.15, 0.0],  # deep learning (similar)
            [0.0, 0.1, 0.9],  # pizza recipe (different)
        ]

        with patch(
            "neural_memory.engine.semantic_discovery._create_provider",
            return_value=mock_provider,
        ):
            result = await discover_semantic_synapses(storage, brain_config)

        assert result.neurons_embedded == 3
        assert result.synapses_created >= 1

        # Check the created synapse
        found_similar = False
        for syn in result.synapses:
            if syn.type == SynapseType.SIMILAR_TO:
                found_similar = True
                assert syn.metadata.get("_semantic_discovery") is True
                assert "cosine_similarity" in syn.metadata
                # Weight should be similarity * 0.6
                assert syn.weight > 0.0
                assert syn.weight <= 0.6
        assert found_similar

    async def test_skips_existing_synapse_pairs(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """Should not create duplicate synapses."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="alpha", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="beta", neuron_id="n2")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Pre-existing synapse
        existing = Synapse.create(
            source_id="n1", target_id="n2", type=SynapseType.RELATED_TO, weight=0.5
        )
        await storage.add_synapse(existing)

        mock_provider = AsyncMock()
        mock_provider.embed_batch.return_value = [
            [1.0, 0.0],
            [0.99, 0.01],  # Very similar
        ]

        with patch(
            "neural_memory.engine.semantic_discovery._create_provider",
            return_value=mock_provider,
        ):
            result = await discover_semantic_synapses(storage, brain_config)

        assert result.skipped_existing >= 1
        assert result.synapses_created == 0

    async def test_respects_max_pairs(self, storage: InMemoryStorage) -> None:
        """Should cap results at max_pairs."""
        config = BrainConfig(
            embedding_enabled=True,
            semantic_discovery_similarity_threshold=0.0,  # Accept all
            semantic_discovery_max_pairs=2,
        )

        for i in range(5):
            n = Neuron.create(type=NeuronType.CONCEPT, content=f"concept {i}", neuron_id=f"n{i}")
            await storage.add_neuron(n)

        mock_provider = AsyncMock()
        # All similar to each other
        mock_provider.embed_batch.return_value = [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
            [0.85, 0.15],
            [0.8, 0.2],
        ]

        with patch(
            "neural_memory.engine.semantic_discovery._create_provider",
            return_value=mock_provider,
        ):
            result = await discover_semantic_synapses(storage, config)

        # Should be capped at 2
        assert result.synapses_created <= 2

    async def test_ignores_non_concept_entity_neurons(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """Only CONCEPT and ENTITY neurons should be considered."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="valid concept", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.TIME, content="2026-01-01", neuron_id="n2")
        n3 = Neuron.create(type=NeuronType.STATE, content="happy", neuron_id="n3")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        mock_provider = AsyncMock()
        with patch(
            "neural_memory.engine.semantic_discovery._create_provider",
            return_value=mock_provider,
        ):
            result = await discover_semantic_synapses(storage, brain_config)

        # Only 1 eligible neuron (n1), so < 2, should skip
        assert result.neurons_embedded == 0

    async def test_handles_embed_batch_failure(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """Should gracefully handle embedding failures."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="alpha", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="beta", neuron_id="n2")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        mock_provider = AsyncMock()
        mock_provider.embed_batch.side_effect = RuntimeError("embedding failed")

        with patch(
            "neural_memory.engine.semantic_discovery._create_provider",
            return_value=mock_provider,
        ):
            result = await discover_semantic_synapses(storage, brain_config)

        assert result.neurons_embedded == 0


class TestConsolidationIntegration:
    """Tests for semantic_link integration in ConsolidationEngine."""

    async def test_semantic_link_strategy_exists(self) -> None:
        """SEMANTIC_LINK should be a valid strategy."""
        assert ConsolidationStrategy.SEMANTIC_LINK == "semantic_link"

    async def test_semantic_link_in_tier(self) -> None:
        """SEMANTIC_LINK should be in a tier."""
        found = False
        for tier in ConsolidationEngine.STRATEGY_TIERS:
            if ConsolidationStrategy.SEMANTIC_LINK in tier:
                found = True
        assert found

    async def test_semantic_link_runs_in_consolidation(self, storage: InMemoryStorage) -> None:
        """Running SEMANTIC_LINK strategy should call discover_semantic_synapses."""
        brain_config = BrainConfig(embedding_enabled=False)
        brain = Brain.create(name="test", config=brain_config)
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        engine = ConsolidationEngine(storage)
        report = await engine.run(strategies=[ConsolidationStrategy.SEMANTIC_LINK])
        # Should complete without error even if embedding is disabled
        assert report.semantic_synapses_created == 0

    async def test_report_has_semantic_synapses_field(self, storage: InMemoryStorage) -> None:
        """ConsolidationReport should track semantic_synapses_created."""
        brain = Brain.create(name="test", config=BrainConfig())
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        engine = ConsolidationEngine(storage)
        report = await engine.run(strategies=[ConsolidationStrategy.SEMANTIC_LINK])
        assert hasattr(report, "semantic_synapses_created")
        assert report.semantic_synapses_created == 0

    async def test_semantic_discovery_metadata_flag(self) -> None:
        """Semantic synapses should have _semantic_discovery metadata."""
        synapse = Synapse.create(
            source_id="a",
            target_id="b",
            type=SynapseType.SIMILAR_TO,
            weight=0.4,
            metadata={"_semantic_discovery": True},
        )
        assert synapse.metadata.get("_semantic_discovery") is True
