"""Tests for schema assimilation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.schema_assimilation import (
    AssimilationAction,
    AssimilationResult,
    _extract_shared_entities,
    assimilate_or_accommodate,
    batch_schema_assimilation,
)


def _make_neuron(
    content: str, tags: list[str] | None = None, ntype: NeuronType = NeuronType.CONCEPT
) -> Neuron:
    return Neuron.create(
        content=content,
        type=ntype,
        metadata={"tags": tags or []},
    )


class TestAssimilationResult:
    def test_frozen(self) -> None:
        r = AssimilationResult(action=AssimilationAction.NO_SCHEMA)
        with pytest.raises(AttributeError):
            r.action = AssimilationAction.SKIPPED  # type: ignore[misc]


class TestExtractSharedEntities:
    def test_finds_capitalized_terms(self) -> None:
        contents = [
            "Django is a framework used by many teams",
            "Django supports advanced queries and indexing",
            "We switched to Django for better performance",
        ]
        shared = _extract_shared_entities(contents)
        assert "Django" in shared

    def test_empty_input(self) -> None:
        assert _extract_shared_entities([]) == []

    def test_no_shared_entities(self) -> None:
        contents = ["hello world", "foo bar"]
        shared = _extract_shared_entities(contents)
        assert shared == []


class TestAssimilateOrAccommodate:
    @pytest.mark.asyncio
    async def test_disabled(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = False
        neuron = _make_neuron("test", ["python"])
        storage = AsyncMock()

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.SKIPPED

    @pytest.mark.asyncio
    async def test_no_tags(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 10
        neuron = _make_neuron("test", [])
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.NO_SCHEMA

    @pytest.mark.asyncio
    async def test_too_few_domain_memories(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 10
        neuron = _make_neuron("Python async patterns", ["python"])

        storage = AsyncMock()
        # No existing schemas
        storage.find_neurons = AsyncMock(
            side_effect=[
                [],  # schema search
                [_make_neuron(f"fact {i}", ["python"]) for i in range(5)],  # domain search
            ]
        )

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.NO_SCHEMA

    @pytest.mark.asyncio
    async def test_schema_created(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 10
        neuron = _make_neuron("Python async patterns", ["python"])

        domain_neurons = [_make_neuron(f"Python fact {i}", ["python"]) for i in range(12)]
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=[
                [],  # no existing schemas
                domain_neurons,  # enough domain memories
            ]
        )
        storage.add_neuron = AsyncMock()
        storage.add_synapse = AsyncMock()

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.SCHEMA_CREATED
        assert result.schema_id is not None
        assert result.version == 1
        storage.add_neuron.assert_called_once()

    @pytest.mark.asyncio
    async def test_assimilated(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 10

        schema = _make_neuron("Schema: python patterns", ["python"], NeuronType.SCHEMA)
        schema_meta = {**schema.metadata, "schema_version": 1}
        from dataclasses import replace

        schema = replace(schema, metadata=schema_meta)

        neuron = _make_neuron("Python decorators are useful", ["python"])
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[schema])
        storage.add_synapse = AsyncMock()

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.ASSIMILATED
        assert result.schema_id == schema.id
        storage.add_synapse.assert_called_once()

    @pytest.mark.asyncio
    async def test_accommodated(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 10

        schema = _make_neuron("Python is fast for development", ["python"], NeuronType.SCHEMA)
        schema_meta = {**schema.metadata, "schema_version": 1}
        from dataclasses import replace

        schema = replace(schema, metadata=schema_meta)

        # Contradictory memory
        neuron = _make_neuron("Python is not fast for development", ["python"])
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[schema])
        storage.add_neuron = AsyncMock()
        storage.add_synapse = AsyncMock()

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.ACCOMMODATED
        assert result.version == 2
        storage.add_neuron.assert_called_once()  # new schema created

    @pytest.mark.asyncio
    async def test_small_brain_skipped(self) -> None:
        """With schema_min_cluster_size=200, small brains skip schema creation."""
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 200

        neuron = _make_neuron("test fact", ["python"])
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=[
                [],  # no schemas
                [_make_neuron(f"fact {i}", ["python"]) for i in range(50)],
            ]
        )

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.NO_SCHEMA


class TestBatchSchemaAssimilation:
    @pytest.mark.asyncio
    async def test_disabled(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = False
        storage = AsyncMock()

        count = await batch_schema_assimilation(storage, config)
        assert count == 0

    @pytest.mark.asyncio
    async def test_dry_run(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 3

        neurons = [_make_neuron(f"fact {i}", ["python"]) for i in range(5)]
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(
            side_effect=[
                [],  # no existing schemas
                neurons,  # all neurons
            ]
        )

        count = await batch_schema_assimilation(storage, config, dry_run=True)
        assert count >= 1
        storage.add_neuron.assert_not_called()


class TestPostEncodeNeuroHook:
    """Test that schema assimilation is wired as a post-encode hook."""

    @pytest.mark.asyncio
    async def test_post_encode_calls_assimilate(self) -> None:
        """When schema_assimilation_enabled=True and brain is large enough, triggers assimilate."""
        from neural_memory.engine.encoder import MemoryEncoder

        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.interference_detection_enabled = False
        config.schema_min_cluster_size = 10

        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_stats = AsyncMock(return_value={"neuron_count": 500})
        storage.find_neurons = AsyncMock(return_value=[])
        storage.add_neuron = AsyncMock()
        storage.add_synapse = AsyncMock()

        encoder = MemoryEncoder(storage, config)
        anchor = _make_neuron("test content", ["python"])

        with patch(
            "neural_memory.engine.schema_assimilation.assimilate_or_accommodate",
            new_callable=AsyncMock,
        ) as mock_assim:
            mock_assim.return_value = AssimilationResult(action=AssimilationAction.NO_SCHEMA)
            await encoder._post_encode_neuro(anchor)
            mock_assim.assert_called_once_with(anchor, storage, config)

    @pytest.mark.asyncio
    async def test_post_encode_skips_when_disabled(self) -> None:
        """When schema_assimilation_enabled=False, post-encode skips assimilate."""
        from neural_memory.engine.encoder import MemoryEncoder

        config = MagicMock()
        config.schema_assimilation_enabled = False
        config.interference_detection_enabled = False

        storage = AsyncMock()
        encoder = MemoryEncoder(storage, config)
        anchor = _make_neuron("test content", ["python"])

        with patch(
            "neural_memory.engine.schema_assimilation.assimilate_or_accommodate",
            new_callable=AsyncMock,
        ) as mock_assim:
            await encoder._post_encode_neuro(anchor)
            mock_assim.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_encode_skips_small_brain(self) -> None:
        """When brain has fewer neurons than schema_min_cluster_size, skip schema assimilation."""
        from neural_memory.engine.encoder import MemoryEncoder

        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.interference_detection_enabled = False
        config.schema_min_cluster_size = 200

        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_stats = AsyncMock(return_value={"neuron_count": 50})

        encoder = MemoryEncoder(storage, config)
        anchor = _make_neuron("test content", ["python"])

        with patch(
            "neural_memory.engine.schema_assimilation.assimilate_or_accommodate",
            new_callable=AsyncMock,
        ) as mock_assim:
            await encoder._post_encode_neuro(anchor)
            mock_assim.assert_not_called()  # Skipped — brain too small

    @pytest.mark.asyncio
    async def test_post_encode_interference_wired(self) -> None:
        """When interference_detection_enabled=True, post-encode runs interference detection."""
        from neural_memory.engine.encoder import MemoryEncoder

        config = MagicMock()
        config.schema_assimilation_enabled = False
        config.interference_detection_enabled = True
        config.fan_effect_threshold = 15

        storage = AsyncMock()
        encoder = MemoryEncoder(storage, config)
        anchor = _make_neuron("test content", ["python"])

        with patch(
            "neural_memory.engine.interference.detect_interference",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_detect:
            await encoder._post_encode_neuro(anchor)
            mock_detect.assert_called_once_with(anchor, storage, config)

    @pytest.mark.asyncio
    async def test_post_encode_swallows_errors(self) -> None:
        """Post-encode hook failures don't crash encoding."""
        from neural_memory.engine.encoder import MemoryEncoder

        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.interference_detection_enabled = True
        config.schema_min_cluster_size = 10

        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_stats = AsyncMock(return_value={"neuron_count": 500})

        encoder = MemoryEncoder(storage, config)
        anchor = _make_neuron("test content", ["python"])

        with patch(
            "neural_memory.engine.schema_assimilation.assimilate_or_accommodate",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            # Should not raise
            await encoder._post_encode_neuro(anchor)
