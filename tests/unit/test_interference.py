"""Tests for interference forgetting."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.interference import (
    InterferenceType,
    batch_interference_scan,
    detect_interference,
    resolve_interference,
)
from neural_memory.utils.timeutils import utcnow


def _make_neuron(
    content: str,
    tags: list[str] | None = None,
    created_offset_hours: int = 0,
) -> Neuron:
    from dataclasses import replace

    n = Neuron.create(
        content=content,
        type=NeuronType.CONCEPT,
        metadata={"tags": tags or []},
    )
    if created_offset_hours:
        return replace(n, created_at=utcnow() - timedelta(hours=created_offset_hours))
    return n


class TestDetectInterference:
    @pytest.mark.asyncio
    async def test_disabled(self) -> None:
        config = MagicMock()
        config.interference_detection_enabled = False
        neuron = _make_neuron("test", ["python"])
        storage = AsyncMock()

        results = await detect_interference(neuron, storage, config)
        assert results == []

    @pytest.mark.asyncio
    async def test_no_tags(self) -> None:
        config = MagicMock()
        config.interference_detection_enabled = True
        neuron = _make_neuron("test", [])
        storage = AsyncMock()

        results = await detect_interference(neuron, storage, config)
        assert results == []

    @pytest.mark.asyncio
    async def test_similar_retroactive(self) -> None:
        config = MagicMock()
        config.interference_detection_enabled = True
        config.fan_effect_threshold = 15

        new_neuron = _make_neuron("Python async await patterns are complex", ["python"])
        old_neuron = _make_neuron(
            "Python async await patterns are complicated", ["python"], created_offset_hours=24
        )

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[old_neuron])

        results = await detect_interference(new_neuron, storage, config)
        retro = [r for r in results if r.interference_type == InterferenceType.RETROACTIVE]
        assert len(retro) >= 1
        assert retro[0].score > 0.0

    @pytest.mark.asyncio
    async def test_low_similarity_no_interference(self) -> None:
        config = MagicMock()
        config.interference_detection_enabled = True
        config.fan_effect_threshold = 15

        new_neuron = _make_neuron("The weather is sunny today", ["weather"])
        old_neuron = _make_neuron(
            "Database indexing improves query speed", ["weather"], created_offset_hours=24
        )

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[old_neuron])

        results = await detect_interference(new_neuron, storage, config)
        # Should have no retroactive/proactive (content too different)
        non_fan = [r for r in results if r.interference_type != InterferenceType.FAN_EFFECT]
        assert len(non_fan) == 0

    @pytest.mark.asyncio
    async def test_fan_effect(self) -> None:
        config = MagicMock()
        config.interference_detection_enabled = True
        config.fan_effect_threshold = 3  # Low threshold for testing

        new_neuron = _make_neuron("Python fact new", ["python"])
        candidates = [
            _make_neuron(
                f"Python fact {i} about different topics entirely",
                ["python"],
                created_offset_hours=i,
            )
            for i in range(5)
        ]

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=candidates)

        results = await detect_interference(new_neuron, storage, config)
        fan = [r for r in results if r.interference_type == InterferenceType.FAN_EFFECT]
        assert len(fan) == 1

    @pytest.mark.asyncio
    async def test_fan_threshold_configurable(self) -> None:
        config = MagicMock()
        config.interference_detection_enabled = True
        config.fan_effect_threshold = 100  # Very high threshold

        new_neuron = _make_neuron("Python fact", ["python"])
        candidates = [
            _make_neuron(f"Different topic {i}", ["python"], created_offset_hours=i)
            for i in range(10)
        ]

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=candidates)

        results = await detect_interference(new_neuron, storage, config)
        fan = [r for r in results if r.interference_type == InterferenceType.FAN_EFFECT]
        assert len(fan) == 0  # Not enough for threshold=100


class TestResolveInterference:
    @pytest.mark.asyncio
    async def test_retroactive_creates_contradicts(self) -> None:
        from neural_memory.engine.interference import InterferenceResult

        config = MagicMock()
        new_neuron = _make_neuron("new content", ["python"])
        results = [
            InterferenceResult(
                neuron_id="old-123",
                score=0.8,
                interference_type=InterferenceType.RETROACTIVE,
            )
        ]

        old_syn = Synapse.create(
            source_id="old-123",
            target_id="other-456",
            type=SynapseType.CO_OCCURS,
            weight=0.5,
        )

        storage = AsyncMock()
        storage.add_synapse = AsyncMock()
        storage.get_synapses = AsyncMock(return_value=[old_syn])
        storage.update_synapse = AsyncMock()

        report = await resolve_interference(results, new_neuron, storage, config)
        assert report.contradicts_created == 1
        storage.add_synapse.assert_called()

    @pytest.mark.asyncio
    async def test_proactive_boosts_priority(self) -> None:
        from neural_memory.engine.interference import InterferenceResult

        config = MagicMock()
        new_neuron = _make_neuron("new content", ["python"])
        results = [
            InterferenceResult(
                neuron_id="old-123",
                score=0.5,
                interference_type=InterferenceType.PROACTIVE,
            )
        ]

        storage = AsyncMock()
        report = await resolve_interference(results, new_neuron, storage, config)
        assert report.priorities_boosted == 1

    @pytest.mark.asyncio
    async def test_report_tracks_all(self) -> None:
        from neural_memory.engine.interference import InterferenceResult

        config = MagicMock()
        new_neuron = _make_neuron("content", ["python"])
        results = [
            InterferenceResult("n1", 0.8, InterferenceType.RETROACTIVE),
            InterferenceResult("n2", 0.5, InterferenceType.PROACTIVE),
            InterferenceResult("n3", 0.6, InterferenceType.FAN_EFFECT),
        ]

        old_syn = Synapse.create(
            source_id="n1",
            target_id="n99",
            type=SynapseType.CO_OCCURS,
            weight=0.5,
        )
        storage = AsyncMock()
        storage.get_synapses = AsyncMock(return_value=[old_syn])
        storage.update_synapse = AsyncMock()
        storage.add_synapse = AsyncMock()

        report = await resolve_interference(results, new_neuron, storage, config)
        assert report.total_detected == 3
        assert report.contradicts_created == 1
        assert report.priorities_boosted == 1
        assert report.fan_effects_flagged == 1


class TestBatchInterferenceScan:
    @pytest.mark.asyncio
    async def test_disabled(self) -> None:
        config = MagicMock()
        config.interference_detection_enabled = False
        storage = AsyncMock()

        report = await batch_interference_scan(storage, config)
        assert report.fan_effects_flagged == 0

    @pytest.mark.asyncio
    async def test_detects_fan_effects(self) -> None:
        config = MagicMock()
        config.interference_detection_enabled = True
        config.fan_effect_threshold = 3

        neurons = [_make_neuron(f"Python fact {i}", ["python"]) for i in range(5)]
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=neurons)

        report = await batch_interference_scan(storage, config)
        assert report.fan_effects_flagged >= 1
