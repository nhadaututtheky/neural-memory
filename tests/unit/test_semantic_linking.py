"""Tests for SemanticLinkingStep (Issue #10 — reduce orphan rate)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.pipeline_steps import SemanticLinkingStep
from neural_memory.utils.timeutils import utcnow


def _make_neuron(neuron_id: str, content: str, ntype: NeuronType = NeuronType.ENTITY) -> Neuron:
    return Neuron(
        id=neuron_id,
        type=ntype,
        content=content,
        created_at=utcnow(),
    )


def _make_ctx(
    entity_neurons: list[Neuron] | None = None,
    concept_neurons: list[Neuron] | None = None,
) -> SimpleNamespace:
    entities = entity_neurons or []
    concepts = concept_neurons or []
    anchor = _make_neuron("anchor-1", "anchor content")
    return SimpleNamespace(
        entity_neurons=entities,
        concept_neurons=concepts,
        time_neurons=[],
        action_neurons=[],
        intent_neurons=[],
        neurons_linked=[],
        anchor_neuron=anchor,
    )


class TestSemanticLinkingStep:
    @pytest.mark.asyncio
    async def test_links_to_existing_matching_neuron(self) -> None:
        """Entity neuron should be linked to existing neuron with same content."""
        step = SemanticLinkingStep()
        storage = AsyncMock()
        config = AsyncMock()

        new_entity = _make_neuron("new-e1", "Python", NeuronType.ENTITY)
        existing = _make_neuron("old-e1", "Python", NeuronType.ENTITY)
        storage.find_neurons.return_value = [existing]

        ctx = _make_ctx(entity_neurons=[new_entity])
        result = await step.execute(ctx, storage, config)

        storage.add_synapse.assert_called_once()
        synapse = storage.add_synapse.call_args[0][0]
        assert synapse.source_id == "new-e1"
        assert synapse.target_id == "old-e1"
        assert synapse.type == SynapseType.RELATED_TO
        assert synapse.weight == 0.4
        assert "old-e1" in result.neurons_linked

    @pytest.mark.asyncio
    async def test_skips_self_and_same_encode_neurons(self) -> None:
        """Should not link to self or neurons created in the same encode."""
        step = SemanticLinkingStep()
        storage = AsyncMock()
        config = AsyncMock()

        n1 = _make_neuron("n1", "FastAPI", NeuronType.ENTITY)
        n2 = _make_neuron("n2", "FastAPI", NeuronType.ENTITY)
        # find_neurons returns both n1 (self) and n2 (same encode)
        storage.find_neurons.return_value = [n1, n2]

        ctx = _make_ctx(entity_neurons=[n1, n2])
        result = await step.execute(ctx, storage, config)

        storage.add_synapse.assert_not_called()
        assert len(result.neurons_linked) == 0

    @pytest.mark.asyncio
    async def test_caps_links_per_neuron(self) -> None:
        """Should create at most MAX_LINKS_PER_NEURON links."""
        step = SemanticLinkingStep()
        storage = AsyncMock()
        config = AsyncMock()

        new_entity = _make_neuron("new-1", "Docker", NeuronType.ENTITY)
        existing = [_make_neuron(f"old-{i}", "Docker", NeuronType.ENTITY) for i in range(7)]
        storage.find_neurons.return_value = existing

        ctx = _make_ctx(entity_neurons=[new_entity])
        await step.execute(ctx, storage, config)

        assert storage.add_synapse.call_count == 5  # MAX_LINKS_PER_NEURON

    @pytest.mark.asyncio
    async def test_skips_short_content(self) -> None:
        """Neurons with content < 3 chars should be skipped."""
        step = SemanticLinkingStep()
        storage = AsyncMock()
        config = AsyncMock()

        short = _make_neuron("n1", "ab", NeuronType.ENTITY)
        ctx = _make_ctx(entity_neurons=[short])
        await step.execute(ctx, storage, config)

        storage.find_neurons.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_no_linkable_neurons(self) -> None:
        """Empty entity/concept lists should return immediately."""
        step = SemanticLinkingStep()
        storage = AsyncMock()
        config = AsyncMock()

        ctx = _make_ctx()
        result = await step.execute(ctx, storage, config)

        storage.find_neurons.assert_not_called()
        assert result is ctx

    @pytest.mark.asyncio
    async def test_handles_duplicate_synapse_gracefully(self) -> None:
        """ValueError from add_synapse (duplicate) should be caught."""
        step = SemanticLinkingStep()
        storage = AsyncMock()
        config = AsyncMock()

        new_entity = _make_neuron("n1", "Redis", NeuronType.ENTITY)
        existing = _make_neuron("old-1", "Redis", NeuronType.ENTITY)
        storage.find_neurons.return_value = [existing]
        storage.add_synapse.side_effect = ValueError("Synapse already exists")

        ctx = _make_ctx(entity_neurons=[new_entity])
        result = await step.execute(ctx, storage, config)

        # Should not crash, but link not added
        assert "old-1" not in result.neurons_linked
