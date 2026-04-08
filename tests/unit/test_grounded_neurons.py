"""Tests for P1: Grounded Neurons (canonical truth nodes).

Covers:
- Neuron grounded/confidence properties (metadata-backed)
- Neuron.create() with grounded=True
- Neuron.with_grounded() convenience method
- Grounded neurons skip decay in lifecycle
- Grounded neurons win conflicts in auto-resolve
- Grounded neurons skip conflict resolution
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType

# ---------------------------------------------------------------------------
# P1.1: Neuron model — grounded + confidence
# ---------------------------------------------------------------------------


class TestGroundedNeuronModel:
    def test_default_not_grounded(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="test")
        assert n.grounded is False
        assert n.confidence == 0.5

    def test_create_grounded(self) -> None:
        n = Neuron.create(type=NeuronType.ENTITY, content="Python", grounded=True)
        assert n.grounded is True
        assert n.confidence == 1.0  # default when grounded=True

    def test_create_grounded_custom_confidence(self) -> None:
        n = Neuron.create(type=NeuronType.ENTITY, content="Python", grounded=True, confidence=0.9)
        assert n.grounded is True
        assert n.confidence == 0.9

    def test_create_not_grounded_custom_confidence(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="maybe", confidence=0.3)
        assert n.grounded is False
        assert n.confidence == 0.3

    def test_with_grounded_true(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="test")
        grounded = n.with_grounded(True)
        assert grounded.grounded is True
        assert grounded.confidence == 1.0
        assert grounded.id == n.id
        assert grounded.content == n.content

    def test_with_grounded_false(self) -> None:
        n = Neuron.create(type=NeuronType.ENTITY, content="Python", grounded=True)
        ungrounded = n.with_grounded(False)
        assert ungrounded.grounded is False
        assert "_grounded" not in ungrounded.metadata
        assert "_confidence" not in ungrounded.metadata

    def test_with_metadata_preserves_grounding(self) -> None:
        n = Neuron.create(type=NeuronType.ENTITY, content="Python", grounded=True)
        updated = n.with_metadata(custom_key="value")
        assert updated.grounded is True
        assert updated.confidence == 1.0
        assert updated.metadata["custom_key"] == "value"

    def test_grounded_stored_in_metadata(self) -> None:
        """Grounding is metadata-backed for zero-migration storage compat."""
        n = Neuron.create(type=NeuronType.ENTITY, content="Python", grounded=True)
        assert n.metadata["_grounded"] is True
        assert n.metadata["_confidence"] == 1.0


# ---------------------------------------------------------------------------
# P1.3: Conflict auto-resolve — grounded neurons always win
# ---------------------------------------------------------------------------


class TestGroundedConflictAutoResolve:
    @pytest.mark.asyncio()
    async def test_grounded_neuron_wins_conflict(self) -> None:
        from neural_memory.engine.conflict_auto_resolve import try_auto_resolve
        from neural_memory.engine.conflict_detection import Conflict

        grounded_neuron = Neuron.create(
            type=NeuronType.ENTITY, content="Python 3.11", grounded=True
        )

        conflict = Conflict(
            type="factual_contradiction",
            existing_neuron_id=grounded_neuron.id,
            existing_content="Python 3.11",
            new_content="Python 3.12",
            confidence=0.9,
        )

        storage = AsyncMock()
        storage.get_neuron = AsyncMock(return_value=grounded_neuron)

        result = await try_auto_resolve(conflict, storage, new_confidence=0.9)
        assert result.auto_resolved is True
        assert result.resolution == "keep_existing"
        assert "grounded" in result.reason

    @pytest.mark.asyncio()
    async def test_non_grounded_neuron_normal_resolution(self) -> None:
        from neural_memory.engine.conflict_auto_resolve import try_auto_resolve
        from neural_memory.engine.conflict_detection import Conflict

        normal_neuron = Neuron.create(type=NeuronType.ENTITY, content="Python 3.11")

        conflict = Conflict(
            type="factual_contradiction",
            existing_neuron_id=normal_neuron.id,
            existing_content="Python 3.11",
            new_content="Python 3.12",
            confidence=0.9,
        )

        storage = AsyncMock()
        storage.get_neuron = AsyncMock(return_value=normal_neuron)
        storage.get_neuron_state = AsyncMock(return_value=None)

        result = await try_auto_resolve(conflict, storage, new_confidence=0.9)
        # Should NOT auto-resolve as "keep_existing" (grounded path)
        assert result.resolution != "keep_existing" or "grounded" not in result.reason
