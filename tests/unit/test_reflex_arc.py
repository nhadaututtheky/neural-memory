"""Unit tests for Reflex Arc — Phase 1.

Tests cover:
- Neuron.reflex property + with_reflex() immutability
- BrainConfig.max_reflexes default
- check_conflicts() SimHash-based conflict detection
- pin_as_reflex() max cap, conflict resolution, supersedes synapse
- unpin_reflex() round-trip
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.core.neuron import Neuron, NeuronStatus, NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.reflex_conflict import (
    REFLEX_CONFLICT_THRESHOLD,
    ReflexConflict,
    ReflexPinResult,
    check_conflicts,
    pin_as_reflex,
    unpin_reflex,
)
from neural_memory.utils.simhash import hamming_distance, simhash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_neuron(
    content: str = "test content",
    reflex: bool = False,
    neuron_id: str | None = None,
    content_hash: int = 0,
) -> Neuron:
    n = Neuron.create(
        type=NeuronType.CONCEPT,
        content=content,
        neuron_id=neuron_id or None,
        content_hash=content_hash,
    )
    if reflex:
        n = n.with_reflex(pinned=True)
    return n


def _mock_storage(
    existing_reflexes: list[Neuron] | None = None,
    get_neuron_result: Neuron | None = None,
) -> AsyncMock:
    storage = AsyncMock()
    storage.find_reflex_neurons = AsyncMock(return_value=existing_reflexes or [])
    storage.get_neuron = AsyncMock(return_value=get_neuron_result)
    storage.update_neuron = AsyncMock()
    storage.add_synapse = AsyncMock(return_value="synapse-id")
    return storage


# ---------------------------------------------------------------------------
# Neuron.reflex property
# ---------------------------------------------------------------------------


class TestNeuronReflexProperty:
    def test_default_is_false(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="hello")
        assert n.reflex is False

    def test_reads_metadata_key(self) -> None:
        n = Neuron.create(
            type=NeuronType.CONCEPT,
            content="hello",
            metadata={"_reflex": True},
        )
        assert n.reflex is True

    def test_with_reflex_pin(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="hello")
        pinned = n.with_reflex(pinned=True)
        assert pinned.reflex is True
        # Original is unchanged (immutability)
        assert n.reflex is False

    def test_with_reflex_unpin(self) -> None:
        n = Neuron.create(
            type=NeuronType.CONCEPT,
            content="hello",
            metadata={"_reflex": True},
        )
        unpinned = n.with_reflex(pinned=False)
        assert unpinned.reflex is False
        # Original is unchanged
        assert n.reflex is True

    def test_with_reflex_preserves_other_metadata(self) -> None:
        n = Neuron.create(
            type=NeuronType.CONCEPT,
            content="hello",
            metadata={"custom_key": "value", "_abstraction_level": 2},
        )
        pinned = n.with_reflex(pinned=True)
        assert pinned.metadata["custom_key"] == "value"
        assert pinned.metadata["_abstraction_level"] == 2
        assert pinned.reflex is True

    def test_with_reflex_preserves_identity(self) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="hello", neuron_id="n-123")
        pinned = n.with_reflex(pinned=True)
        assert pinned.id == "n-123"
        assert pinned.content == "hello"
        assert pinned.type == NeuronType.CONCEPT


# ---------------------------------------------------------------------------
# BrainConfig.max_reflexes
# ---------------------------------------------------------------------------


class TestBrainConfigMaxReflexes:
    def test_default_value(self) -> None:
        config = BrainConfig()
        assert config.max_reflexes == 20

    def test_custom_value(self) -> None:
        config = BrainConfig(max_reflexes=5)
        assert config.max_reflexes == 5


# ---------------------------------------------------------------------------
# check_conflicts()
# ---------------------------------------------------------------------------


class TestCheckConflicts:
    @pytest.mark.asyncio
    async def test_no_conflicts_when_no_reflexes(self) -> None:
        neuron = _make_neuron(content="always use parameterized SQL")
        storage = _mock_storage(existing_reflexes=[])
        conflicts = await check_conflicts(neuron, storage)
        assert conflicts == []

    @pytest.mark.asyncio
    async def test_no_conflict_with_dissimilar_content(self) -> None:
        neuron = _make_neuron(content="always use parameterized SQL queries")
        existing = _make_neuron(
            content="the weather today is sunny and warm",
            reflex=True,
            neuron_id="existing-1",
        )
        storage = _mock_storage(existing_reflexes=[existing])
        conflicts = await check_conflicts(neuron, storage)
        assert conflicts == []

    @pytest.mark.asyncio
    async def test_detects_conflict_with_similar_content(self) -> None:
        content_a = "always use parameterized SQL queries for safety"
        content_b = "always use parameterized SQL queries for better safety"
        # Verify these are actually similar via SimHash
        hash_a = simhash(content_a)
        hash_b = simhash(content_b)
        dist = hamming_distance(hash_a, hash_b)
        assert dist <= REFLEX_CONFLICT_THRESHOLD, (
            f"Test assumption failed: hamming={dist}, need <={REFLEX_CONFLICT_THRESHOLD}"
        )

        neuron = _make_neuron(content=content_a, neuron_id="new-1")
        existing = _make_neuron(content=content_b, reflex=True, neuron_id="existing-1")
        storage = _mock_storage(existing_reflexes=[existing])

        conflicts = await check_conflicts(neuron, storage)
        assert len(conflicts) == 1
        assert conflicts[0].existing_id == "existing-1"
        assert conflicts[0].hamming_distance <= REFLEX_CONFLICT_THRESHOLD
        assert conflicts[0].action == "supersede"

    @pytest.mark.asyncio
    async def test_skips_self_comparison(self) -> None:
        neuron = _make_neuron(content="test content", neuron_id="same-id")
        existing = _make_neuron(content="test content", reflex=True, neuron_id="same-id")
        storage = _mock_storage(existing_reflexes=[existing])

        conflicts = await check_conflicts(neuron, storage)
        assert conflicts == []

    @pytest.mark.asyncio
    async def test_uses_content_hash_if_available(self) -> None:
        content = "always use parameterized SQL queries"
        h = simhash(content)
        neuron = _make_neuron(content=content, neuron_id="new-1", content_hash=h)
        existing = _make_neuron(
            content=content,
            reflex=True,
            neuron_id="existing-1",
            content_hash=h,
        )
        storage = _mock_storage(existing_reflexes=[existing])

        conflicts = await check_conflicts(neuron, storage)
        # Same hash → hamming 0 → conflict
        assert len(conflicts) == 1
        assert conflicts[0].hamming_distance == 0

    @pytest.mark.asyncio
    async def test_empty_content_no_crash(self) -> None:
        neuron = _make_neuron(content="")
        storage = _mock_storage(existing_reflexes=[_make_neuron(content="something", reflex=True)])
        conflicts = await check_conflicts(neuron, storage)
        assert conflicts == []


# ---------------------------------------------------------------------------
# pin_as_reflex()
# ---------------------------------------------------------------------------


class TestPinAsReflex:
    @pytest.mark.asyncio
    async def test_pin_success(self) -> None:
        neuron = _make_neuron(content="important rule", neuron_id="n-1")
        storage = _mock_storage(existing_reflexes=[], get_neuron_result=neuron)
        config = BrainConfig(max_reflexes=20)

        result = await pin_as_reflex("n-1", storage, config)

        assert result.pinned is True
        assert result.error is None
        assert result.conflicts_resolved == []
        # Verify update_neuron was called with reflex=True
        storage.update_neuron.assert_called_once()
        updated = storage.update_neuron.call_args[0][0]
        assert updated.reflex is True

    @pytest.mark.asyncio
    async def test_pin_neuron_not_found(self) -> None:
        storage = _mock_storage(get_neuron_result=None)
        config = BrainConfig()

        result = await pin_as_reflex("nonexistent", storage, config)

        assert result.pinned is False
        assert result.error == "Neuron not found"

    @pytest.mark.asyncio
    async def test_pin_already_reflex_is_noop(self) -> None:
        neuron = _make_neuron(content="rule", neuron_id="n-1", reflex=True)
        storage = _mock_storage(get_neuron_result=neuron)
        config = BrainConfig()

        result = await pin_as_reflex("n-1", storage, config)

        assert result.pinned is True
        assert result.conflicts_resolved == []
        storage.update_neuron.assert_not_called()

    @pytest.mark.asyncio
    async def test_pin_max_cap_reached(self) -> None:
        neuron = _make_neuron(content="new rule", neuron_id="n-new")
        # 3 existing reflexes, max is 3
        existing = [
            _make_neuron(content=f"rule {i}", reflex=True, neuron_id=f"n-{i}") for i in range(3)
        ]
        storage = _mock_storage(existing_reflexes=existing, get_neuron_result=neuron)
        config = BrainConfig(max_reflexes=3)

        result = await pin_as_reflex("n-new", storage, config)

        assert result.pinned is False
        assert "Max reflexes reached" in (result.error or "")

    @pytest.mark.asyncio
    async def test_pin_with_conflict_auto_supersedes(self) -> None:
        content = "always validate user input before processing"
        new_neuron = _make_neuron(content=content, neuron_id="n-new")
        old_neuron = _make_neuron(content=content, neuron_id="n-old", reflex=True)

        # get_neuron returns new_neuron first, then old_neuron for conflict resolution
        storage = AsyncMock()
        storage.get_neuron = AsyncMock(side_effect=[new_neuron, old_neuron, new_neuron])
        # First call: check_conflicts returns [old_neuron]
        # Second call (after unpin): returns [] (old unpinned, slot freed)
        storage.find_reflex_neurons = AsyncMock(side_effect=[[old_neuron], []])
        storage.update_neuron = AsyncMock()
        storage.add_synapse = AsyncMock(return_value="syn-1")
        config = BrainConfig(max_reflexes=5)

        result = await pin_as_reflex("n-new", storage, config)

        assert result.pinned is True
        assert len(result.conflicts_resolved) == 1
        assert result.conflicts_resolved[0].existing_id == "n-old"

        # Verify supersedes synapse was created
        storage.add_synapse.assert_called_once()
        synapse_arg = storage.add_synapse.call_args[0][0]
        assert synapse_arg.source_id == "n-new"
        assert synapse_arg.target_id == "n-old"
        assert synapse_arg.type == SynapseType.SUPERSEDES

        # Verify old neuron was unpinned + new neuron was pinned
        assert storage.update_neuron.call_count == 2
        first_update = storage.update_neuron.call_args_list[0][0][0]
        assert first_update.id == "n-old"
        assert first_update.reflex is False
        # Item #2: superseded loser gets status flipped + winner recorded.
        from neural_memory.core.neuron import NeuronStatus

        assert first_update.status == NeuronStatus.SUPERSEDED
        assert first_update.metadata.get("_superseded_by") == "n-new"
        second_update = storage.update_neuron.call_args_list[1][0][0]
        assert second_update.id == "n-new"
        assert second_update.reflex is True
        assert second_update.status == NeuronStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_pin_revives_formerly_superseded_winner(self) -> None:
        """Item #2 review C1: a neuron pinned as reflex always becomes ACTIVE.

        Previously, if `neuron` carried a stale `_status="superseded"` from
        a prior cycle, `pin_as_reflex` would store it with both `_reflex=True`
        and `_status="superseded"` — the status filter would then hard-drop
        it from every recall.
        """
        content = "always validate user input before processing"
        formerly_superseded = _make_neuron(content=content, neuron_id="n-revive").with_status(
            NeuronStatus.SUPERSEDED, superseded_by="old-winner"
        )

        storage = AsyncMock()
        storage.get_neuron = AsyncMock(return_value=formerly_superseded)
        storage.find_reflex_neurons = AsyncMock(return_value=[])
        storage.update_neuron = AsyncMock()
        storage.add_synapse = AsyncMock(return_value="syn-x")
        config = BrainConfig(max_reflexes=10)

        result = await pin_as_reflex("n-revive", storage, config)
        assert result.pinned is True

        stored = storage.update_neuron.call_args_list[-1][0][0]
        assert stored.id == "n-revive"
        assert stored.reflex is True
        assert stored.status == NeuronStatus.ACTIVE
        # Revive should also clear the now-stale winner reference (M1).
        assert "_superseded_by" not in stored.metadata

    @pytest.mark.asyncio
    async def test_pin_conflict_frees_slot_for_cap(self) -> None:
        """Conflict resolution unpins old reflex, freeing a slot under the cap."""
        content = "use async for all IO operations"
        new_neuron = _make_neuron(content=content, neuron_id="n-new")
        old_neuron = _make_neuron(content=content, neuron_id="n-old", reflex=True)
        other_reflex = _make_neuron(content="unrelated rule xyz", neuron_id="n-other", reflex=True)

        storage = AsyncMock()
        storage.get_neuron = AsyncMock(side_effect=[new_neuron, old_neuron, new_neuron])
        # First call for conflict check: 2 existing reflexes
        # Second call for cap check: only 1 left (old was unpinned)
        storage.find_reflex_neurons = AsyncMock(
            side_effect=[[old_neuron, other_reflex], [other_reflex]]
        )
        storage.update_neuron = AsyncMock()
        storage.add_synapse = AsyncMock(return_value="syn-1")
        config = BrainConfig(max_reflexes=2)  # Cap = 2, currently 2, but conflict frees 1

        result = await pin_as_reflex("n-new", storage, config)

        assert result.pinned is True
        assert len(result.conflicts_resolved) == 1


# ---------------------------------------------------------------------------
# unpin_reflex()
# ---------------------------------------------------------------------------


class TestUnpinReflex:
    @pytest.mark.asyncio
    async def test_unpin_success(self) -> None:
        neuron = _make_neuron(content="rule", neuron_id="n-1", reflex=True)
        storage = _mock_storage(get_neuron_result=neuron)

        result = await unpin_reflex("n-1", storage)

        assert result is True
        storage.update_neuron.assert_called_once()
        updated = storage.update_neuron.call_args[0][0]
        assert updated.reflex is False

    @pytest.mark.asyncio
    async def test_unpin_not_found(self) -> None:
        storage = _mock_storage(get_neuron_result=None)
        result = await unpin_reflex("nonexistent", storage)
        assert result is False

    @pytest.mark.asyncio
    async def test_unpin_not_a_reflex(self) -> None:
        neuron = _make_neuron(content="rule", neuron_id="n-1", reflex=False)
        storage = _mock_storage(get_neuron_result=neuron)
        result = await unpin_reflex("n-1", storage)
        assert result is False
        storage.update_neuron.assert_not_called()


# ---------------------------------------------------------------------------
# ReflexConflict / ReflexPinResult dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_reflex_conflict_defaults(self) -> None:
        c = ReflexConflict(
            existing_id="n-1",
            existing_content="some rule",
            hamming_distance=5,
        )
        assert c.action == "supersede"
        assert c.existing_id == "n-1"

    def test_reflex_pin_result_success(self) -> None:
        r = ReflexPinResult(pinned=True, conflicts_resolved=[])
        assert r.error is None

    def test_reflex_pin_result_failure(self) -> None:
        r = ReflexPinResult(pinned=False, conflicts_resolved=[], error="cap reached")
        assert r.error == "cap reached"
