"""Tests for conflict management tool handler."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.mcp.conflict_handler import ConflictHandler
from neural_memory.mcp.constants import MAX_CONTENT_LENGTH
from neural_memory.storage.memory_store import InMemoryStorage


class _FakeServer(ConflictHandler):
    """Minimal server stub that provides get_storage()."""

    def __init__(self, storage: InMemoryStorage) -> None:
        self._storage = storage

    async def get_storage(self) -> InMemoryStorage:
        return self._storage


# ========== Helpers ==========

BRAIN_ID = "test-brain"


def _make_storage() -> InMemoryStorage:
    """Create an InMemoryStorage with a brain context set."""
    storage = InMemoryStorage()
    storage.set_brain(BRAIN_ID)
    # Add no-op methods that the conflict handler expects from SQLite store.
    storage.disable_auto_save = lambda: None  # type: ignore[attr-defined]
    storage.enable_auto_save = lambda: None  # type: ignore[attr-defined]

    async def _batch_save() -> None:
        pass

    storage.batch_save = _batch_save  # type: ignore[attr-defined]
    return storage


async def _add_neuron(
    storage: InMemoryStorage,
    content: str,
    *,
    neuron_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    neuron_type: NeuronType = NeuronType.CONCEPT,
) -> Neuron:
    """Helper: create a neuron, add it to storage, return it."""
    nid = neuron_id or str(uuid4())
    neuron = Neuron.create(
        type=neuron_type,
        content=content,
        metadata=metadata,
        neuron_id=nid,
    )
    await storage.add_neuron(neuron)
    return neuron


async def _add_contradicts_synapse(
    storage: InMemoryStorage,
    source_id: str,
    target_id: str,
    *,
    weight: float = 0.8,
    metadata: dict[str, Any] | None = None,
) -> Synapse:
    """Helper: create a CONTRADICTS synapse between two neurons."""
    synapse = Synapse.create(
        source_id=source_id,
        target_id=target_id,
        type=SynapseType.CONTRADICTS,
        weight=weight,
        metadata=metadata,
    )
    await storage.add_synapse(synapse)
    return synapse


# ========== List Action ==========


class TestConflictsList:
    """Tests for _conflicts_list action."""

    async def test_list_empty(self) -> None:
        """No CONTRADICTS synapses results in empty list."""
        storage = _make_storage()
        server = _FakeServer(storage)

        result = await server._conflicts({"action": "list"})

        assert result == {"conflicts": [], "count": 0}

    async def test_list_with_disputed(self) -> None:
        """A CONTRADICTS synapse between two neurons appears in the list."""
        storage = _make_storage()
        server = _FakeServer(storage)

        existing = await _add_neuron(
            storage,
            "We use PostgreSQL for our database",
            metadata={"_disputed": True, "_pre_dispute_activation": 0.8},
        )
        disputing = await _add_neuron(
            storage,
            "We use MySQL for our database",
        )
        synapse = await _add_contradicts_synapse(
            storage,
            source_id=disputing.id,
            target_id=existing.id,
            metadata={
                "conflict_type": "factual_contradiction",
                "detected_at": "2026-01-01T00:00:00",
            },
        )

        result = await server._conflicts({"action": "list"})

        assert result["count"] == 1
        conflict = result["conflicts"][0]
        assert conflict["existing_neuron_id"] == existing.id
        assert "PostgreSQL" in conflict["content"]
        assert "MySQL" in conflict["disputed_by_preview"]
        assert conflict["conflict_type"] == "factual_contradiction"
        assert conflict["confidence"] == synapse.weight
        assert conflict["is_superseded"] is False

    async def test_list_excludes_resolved(self) -> None:
        """A CONTRADICTS synapse with _resolved metadata is excluded from list."""
        storage = _make_storage()
        server = _FakeServer(storage)

        existing = await _add_neuron(storage, "We use PostgreSQL")
        disputing = await _add_neuron(storage, "We use MySQL")
        await _add_contradicts_synapse(
            storage,
            source_id=disputing.id,
            target_id=existing.id,
            metadata={"_resolved": "keep_existing"},
        )

        result = await server._conflicts({"action": "list"})

        assert result == {"conflicts": [], "count": 0}


# ========== Resolve Action ==========


class TestConflictsResolve:
    """Tests for _conflicts_resolve action."""

    async def test_resolve_keep_existing(self) -> None:
        """keep_existing restores activation, clears dispute, supersedes disputing neuron."""
        storage = _make_storage()
        server = _FakeServer(storage)

        existing = await _add_neuron(
            storage,
            "We use PostgreSQL",
            metadata={"_disputed": True, "_pre_dispute_activation": 0.8},
        )
        # Manually set state to a reduced activation (simulating dispute reduction).
        lowered_state = NeuronState(neuron_id=existing.id, activation_level=0.3)
        await storage.update_neuron_state(lowered_state)

        disputing = await _add_neuron(storage, "We use MySQL")
        synapse = await _add_contradicts_synapse(
            storage,
            source_id=disputing.id,
            target_id=existing.id,
        )

        result = await server._conflicts(
            {
                "action": "resolve",
                "neuron_id": existing.id,
                "resolution": "keep_existing",
            }
        )

        assert result["success"] is True
        assert result["resolution"] == "keep_existing"

        # Existing neuron: activation restored, dispute cleared.
        updated_existing = await storage.get_neuron(existing.id)
        assert updated_existing is not None
        assert updated_existing.metadata.get("_disputed") is False

        restored_state = await storage.get_neuron_state(existing.id)
        assert restored_state is not None
        assert restored_state.activation_level == pytest.approx(0.8)

        # Disputing neuron superseded.
        updated_disputing = await storage.get_neuron(disputing.id)
        assert updated_disputing is not None
        assert updated_disputing.metadata.get("_superseded") is True

        # Synapse marked resolved.
        updated_synapse = await storage.get_synapse(synapse.id)
        assert updated_synapse is not None
        assert updated_synapse.metadata.get("_resolved") == "keep_existing"

    async def test_resolve_keep_new(self) -> None:
        """keep_new supersedes the existing neuron."""
        storage = _make_storage()
        server = _FakeServer(storage)

        existing = await _add_neuron(
            storage,
            "We use PostgreSQL",
            metadata={"_disputed": True},
        )
        disputing = await _add_neuron(storage, "We use MySQL")
        synapse = await _add_contradicts_synapse(
            storage,
            source_id=disputing.id,
            target_id=existing.id,
        )

        result = await server._conflicts(
            {
                "action": "resolve",
                "neuron_id": existing.id,
                "resolution": "keep_new",
            }
        )

        assert result["success"] is True

        # Existing neuron superseded.
        updated_existing = await storage.get_neuron(existing.id)
        assert updated_existing is not None
        assert updated_existing.metadata.get("_superseded") is True

        # Synapse marked resolved.
        updated_synapse = await storage.get_synapse(synapse.id)
        assert updated_synapse is not None
        assert updated_synapse.metadata.get("_resolved") == "keep_new"

    async def test_resolve_keep_both(self) -> None:
        """keep_both clears dispute on existing, marks both as conflict_resolved."""
        storage = _make_storage()
        server = _FakeServer(storage)

        existing = await _add_neuron(
            storage,
            "We use PostgreSQL",
            metadata={"_disputed": True},
        )
        disputing = await _add_neuron(storage, "We use MySQL")
        synapse = await _add_contradicts_synapse(
            storage,
            source_id=disputing.id,
            target_id=existing.id,
        )

        result = await server._conflicts(
            {
                "action": "resolve",
                "neuron_id": existing.id,
                "resolution": "keep_both",
            }
        )

        assert result["success"] is True

        # Existing: disputed cleared, conflict_resolved set.
        updated_existing = await storage.get_neuron(existing.id)
        assert updated_existing is not None
        assert updated_existing.metadata.get("_disputed") is False
        assert updated_existing.metadata.get("_conflict_resolved") is True

        # Disputing: conflict_resolved set.
        updated_disputing = await storage.get_neuron(disputing.id)
        assert updated_disputing is not None
        assert updated_disputing.metadata.get("_conflict_resolved") is True

        # Synapse marked resolved.
        updated_synapse = await storage.get_synapse(synapse.id)
        assert updated_synapse is not None
        assert updated_synapse.metadata.get("_resolved") == "keep_both"

    async def test_resolve_invalid_uuid(self) -> None:
        """Non-UUID neuron_id returns an error without touching storage."""
        storage = _make_storage()
        server = _FakeServer(storage)

        result = await server._conflicts(
            {
                "action": "resolve",
                "neuron_id": "not-a-uuid",
                "resolution": "keep_existing",
            }
        )

        assert "error" in result
        assert "Invalid neuron_id" in result["error"]

    async def test_resolve_not_disputed(self) -> None:
        """Attempting to resolve a non-disputed neuron returns an error."""
        storage = _make_storage()
        server = _FakeServer(storage)

        neuron = await _add_neuron(storage, "We use PostgreSQL")

        result = await server._conflicts(
            {
                "action": "resolve",
                "neuron_id": neuron.id,
                "resolution": "keep_existing",
            }
        )

        assert "error" in result
        assert "not disputed" in result["error"]


# ========== Check Action ==========


class TestConflictsCheck:
    """Tests for _conflicts_check action."""

    async def test_check_with_conflicts(self) -> None:
        """Checking content that contradicts an existing neuron returns potential conflicts."""
        storage = _make_storage()
        server = _FakeServer(storage)

        await _add_neuron(storage, "We use PostgreSQL for our database")

        result = await server._conflicts(
            {
                "action": "check",
                "content": "We use MySQL for our database",
            }
        )

        assert "potential_conflicts" in result
        assert result["count"] >= 1
        assert any("PostgreSQL" in c["existing_content"] for c in result["potential_conflicts"])

    async def test_check_content_length(self) -> None:
        """Content exceeding MAX_CONTENT_LENGTH returns an error."""
        storage = _make_storage()
        server = _FakeServer(storage)

        oversized = "x" * (MAX_CONTENT_LENGTH + 1)
        result = await server._conflicts(
            {
                "action": "check",
                "content": oversized,
            }
        )

        assert "error" in result
        assert "too long" in result["error"].lower()

    async def test_check_no_conflicts(self) -> None:
        """Checking content with empty storage returns no conflicts."""
        storage = _make_storage()
        server = _FakeServer(storage)

        result = await server._conflicts(
            {
                "action": "check",
                "content": "We use PostgreSQL for our database",
            }
        )

        assert result["potential_conflicts"] == []
        assert result["count"] == 0


# ========== Audit-driven tests ==========


class TestConflictsAuditFixes:
    """Tests added by post-implementation audit."""

    async def test_resolve_keep_new_clears_disputed(self) -> None:
        """keep_new must set _disputed=False on existing neuron (audit fix)."""
        storage = _make_storage()
        server = _FakeServer(storage)

        existing = await _add_neuron(
            storage,
            "We use PostgreSQL",
            metadata={"_disputed": True},
        )
        disputing = await _add_neuron(storage, "We use MySQL")
        await _add_contradicts_synapse(
            storage,
            source_id=disputing.id,
            target_id=existing.id,
        )

        await server._conflicts(
            {
                "action": "resolve",
                "neuron_id": existing.id,
                "resolution": "keep_new",
            }
        )

        updated = await storage.get_neuron(existing.id)
        assert updated is not None
        assert updated.metadata.get("_disputed") is False
        assert updated.metadata.get("_superseded") is True

    async def test_list_handles_deleted_neurons(self) -> None:
        """Deleted neurons show '(deleted)' in conflict list."""
        storage = _make_storage()
        server = _FakeServer(storage)

        # Create neurons, add synapse, then remove neurons from internal dict
        # (can't use delete_neuron because InMemoryStorage cascade-deletes synapses)
        source = await _add_neuron(storage, "temporary source")
        target = await _add_neuron(storage, "temporary target")
        await _add_contradicts_synapse(
            storage,
            source_id=source.id,
            target_id=target.id,
        )
        del storage._neurons[BRAIN_ID][source.id]
        del storage._neurons[BRAIN_ID][target.id]

        result = await server._conflicts({"action": "list"})

        assert result["count"] == 1
        conflict = result["conflicts"][0]
        assert conflict["content"] == "(deleted)"
        assert conflict["disputed_by_preview"] == "(deleted)"

    async def test_unknown_action_returns_error(self) -> None:
        """Unknown action returns error dict."""
        storage = _make_storage()
        server = _FakeServer(storage)

        result = await server._conflicts({"action": "invalid"})
        assert "error" in result
        assert "Unknown action" in result["error"]

    async def test_resolve_invalid_resolution(self) -> None:
        """Invalid resolution type returns descriptive error."""
        storage = _make_storage()
        server = _FakeServer(storage)

        nid = str(uuid4())
        result = await server._conflicts(
            {
                "action": "resolve",
                "neuron_id": nid,
                "resolution": "merge",
            }
        )

        assert "error" in result
        assert "Invalid resolution" in result["error"]

    async def test_resolve_neuron_not_found(self) -> None:
        """Resolve on non-existent neuron ID returns error."""
        storage = _make_storage()
        server = _FakeServer(storage)

        nid = str(uuid4())
        result = await server._conflicts(
            {
                "action": "resolve",
                "neuron_id": nid,
                "resolution": "keep_existing",
            }
        )

        assert "error" in result
        assert "not found" in result["error"]

    async def test_resolve_no_synapse_found(self) -> None:
        """Disputed neuron with no CONTRADICTS synapse returns error."""
        storage = _make_storage()
        server = _FakeServer(storage)

        neuron = await _add_neuron(
            storage,
            "We use PostgreSQL",
            metadata={"_disputed": True},
        )

        result = await server._conflicts(
            {
                "action": "resolve",
                "neuron_id": neuron.id,
                "resolution": "keep_existing",
            }
        )

        assert "error" in result
        assert "No unresolved CONTRADICTS synapse" in result["error"]

    async def test_check_tag_validation_too_many(self) -> None:
        """More than 50 tags returns error."""
        storage = _make_storage()
        server = _FakeServer(storage)

        result = await server._conflicts(
            {
                "action": "check",
                "content": "Some content",
                "tags": [f"tag-{i}" for i in range(60)],
            }
        )

        assert "error" in result
        assert "Too many tags" in result["error"]

    async def test_check_tag_validation_too_long(self) -> None:
        """Tag exceeding 100 chars returns error."""
        storage = _make_storage()
        server = _FakeServer(storage)

        result = await server._conflicts(
            {
                "action": "check",
                "content": "Some content",
                "tags": ["x" * 150],
            }
        )

        assert "error" in result
        assert "Invalid tag" in result["error"]

    async def test_error_messages_do_not_leak_internals(self) -> None:
        """Error responses should not contain raw exception details."""
        storage = _make_storage()
        server = _FakeServer(storage)

        # Sabotage storage to trigger an exception in list
        async def _explode(**kwargs: Any) -> list:
            msg = "sqlite3.OperationalError: database is locked"
            raise RuntimeError(msg)

        storage.get_synapses = _explode  # type: ignore[assignment]

        result = await server._conflicts({"action": "list"})

        assert "error" in result
        assert "sqlite3" not in result["error"]
        assert "Check server logs" in result["error"]
