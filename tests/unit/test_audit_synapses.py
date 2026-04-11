"""Tests for audit synapses and provenance (Phase 4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.synapse import Synapse, SynapseType

# ──────────────────── SynapseType Audit Values ────────────────────


class TestAuditSynapseTypes:
    """Verify audit synapse types exist."""

    def test_stored_by_exists(self) -> None:
        assert SynapseType.STORED_BY.value == "stored_by"

    def test_verified_at_exists(self) -> None:
        assert SynapseType.VERIFIED_AT.value == "verified_at"

    def test_approved_by_exists(self) -> None:
        assert SynapseType.APPROVED_BY.value == "approved_by"

    def test_audit_types_unidirectional(self) -> None:
        from neural_memory.core.synapse import BIDIRECTIONAL_TYPES

        assert SynapseType.STORED_BY not in BIDIRECTIONAL_TYPES
        assert SynapseType.VERIFIED_AT not in BIDIRECTIONAL_TYPES
        assert SynapseType.APPROVED_BY not in BIDIRECTIONAL_TYPES


# ──────────────────── Synapse Creation ────────────────────


class TestAuditSynapseCreation:
    """Verify audit synapses can be created."""

    def test_create_stored_by(self) -> None:
        syn = Synapse.create(
            source_id="neuron-1",
            target_id="neuron-1",
            type=SynapseType.STORED_BY,
            metadata={"actor": "claude", "tool": "nmem_remember"},
        )
        assert syn.type == SynapseType.STORED_BY
        assert syn.metadata["actor"] == "claude"
        assert syn.weight == 0.5  # default

    def test_create_verified_at(self) -> None:
        syn = Synapse.create(
            source_id="neuron-1",
            target_id="neuron-1",
            type=SynapseType.VERIFIED_AT,
            weight=1.0,
            metadata={"actor": "admin"},
        )
        assert syn.type == SynapseType.VERIFIED_AT
        assert syn.weight == 1.0

    def test_create_approved_by(self) -> None:
        syn = Synapse.create(
            source_id="neuron-1",
            target_id="neuron-1",
            type=SynapseType.APPROVED_BY,
            metadata={"actor": "manager"},
        )
        assert syn.type == SynapseType.APPROVED_BY


# ──────────────────── _build_citation_audit ────────────────────


# ──────────────────── Provenance Handler ────────────────────


class TestProvenanceHandler:
    """Test nmem_provenance MCP tool handler."""

    def _make_handler(self) -> MagicMock:
        """Create a mock ToolHandler with _provenance method."""
        from neural_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._provenance = ToolHandler._provenance.__get__(handler, type(handler))
        handler._provenance_trace = ToolHandler._provenance_trace.__get__(handler, type(handler))
        handler._provenance_add_audit = ToolHandler._provenance_add_audit.__get__(
            handler, type(handler)
        )
        return handler

    @pytest.mark.asyncio
    async def test_trace_empty(self) -> None:
        handler = self._make_handler()
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_neuron = AsyncMock(return_value=MagicMock())
        storage.get_synapses = AsyncMock(return_value=[])
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._provenance({"action": "trace", "neuron_id": "n-1"})
        assert result["neuron_id"] == "n-1"
        assert result["provenance"] == []
        assert result["has_source"] is False
        assert result["is_verified"] is False

    @pytest.mark.asyncio
    async def test_trace_with_chain(self) -> None:
        handler = self._make_handler()
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_neuron = AsyncMock(return_value=MagicMock())

        source_syn = Synapse.create(
            source_id="src-1",
            target_id="n-1",
            type=SynapseType.SOURCE_OF,
        )
        stored_syn = Synapse.create(
            source_id="n-1",
            target_id="n-1",
            type=SynapseType.STORED_BY,
            metadata={"actor": "claude"},
        )
        storage.get_synapses = AsyncMock(return_value=[source_syn, stored_syn])
        source_obj = MagicMock()
        source_obj.name = "Doc"
        source_obj.source_type = MagicMock(value="document")
        storage.get_source = AsyncMock(return_value=source_obj)
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._provenance({"action": "trace", "neuron_id": "n-1"})
        assert result["has_source"] is True
        assert len(result["provenance"]) == 2

    @pytest.mark.asyncio
    async def test_verify_creates_synapse(self) -> None:
        handler = self._make_handler()
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_neuron = AsyncMock(return_value=MagicMock())
        storage.add_synapse = AsyncMock()
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._provenance(
            {"action": "verify", "neuron_id": "n-1", "actor": "admin"}
        )
        assert result["success"] is True
        assert result["action"] == "verified_at"
        storage.add_synapse.assert_called_once()
        syn_arg = storage.add_synapse.call_args[0][0]
        assert syn_arg.type == SynapseType.VERIFIED_AT
        assert syn_arg.metadata["actor"] == "admin"

    @pytest.mark.asyncio
    async def test_approve_creates_synapse(self) -> None:
        handler = self._make_handler()
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_neuron = AsyncMock(return_value=MagicMock())
        storage.add_synapse = AsyncMock()
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._provenance(
            {"action": "approve", "neuron_id": "n-1", "actor": "mgr"}
        )
        assert result["success"] is True
        assert result["action"] == "approved_by"

    @pytest.mark.asyncio
    async def test_missing_action(self) -> None:
        handler = self._make_handler()
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._provenance({"neuron_id": "n-1"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_neuron_id(self) -> None:
        handler = self._make_handler()
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._provenance({"action": "trace"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_neuron_not_found(self) -> None:
        handler = self._make_handler()
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_neuron = AsyncMock(return_value=None)
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._provenance({"action": "trace", "neuron_id": "bad-id"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self) -> None:
        handler = self._make_handler()
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_neuron = AsyncMock(return_value=MagicMock())
        handler.get_storage = AsyncMock(return_value=storage)

        result = await handler._provenance({"action": "bogus", "neuron_id": "n-1"})
        assert "error" in result


# ──────────────────── Backward Compatibility ────────────────────


class TestBackwardCompat:
    """Neurons without source/audit should still work."""

    def test_synapse_type_source_of_still_exists(self) -> None:
        """SOURCE_OF from Phase 2 still works."""
        assert SynapseType.SOURCE_OF.value == "source_of"
