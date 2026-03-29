"""Tests for A6 Phase 2: Tiered Memory — MCP tool wiring.

Covers:
1. Schema: tier property in nmem_remember + nmem_edit, boundary in type enum
2. Remember handler: tier param flows to TypedMemory.create()
3. Edit handler: tier editing via dc_replace
4. Pin handler: auto-promotes tier to HOT
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.memory_types import (
    MemoryTier,
    MemoryType,
    TypedMemory,
)

# ── Schema tests ─────────────────────────────────────────


class TestTierInSchemas:
    """Verify tier and boundary are exposed in MCP tool schemas."""

    def _get_schema(self, tool_name: str) -> dict:
        from neural_memory.mcp.tool_schemas import get_tool_schemas

        for schema in get_tool_schemas():
            if schema["name"] == tool_name:
                return schema["inputSchema"]
        raise ValueError(f"Tool {tool_name} not found in schemas")

    def test_remember_schema_has_tier(self) -> None:
        schema = self._get_schema("nmem_remember")
        props = schema["properties"]
        assert "tier" in props
        assert props["tier"]["enum"] == ["hot", "warm", "cold"]

    def test_remember_schema_has_boundary_type(self) -> None:
        schema = self._get_schema("nmem_remember")
        type_enum = schema["properties"]["type"]["enum"]
        assert "boundary" in type_enum

    def test_edit_schema_has_tier(self) -> None:
        schema = self._get_schema("nmem_edit")
        props = schema["properties"]
        assert "tier" in props
        assert props["tier"]["enum"] == ["hot", "warm", "cold"]

    def test_edit_schema_has_boundary_type(self) -> None:
        schema = self._get_schema("nmem_edit")
        type_enum = schema["properties"]["type"]["enum"]
        assert "boundary" in type_enum


# ── Remember handler tier wiring ─────────────────────────


class TestRememberTierWiring:
    """Verify tier param in TypedMemory.create() via remember handler."""

    def test_typed_memory_create_with_tier_hot(self) -> None:
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
            tier="hot",
        )
        assert tm.tier == MemoryTier.HOT

    def test_typed_memory_create_with_tier_cold(self) -> None:
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
            tier="cold",
        )
        assert tm.tier == MemoryTier.COLD

    def test_typed_memory_create_default_warm(self) -> None:
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
        )
        assert tm.tier == MemoryTier.WARM

    def test_boundary_type_auto_promotes_to_hot(self) -> None:
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.BOUNDARY,
        )
        assert tm.tier == MemoryTier.HOT

    def test_boundary_type_ignores_explicit_cold(self) -> None:
        """Boundary always gets HOT, even if user requests cold."""
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.BOUNDARY,
            tier="cold",
        )
        assert tm.tier == MemoryTier.HOT


# ── Edit handler tier wiring ─────────────────────────────


class TestEditTierWiring:
    """Test _edit() handler tier editing."""

    def _make_handler(self, brain_id: str = "test-brain"):
        from neural_memory.mcp.tool_handlers import ToolHandler

        config = MagicMock()
        config.auto = MagicMock(enabled=False)
        config.encryption = MagicMock(enabled=False, auto_encrypt_sensitive=False)
        config.safety = MagicMock(auto_redact_min_severity=3)

        handler = ToolHandler.__new__(ToolHandler)
        handler.config = config
        handler.hooks = MagicMock()
        handler.hooks.emit = AsyncMock()

        storage = AsyncMock()
        storage.brain_id = brain_id
        storage._current_brain_id = brain_id
        storage.current_brain_id = brain_id

        handler.storage = storage
        handler.get_storage = AsyncMock(return_value=storage)
        handler._check_maintenance = AsyncMock(return_value=None)
        handler._get_maintenance_hint = MagicMock(return_value=None)
        handler.get_update_hint = MagicMock(return_value=None)
        handler._record_tool_action = AsyncMock()
        handler._check_onboarding = AsyncMock(return_value=None)
        handler._surface_pending_alerts = AsyncMock(return_value=None)
        handler._check_cross_language_hint = AsyncMock(return_value=None)

        return handler, storage

    @pytest.mark.asyncio
    async def test_edit_tier_to_cold(self) -> None:
        handler, storage = self._make_handler()
        mock_tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT, tier="warm")
        mock_fiber = MagicMock()
        mock_fiber.metadata = {}
        mock_fiber.anchor_neuron_id = "n1"

        storage.get_typed_memory = AsyncMock(return_value=mock_tm)
        storage.get_fiber = AsyncMock(return_value=mock_fiber)
        storage.update_typed_memory = AsyncMock()

        result = await handler._edit({"memory_id": "f1", "tier": "cold"})
        assert result["status"] == "edited"
        assert "tier: warm → cold" in result["changes"]
        storage.update_typed_memory.assert_called_once()
        updated_tm = storage.update_typed_memory.call_args[0][0]
        assert updated_tm.tier == "cold"

    @pytest.mark.asyncio
    async def test_edit_tier_to_hot(self) -> None:
        handler, storage = self._make_handler()
        mock_tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT, tier="warm")
        mock_fiber = MagicMock()
        mock_fiber.metadata = {}
        mock_fiber.anchor_neuron_id = "n1"

        storage.get_typed_memory = AsyncMock(return_value=mock_tm)
        storage.get_fiber = AsyncMock(return_value=mock_fiber)
        storage.update_typed_memory = AsyncMock()

        result = await handler._edit({"memory_id": "f1", "tier": "hot"})
        assert result["status"] == "edited"
        assert "tier: warm → hot" in result["changes"]

    @pytest.mark.asyncio
    async def test_edit_invalid_tier(self) -> None:
        handler, _ = self._make_handler()
        result = await handler._edit({"memory_id": "f1", "tier": "invalid"})
        assert "error" in result
        assert "Invalid tier" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_tier_only_no_other_fields(self) -> None:
        """Editing only tier (no type/content/priority) should work."""
        handler, storage = self._make_handler()
        mock_tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT, tier="warm")
        mock_fiber = MagicMock()
        mock_fiber.metadata = {}
        mock_fiber.anchor_neuron_id = "n1"

        storage.get_typed_memory = AsyncMock(return_value=mock_tm)
        storage.get_fiber = AsyncMock(return_value=mock_fiber)
        storage.update_typed_memory = AsyncMock()

        result = await handler._edit({"memory_id": "f1", "tier": "hot"})
        assert result["status"] == "edited"

    @pytest.mark.asyncio
    async def test_edit_type_to_boundary_auto_promotes_hot(self) -> None:
        """Changing type to boundary should auto-promote tier to HOT."""
        from neural_memory.core.fiber import Fiber

        handler, storage = self._make_handler()
        mock_tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT, tier="cold")
        mock_fiber = Fiber(
            id="f1",
            neuron_ids=set(),
            synapse_ids=set(),
            anchor_neuron_id="n1",
            metadata={},
        )

        storage.get_typed_memory = AsyncMock(return_value=mock_tm)
        storage.get_fiber = AsyncMock(return_value=mock_fiber)
        storage.update_typed_memory = AsyncMock()
        storage.update_fiber = AsyncMock()

        result = await handler._edit({"memory_id": "f1", "type": "boundary"})
        assert result["status"] == "edited"
        updated_tm = storage.update_typed_memory.call_args[0][0]
        assert updated_tm.tier == "hot"
        assert updated_tm.memory_type == MemoryType.BOUNDARY

    @pytest.mark.asyncio
    async def test_edit_no_fields_returns_error(self) -> None:
        """Must provide at least one field to edit."""
        handler, _ = self._make_handler()
        result = await handler._edit({"memory_id": "f1"})
        assert "error" in result
        assert "tier" in result["error"]  # error message mentions tier now


# ── Pin handler tier promotion ───────────────────────────


class TestPinTierPromotion:
    """Test _pin() auto-promotes tier to HOT."""

    def _make_handler(self, brain_id: str = "test-brain"):
        from neural_memory.mcp.train_handler import TrainHandler

        handler = TrainHandler.__new__(TrainHandler)

        storage = AsyncMock()
        storage.brain_id = brain_id
        storage._current_brain_id = brain_id
        storage.current_brain_id = brain_id
        storage.pin_fibers = AsyncMock(return_value=2)

        handler.get_storage = AsyncMock(return_value=storage)

        return handler, storage

    @pytest.mark.asyncio
    async def test_pin_promotes_to_hot(self) -> None:
        handler, storage = self._make_handler()
        mock_tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT, tier="warm")
        storage.get_typed_memory = AsyncMock(return_value=mock_tm)
        storage.update_typed_memory = AsyncMock()

        result = await handler._pin({"fiber_ids": ["f1"], "action": "pin"})
        assert result["updated"] == 2
        assert result.get("tier_promoted") == 1
        storage.update_typed_memory.assert_called_once()
        updated_tm = storage.update_typed_memory.call_args[0][0]
        assert updated_tm.tier == MemoryTier.HOT

    @pytest.mark.asyncio
    async def test_pin_already_hot_no_promotion(self) -> None:
        handler, storage = self._make_handler()
        mock_tm = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT, tier="hot")
        storage.get_typed_memory = AsyncMock(return_value=mock_tm)
        storage.update_typed_memory = AsyncMock()

        result = await handler._pin({"fiber_ids": ["f1"], "action": "pin"})
        assert result["updated"] == 2
        assert "tier_promoted" not in result
        storage.update_typed_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_unpin_does_not_promote(self) -> None:
        handler, storage = self._make_handler()
        storage.get_typed_memory = AsyncMock()
        storage.update_typed_memory = AsyncMock()

        result = await handler._pin({"fiber_ids": ["f1"], "action": "unpin"})
        assert result["updated"] == 2
        assert "tier_promoted" not in result
        storage.get_typed_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_pin_no_typed_memory_skips_promotion(self) -> None:
        """Fiber without typed_memory — pin succeeds, no promotion."""
        handler, storage = self._make_handler()
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.update_typed_memory = AsyncMock()

        result = await handler._pin({"fiber_ids": ["f1"], "action": "pin"})
        assert result["updated"] == 2
        assert "tier_promoted" not in result
        storage.update_typed_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_pin_multiple_fibers_promotes_all(self) -> None:
        handler, storage = self._make_handler()
        tm1 = TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT, tier="warm")
        tm2 = TypedMemory.create(fiber_id="f2", memory_type=MemoryType.DECISION, tier="cold")

        async def get_tm(fid):
            return {"f1": tm1, "f2": tm2}.get(fid)

        storage.get_typed_memory = AsyncMock(side_effect=get_tm)
        storage.update_typed_memory = AsyncMock()

        result = await handler._pin({"fiber_ids": ["f1", "f2"], "action": "pin"})
        assert result["tier_promoted"] == 2
        assert storage.update_typed_memory.call_count == 2
