"""Tests for the background expiry cleanup handler."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.core.memory_types import MemoryType, Priority, Provenance, TypedMemory
from neural_memory.mcp.expiry_cleanup_handler import ExpiryCleanupHandler
from neural_memory.unified_config import MaintenanceConfig
from neural_memory.utils.timeutils import utcnow


def _make_cfg(
    *,
    enabled: bool = True,
    cleanup_enabled: bool = True,
    interval_hours: int = 12,
    max_per_run: int = 100,
) -> MagicMock:
    """Create a mock config with MaintenanceConfig."""
    cfg = MagicMock()
    cfg.maintenance = MaintenanceConfig(
        enabled=enabled,
        expiry_cleanup_enabled=cleanup_enabled,
        expiry_cleanup_interval_hours=interval_hours,
        expiry_cleanup_max_per_run=max_per_run,
    )
    return cfg


def _make_expired_memory(fiber_id: str) -> TypedMemory:
    """Create an expired TypedMemory."""
    now = utcnow()
    return TypedMemory(
        fiber_id=fiber_id,
        memory_type=MemoryType.TODO,
        priority=Priority.from_int(5),
        provenance=Provenance(source="test"),
        expires_at=now - timedelta(days=1),
        created_at=now - timedelta(days=10),
    )


def _make_handler(
    cfg: MagicMock | None = None,
    expired: list[TypedMemory] | None = None,
    storage_error: bool = False,
) -> ExpiryCleanupHandler:
    """Create a minimal ExpiryCleanupHandler with mocked deps."""
    handler = ExpiryCleanupHandler()
    handler.config = cfg or _make_cfg()  # type: ignore[attr-defined]
    handler.hooks = AsyncMock()  # type: ignore[attr-defined]

    mock_storage = AsyncMock()
    if storage_error:
        mock_storage.get_expired_memories = AsyncMock(side_effect=RuntimeError("db err"))
    else:
        mock_storage.get_expired_memories = AsyncMock(return_value=expired or [])
    mock_storage.delete_typed_memory = AsyncMock(return_value=True)
    mock_storage.delete_fiber = AsyncMock(return_value=True)

    handler.get_storage = AsyncMock(return_value=mock_storage)  # type: ignore[attr-defined]
    return handler


class TestMaybeRunExpiryCleanup:
    """Tests for _maybe_run_expiry_cleanup interval checks."""

    @pytest.mark.asyncio
    async def test_disabled_returns_zero(self) -> None:
        """Returns 0 when maintenance disabled."""
        handler = _make_handler(cfg=_make_cfg(enabled=False))
        result = await handler._maybe_run_expiry_cleanup()
        assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_disabled_returns_zero(self) -> None:
        """Returns 0 when expiry_cleanup_enabled=False."""
        handler = _make_handler(cfg=_make_cfg(cleanup_enabled=False))
        result = await handler._maybe_run_expiry_cleanup()
        assert result == 0

    @pytest.mark.asyncio
    async def test_respects_interval(self) -> None:
        """Does not launch task if interval not elapsed."""
        handler = _make_handler()
        handler._last_expiry_cleanup_at = utcnow()

        result = await handler._maybe_run_expiry_cleanup()
        assert result == 0
        assert handler._expiry_cleanup_task is None

    @pytest.mark.asyncio
    async def test_fires_after_interval(self) -> None:
        """Launches task when interval elapsed."""
        handler = _make_handler(expired=[_make_expired_memory("f-1")])
        handler._last_expiry_cleanup_at = utcnow() - timedelta(hours=13)

        result = await handler._maybe_run_expiry_cleanup()
        assert result == 0  # Always returns 0 (fire-and-forget)
        assert handler._expiry_cleanup_task is not None

        # Wait for background task
        await handler._expiry_cleanup_task

    @pytest.mark.asyncio
    async def test_first_call_fires(self) -> None:
        """First call (no previous timestamp) launches task."""
        handler = _make_handler(expired=[])
        assert handler._last_expiry_cleanup_at is None

        await handler._maybe_run_expiry_cleanup()
        assert handler._last_expiry_cleanup_at is not None

    @pytest.mark.asyncio
    async def test_prevents_duplicate_tasks(self) -> None:
        """Does not launch new task if one is still running."""
        handler = _make_handler()
        # Create a long-running fake task
        handler._expiry_cleanup_task = asyncio.create_task(asyncio.sleep(10))
        handler._last_expiry_cleanup_at = None  # Would normally trigger

        result = await handler._maybe_run_expiry_cleanup()
        assert result == 0

        handler._expiry_cleanup_task.cancel()
        try:
            await handler._expiry_cleanup_task
        except asyncio.CancelledError:
            pass


class TestRunExpiryCleanup:
    """Tests for _run_expiry_cleanup execution."""

    @pytest.mark.asyncio
    async def test_deletes_expired_memories(self) -> None:
        """Deletes typed_memory + fiber for each expired memory."""
        expired = [_make_expired_memory("f-1"), _make_expired_memory("f-2")]
        handler = _make_handler(expired=expired)
        cfg = handler.config.maintenance  # type: ignore[attr-defined]

        count = await handler._run_expiry_cleanup(cfg)

        assert count == 2
        storage = await handler.get_storage()
        assert storage.delete_typed_memory.call_count == 2
        assert storage.delete_fiber.call_count == 2

    @pytest.mark.asyncio
    async def test_fires_memory_expired_hook(self) -> None:
        """Fires MEMORY_EXPIRED hook for each deletion."""
        expired = [_make_expired_memory("f-1")]
        handler = _make_handler(expired=expired)
        cfg = handler.config.maintenance  # type: ignore[attr-defined]

        await handler._run_expiry_cleanup(cfg)

        handler.hooks.emit.assert_called_once()
        call_args = handler.hooks.emit.call_args
        assert call_args[0][1]["fiber_id"] == "f-1"
        assert call_args[0][1]["memory_type"] == "todo"

    @pytest.mark.asyncio
    async def test_empty_expired_list_returns_zero(self) -> None:
        """No expired memories = no deletions."""
        handler = _make_handler(expired=[])
        cfg = handler.config.maintenance  # type: ignore[attr-defined]

        count = await handler._run_expiry_cleanup(cfg)
        assert count == 0

    @pytest.mark.asyncio
    async def test_storage_error_returns_zero(self) -> None:
        """Storage error returns 0 gracefully."""
        handler = _make_handler(storage_error=True)
        cfg = handler.config.maintenance  # type: ignore[attr-defined]

        count = await handler._run_expiry_cleanup(cfg)
        assert count == 0

    @pytest.mark.asyncio
    async def test_handles_delete_failure(self) -> None:
        """One delete failure doesn't stop others."""
        expired = [_make_expired_memory("f-1"), _make_expired_memory("f-2")]
        handler = _make_handler(expired=expired)
        cfg = handler.config.maintenance  # type: ignore[attr-defined]

        storage = await handler.get_storage()
        # First delete_typed_memory fails, second succeeds
        storage.delete_typed_memory = AsyncMock(
            side_effect=[RuntimeError("fail"), True]
        )

        count = await handler._run_expiry_cleanup(cfg)
        assert count == 1  # Only second succeeded

    @pytest.mark.asyncio
    async def test_cancel_cleanup(self) -> None:
        """cancel_expiry_cleanup cancels running task."""
        handler = _make_handler()
        handler._expiry_cleanup_task = asyncio.create_task(asyncio.sleep(10))

        handler.cancel_expiry_cleanup()

        # Wait for cancellation to propagate
        try:
            await handler._expiry_cleanup_task
        except asyncio.CancelledError:
            pass

        assert handler._expiry_cleanup_task.cancelled() or handler._expiry_cleanup_task.done()


class TestExpiryCleanupConfig:
    """Tests for config roundtrip with new fields."""

    def test_defaults(self) -> None:
        """Default values are correct."""
        cfg = MaintenanceConfig()
        assert cfg.expiry_cleanup_enabled is True
        assert cfg.expiry_cleanup_interval_hours == 12
        assert cfg.expiry_cleanup_max_per_run == 100

    def test_roundtrip(self) -> None:
        """to_dict/from_dict preserves new fields."""
        cfg = MaintenanceConfig(
            expiry_cleanup_enabled=False,
            expiry_cleanup_interval_hours=6,
            expiry_cleanup_max_per_run=50,
        )
        data = cfg.to_dict()
        restored = MaintenanceConfig.from_dict(data)

        assert restored.expiry_cleanup_enabled is False
        assert restored.expiry_cleanup_interval_hours == 6
        assert restored.expiry_cleanup_max_per_run == 50

    def test_from_dict_defaults(self) -> None:
        """Missing keys use defaults."""
        restored = MaintenanceConfig.from_dict({})
        assert restored.expiry_cleanup_enabled is True
        assert restored.expiry_cleanup_interval_hours == 12
        assert restored.expiry_cleanup_max_per_run == 100
