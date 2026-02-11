"""Tests for Mem0 auto-sync handler."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.integration.models import ImportResult, SyncState
from neural_memory.mcp.mem0_sync_handler import Mem0SyncHandler
from neural_memory.unified_config import Mem0SyncConfig, UnifiedConfig

# ========== Helpers ==========

BRAIN_ID = "test-brain"


def _make_storage() -> MagicMock:
    """Create a mock storage with sync state methods."""
    storage = MagicMock()
    storage._current_brain_id = BRAIN_ID
    storage.disable_auto_save = MagicMock()
    storage.enable_auto_save = MagicMock()
    storage.batch_save = AsyncMock()

    brain = MagicMock()
    brain.id = BRAIN_ID
    brain.config = MagicMock()
    storage.get_brain = AsyncMock(return_value=brain)
    storage.get_sync_state = AsyncMock(return_value=None)
    storage.save_sync_state = AsyncMock()
    return storage


class _FakeServer(Mem0SyncHandler):
    """Minimal server stub providing get_storage() and config."""

    def __init__(
        self,
        storage: MagicMock,
        mem0_cfg: Mem0SyncConfig | None = None,
    ) -> None:
        self._storage = storage
        self.config = UnifiedConfig(
            mem0_sync=mem0_cfg or Mem0SyncConfig(),
        )

    async def get_storage(self) -> MagicMock:
        return self._storage


# ========== Mem0SyncConfig tests ==========


class TestMem0SyncConfig:
    def test_defaults(self) -> None:
        cfg = Mem0SyncConfig()
        assert cfg.enabled is True
        assert cfg.self_hosted is False
        assert cfg.user_id == ""
        assert cfg.agent_id == ""
        assert cfg.cooldown_minutes == 60
        assert cfg.sync_on_startup is True
        assert cfg.limit is None

    def test_from_dict(self) -> None:
        cfg = Mem0SyncConfig.from_dict(
            {"enabled": False, "self_hosted": True, "user_id": "alice", "limit": 100}
        )
        assert cfg.enabled is False
        assert cfg.self_hosted is True
        assert cfg.user_id == "alice"
        assert cfg.limit == 100

    def test_to_dict(self) -> None:
        cfg = Mem0SyncConfig(user_id="bob", limit=50)
        d = cfg.to_dict()
        assert d["user_id"] == "bob"
        assert d["limit"] == 50
        assert d["enabled"] is True

    def test_to_dict_no_limit(self) -> None:
        cfg = Mem0SyncConfig()
        d = cfg.to_dict()
        assert "limit" not in d

    def test_frozen(self) -> None:
        cfg = Mem0SyncConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]


# ========== maybe_start_mem0_sync tests ==========


class TestMaybeStartMem0Sync:
    @pytest.mark.asyncio
    async def test_disabled_returns_none(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage, Mem0SyncConfig(enabled=False))
        result = await server.maybe_start_mem0_sync()
        assert result is None

    @pytest.mark.asyncio
    async def test_sync_on_startup_false_returns_none(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage, Mem0SyncConfig(sync_on_startup=False))
        result = await server.maybe_start_mem0_sync()
        assert result is None

    @pytest.mark.asyncio
    async def test_no_api_key_no_self_hosted_returns_none(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage, Mem0SyncConfig())
        with patch.dict("os.environ", {}, clear=True):
            result = await server.maybe_start_mem0_sync()
        assert result is None

    @pytest.mark.asyncio
    async def test_api_key_starts_task(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage, Mem0SyncConfig())

        with patch.dict("os.environ", {"MEM0_API_KEY": "test-key"}):
            with patch.object(server, "_run_mem0_sync", new_callable=AsyncMock) as mock_run:
                task = await server.maybe_start_mem0_sync()
                assert task is not None
                # Let task run
                await task
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_self_hosted_starts_task(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage, Mem0SyncConfig(self_hosted=True))

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(server, "_run_mem0_sync", new_callable=AsyncMock) as mock_run:
                task = await server.maybe_start_mem0_sync()
                assert task is not None
                await task
                mock_run.assert_called_once()


# ========== _run_mem0_sync tests ==========


class TestRunMem0Sync:
    @pytest.mark.asyncio
    async def test_no_brain_skips(self) -> None:
        storage = _make_storage()
        storage.get_brain = AsyncMock(return_value=None)
        server = _FakeServer(storage)

        cfg = Mem0SyncConfig()
        await server._run_mem0_sync(True, cfg)
        # Should not raise, just log and return

    @pytest.mark.asyncio
    async def test_cooldown_active_skips(self) -> None:
        storage = _make_storage()
        # Last sync was 10 minutes ago, cooldown is 60 minutes
        recent_state = SyncState(
            source_system="mem0",
            source_collection="default",
            last_sync_at=datetime.now(UTC) - timedelta(minutes=10),
            records_imported=5,
        )
        storage.get_sync_state = AsyncMock(return_value=recent_state)
        server = _FakeServer(storage, Mem0SyncConfig(cooldown_minutes=60))

        with patch("neural_memory.integration.sync_engine.SyncEngine") as mock_engine_cls:
            await server._run_mem0_sync(True, Mem0SyncConfig(cooldown_minutes=60))
            mock_engine_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_cooldown_expired_runs_sync(self) -> None:
        storage = _make_storage()
        old_state = SyncState(
            source_system="mem0",
            source_collection="default",
            last_sync_at=datetime.now(UTC) - timedelta(minutes=120),
            records_imported=5,
        )
        storage.get_sync_state = AsyncMock(return_value=old_state)
        server = _FakeServer(storage, Mem0SyncConfig(cooldown_minutes=60))

        import_result = ImportResult(
            source_system="mem0",
            source_collection="default",
            records_imported=3,
            records_skipped=1,
            duration_seconds=1.5,
        )
        updated_state = old_state.with_update(last_sync_at=datetime.now(UTC), records_imported=8)

        mock_engine = MagicMock()
        mock_engine.sync = AsyncMock(return_value=(import_result, updated_state))

        with patch(
            "neural_memory.integration.sync_engine.SyncEngine",
            return_value=mock_engine,
        ):
            with patch.object(server, "_create_mem0_adapter", return_value=MagicMock()):
                await server._run_mem0_sync(True, Mem0SyncConfig(cooldown_minutes=60))

        mock_engine.sync.assert_called_once()
        storage.save_sync_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_import_error_handled(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)

        mock_engine = MagicMock()
        mock_engine.sync = AsyncMock(side_effect=RuntimeError("Connection failed"))

        with patch(
            "neural_memory.integration.sync_engine.SyncEngine",
            return_value=mock_engine,
        ):
            with patch.object(server, "_create_mem0_adapter", return_value=MagicMock()):
                # Should not raise
                await server._run_mem0_sync(True, Mem0SyncConfig())

    @pytest.mark.asyncio
    async def test_missing_mem0_package_handled(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)

        with patch.object(
            server,
            "_create_mem0_adapter",
            side_effect=ImportError("No module named 'mem0'"),
        ):
            # Should not raise
            await server._run_mem0_sync(True, Mem0SyncConfig())


# ========== _create_mem0_adapter tests ==========


class TestCreateMem0Adapter:
    def test_platform_adapter(self) -> None:
        with patch.dict("os.environ", {"MEM0_API_KEY": "test-key"}):
            with patch("neural_memory.integration.adapters.mem0_adapter.Mem0Adapter") as mock_cls:
                cfg = Mem0SyncConfig(user_id="alice")
                Mem0SyncHandler._create_mem0_adapter(True, False, cfg)
                mock_cls.assert_called_once_with(api_key="test-key", user_id="alice")

    def test_self_hosted_adapter(self) -> None:
        with patch(
            "neural_memory.integration.adapters.mem0_adapter.Mem0SelfHostedAdapter"
        ) as mock_cls:
            cfg = Mem0SyncConfig(user_id="bob", self_hosted=True)
            Mem0SyncHandler._create_mem0_adapter(False, True, cfg)
            mock_cls.assert_called_once_with(user_id="bob")

    def test_agent_id_passed(self) -> None:
        with patch.dict("os.environ", {"MEM0_API_KEY": "key"}):
            with patch("neural_memory.integration.adapters.mem0_adapter.Mem0Adapter") as mock_cls:
                cfg = Mem0SyncConfig(agent_id="agent-1")
                Mem0SyncHandler._create_mem0_adapter(True, False, cfg)
                mock_cls.assert_called_once_with(api_key="key", agent_id="agent-1")


# ========== cancel_mem0_sync tests ==========


class TestCancelMem0Sync:
    def test_cancel_no_task(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        server._mem0_sync_task = None
        server.cancel_mem0_sync()  # Should not raise

    @pytest.mark.asyncio
    async def test_cancel_running_task(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)

        async def long_task() -> None:
            await asyncio.sleep(100)

        server._mem0_sync_task = asyncio.create_task(long_task())
        server.cancel_mem0_sync()
        # Allow event loop to process the cancellation
        await asyncio.sleep(0)
        assert server._mem0_sync_task.cancelled()
