"""Tests for the background version check handler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.mcp.version_check_handler import (
    VersionCheckHandler,
    VersionInfo,
    _is_newer,
    _parse_version,
)
from neural_memory.unified_config import MaintenanceConfig
from neural_memory.utils.timeutils import utcnow


def _make_cfg(
    *,
    enabled: bool = True,
    version_check_enabled: bool = True,
    version_check_interval_hours: int = 24,
) -> MagicMock:
    """Create a mock config with MaintenanceConfig."""
    cfg = MagicMock()
    cfg.maintenance = MaintenanceConfig(
        enabled=enabled,
        version_check_enabled=version_check_enabled,
        version_check_interval_hours=version_check_interval_hours,
    )
    return cfg


def _make_handler(cfg: MagicMock | None = None) -> VersionCheckHandler:
    """Create a minimal handler with mocked deps."""
    handler = VersionCheckHandler()
    handler.config = cfg or _make_cfg()  # type: ignore[attr-defined]
    return handler


class TestParseVersion:
    """Tests for _parse_version."""

    def test_simple_version(self) -> None:
        assert _parse_version("2.4.0") == (2, 4, 0)

    def test_two_part_version(self) -> None:
        assert _parse_version("1.0") == (1, 0)

    def test_prerelease_stripped(self) -> None:
        assert _parse_version("2.5.0a1") == (2, 5, 0)
        assert _parse_version("2.5.0b2") == (2, 5, 0)
        assert _parse_version("2.5.0rc1") == (2, 5, 0)
        assert _parse_version("2.5.0dev3") == (2, 5, 0)

    def test_single_part(self) -> None:
        assert _parse_version("3") == (3,)


class TestIsNewer:
    """Tests for _is_newer."""

    def test_newer_patch(self) -> None:
        assert _is_newer("2.4.1", "2.4.0") is True

    def test_newer_minor(self) -> None:
        assert _is_newer("2.5.0", "2.4.0") is True

    def test_newer_major(self) -> None:
        assert _is_newer("3.0.0", "2.4.0") is True

    def test_same_version(self) -> None:
        assert _is_newer("2.4.0", "2.4.0") is False

    def test_older_version(self) -> None:
        assert _is_newer("2.3.0", "2.4.0") is False

    def test_prerelease_not_newer(self) -> None:
        # 2.4.0a1 strips to 2.4.0 which is not newer than 2.4.0
        assert _is_newer("2.4.0a1", "2.4.0") is False


class TestMaybeStartVersionCheck:
    """Tests for maybe_start_version_check."""

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self) -> None:
        handler = _make_handler(cfg=_make_cfg(enabled=False))
        result = await handler.maybe_start_version_check()
        assert result is None

    @pytest.mark.asyncio
    async def test_version_check_disabled_returns_none(self) -> None:
        handler = _make_handler(cfg=_make_cfg(version_check_enabled=False))
        result = await handler.maybe_start_version_check()
        assert result is None

    @pytest.mark.asyncio
    async def test_starts_task(self) -> None:
        handler = _make_handler()
        task = await handler.maybe_start_version_check()
        assert task is not None
        assert handler._version_check_task is task
        assert not task.done()

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_prevents_duplicate_tasks(self) -> None:
        handler = _make_handler()
        handler._version_check_task = asyncio.create_task(asyncio.sleep(10))

        result = await handler.maybe_start_version_check()
        assert result is handler._version_check_task

        handler._version_check_task.cancel()
        try:
            await handler._version_check_task
        except asyncio.CancelledError:
            pass


class TestCheckLatestVersion:
    """Tests for _check_latest_version."""

    @pytest.mark.asyncio
    async def test_caches_version_info(self) -> None:
        handler = _make_handler()

        with patch(
            "neural_memory.mcp.version_check_handler._fetch_latest_version",
            new_callable=AsyncMock,
            return_value="99.0.0",
        ):
            await handler._check_latest_version()

        assert handler._version_info is not None
        assert handler._version_info.latest == "99.0.0"
        assert handler._version_info.update_available is True

    @pytest.mark.asyncio
    async def test_no_update_same_version(self) -> None:
        handler = _make_handler()

        with (
            patch(
                "neural_memory.mcp.version_check_handler._fetch_latest_version",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("neural_memory.mcp.version_check_handler.__version__", "2.4.0"),
        ):
            mock_fetch.return_value = "2.4.0"
            await handler._check_latest_version()

        assert handler._version_info is not None
        assert handler._version_info.update_available is False

    @pytest.mark.asyncio
    async def test_handles_fetch_failure(self) -> None:
        handler = _make_handler()

        with patch(
            "neural_memory.mcp.version_check_handler._fetch_latest_version",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await handler._check_latest_version()

        assert handler._version_info is None

    @pytest.mark.asyncio
    async def test_handles_exception(self) -> None:
        handler = _make_handler()

        with patch(
            "neural_memory.mcp.version_check_handler._fetch_latest_version",
            new_callable=AsyncMock,
            side_effect=RuntimeError("network error"),
        ):
            await handler._check_latest_version()

        assert handler._version_info is None


class TestGetUpdateHint:
    """Tests for get_update_hint."""

    def test_no_info_returns_none(self) -> None:
        handler = _make_handler()
        assert handler.get_update_hint() is None

    def test_no_update_returns_none(self) -> None:
        handler = _make_handler()
        handler._version_info = VersionInfo(
            current="2.4.0",
            latest="2.4.0",
            update_available=False,
            checked_at=utcnow(),
        )
        assert handler.get_update_hint() is None

    def test_update_available_returns_hint(self) -> None:
        handler = _make_handler()
        handler._version_info = VersionInfo(
            current="2.4.0",
            latest="2.5.0",
            update_available=True,
            checked_at=utcnow(),
        )
        hint = handler.get_update_hint()
        assert hint is not None
        assert "2.5.0" in hint["message"]
        assert "2.4.0" in hint["message"]
        assert hint["current_version"] == "2.4.0"
        assert hint["latest_version"] == "2.5.0"
        assert "pip install --upgrade" in hint["message"]


class TestCancelVersionCheck:
    """Tests for cancel_version_check."""

    @pytest.mark.asyncio
    async def test_cancels_running_task(self) -> None:
        handler = _make_handler()
        handler._version_check_task = asyncio.create_task(asyncio.sleep(10))

        handler.cancel_version_check()

        try:
            await handler._version_check_task
        except asyncio.CancelledError:
            pass

        assert handler._version_check_task.cancelled() or handler._version_check_task.done()

    def test_noop_when_no_task(self) -> None:
        handler = _make_handler()
        handler._version_check_task = None
        handler.cancel_version_check()

    def test_noop_when_task_done(self) -> None:
        handler = _make_handler()
        handler._version_check_task = MagicMock()
        handler._version_check_task.done.return_value = True
        handler.cancel_version_check()


class TestVersionCheckLoop:
    """Tests for _version_check_loop."""

    @pytest.mark.asyncio
    async def test_loop_calls_check(self) -> None:
        """Loop calls _check_latest_version after initial delay."""
        handler = _make_handler(cfg=_make_cfg(version_check_interval_hours=24))
        cfg = handler.config.maintenance  # type: ignore[attr-defined]

        call_count = 0

        async def fake_sleep(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            # First call is 30s startup delay, second is the interval sleep
            if call_count > 1:
                raise asyncio.CancelledError

        with patch(
            "neural_memory.mcp.version_check_handler.asyncio.sleep",
            fake_sleep,
        ):
            with patch.object(
                handler,
                "_check_latest_version",
                new_callable=AsyncMock,
            ) as mock_check:
                try:
                    await handler._version_check_loop(cfg)
                except asyncio.CancelledError:
                    pass

                mock_check.assert_called_once()


class TestVersionCheckConfig:
    """Tests for config fields roundtrip."""

    def test_defaults(self) -> None:
        cfg = MaintenanceConfig()
        assert cfg.version_check_enabled is True
        assert cfg.version_check_interval_hours == 24

    def test_roundtrip(self) -> None:
        cfg = MaintenanceConfig(
            version_check_enabled=False,
            version_check_interval_hours=12,
        )
        data = cfg.to_dict()
        restored = MaintenanceConfig.from_dict(data)

        assert restored.version_check_enabled is False
        assert restored.version_check_interval_hours == 12

    def test_from_dict_defaults(self) -> None:
        restored = MaintenanceConfig.from_dict({})
        assert restored.version_check_enabled is True
        assert restored.version_check_interval_hours == 24
