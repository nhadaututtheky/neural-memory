"""Tests for the scheduled consolidation handler."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.mcp.scheduled_consolidation_handler import (
    ScheduledConsolidationHandler,
)
from neural_memory.unified_config import MaintenanceConfig
from neural_memory.utils.timeutils import utcnow


def _make_cfg(
    *,
    enabled: bool = True,
    sched_enabled: bool = True,
    interval_hours: int = 24,
    strategies: tuple[str, ...] = ("prune", "merge", "enrich"),
) -> MagicMock:
    """Create a mock config with MaintenanceConfig."""
    cfg = MagicMock()
    cfg.maintenance = MaintenanceConfig(
        enabled=enabled,
        scheduled_consolidation_enabled=sched_enabled,
        scheduled_consolidation_interval_hours=interval_hours,
        scheduled_consolidation_strategies=strategies,
    )
    return cfg


def _make_handler(
    cfg: MagicMock | None = None,
    consolidation_error: bool = False,
) -> ScheduledConsolidationHandler:
    """Create a minimal handler with mocked deps."""
    handler = ScheduledConsolidationHandler()
    handler.config = cfg or _make_cfg()  # type: ignore[attr-defined]
    handler._last_consolidation_at = None  # type: ignore[attr-defined]

    mock_storage = AsyncMock()
    mock_storage._current_brain_id = "test-brain"
    handler.get_storage = AsyncMock(return_value=mock_storage)  # type: ignore[attr-defined]

    return handler


class TestMaybeStartScheduledConsolidation:
    """Tests for maybe_start_scheduled_consolidation."""

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self) -> None:
        """Returns None when maintenance disabled."""
        handler = _make_handler(cfg=_make_cfg(enabled=False))
        result = await handler.maybe_start_scheduled_consolidation()
        assert result is None

    @pytest.mark.asyncio
    async def test_sched_disabled_returns_none(self) -> None:
        """Returns None when scheduled_consolidation_enabled=False."""
        handler = _make_handler(cfg=_make_cfg(sched_enabled=False))
        result = await handler.maybe_start_scheduled_consolidation()
        assert result is None

    @pytest.mark.asyncio
    async def test_starts_task(self) -> None:
        """Starts a background task when enabled."""
        handler = _make_handler()
        task = await handler.maybe_start_scheduled_consolidation()
        assert task is not None
        assert handler._scheduled_consolidation_task is task
        assert not task.done()

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_prevents_duplicate_tasks(self) -> None:
        """Does not start a second task if one is running."""
        handler = _make_handler()
        handler._scheduled_consolidation_task = asyncio.create_task(asyncio.sleep(10))

        result = await handler.maybe_start_scheduled_consolidation()
        assert result is handler._scheduled_consolidation_task

        handler._scheduled_consolidation_task.cancel()
        try:
            await handler._scheduled_consolidation_task
        except asyncio.CancelledError:
            pass


class TestScheduledConsolidationLoop:
    """Tests for _scheduled_consolidation_loop behavior."""

    @pytest.mark.asyncio
    async def test_skips_if_recent_consolidation(self) -> None:
        """Skips run if _last_consolidation_at is recent."""
        handler = _make_handler(cfg=_make_cfg(interval_hours=24))
        # Set last consolidation to 1 hour ago (well within half-interval of 12h)
        handler._last_consolidation_at = utcnow() - timedelta(hours=1)  # type: ignore[attr-defined]

        cfg = handler.config.maintenance  # type: ignore[attr-defined]

        # Patch sleep to avoid waiting, run loop once then cancel
        call_count = 0

        async def fake_sleep(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with patch("neural_memory.mcp.scheduled_consolidation_handler.asyncio.sleep", fake_sleep):
            with patch.object(handler, "_run_scheduled_consolidation") as mock_run:
                try:
                    await handler._scheduled_consolidation_loop(cfg)
                except asyncio.CancelledError:
                    pass

                mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_when_interval_elapsed(self) -> None:
        """Runs consolidation when enough time has passed."""
        handler = _make_handler(cfg=_make_cfg(interval_hours=24))
        # Set last consolidation to 13 hours ago (past half-interval of 12h)
        handler._last_consolidation_at = utcnow() - timedelta(hours=13)  # type: ignore[attr-defined]

        cfg = handler.config.maintenance  # type: ignore[attr-defined]

        call_count = 0

        async def fake_sleep(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with patch("neural_memory.mcp.scheduled_consolidation_handler.asyncio.sleep", fake_sleep):
            with patch.object(
                handler, "_run_scheduled_consolidation", new_callable=AsyncMock
            ) as mock_run:
                try:
                    await handler._scheduled_consolidation_loop(cfg)
                except asyncio.CancelledError:
                    pass

                mock_run.assert_called_once_with(cfg)

    @pytest.mark.asyncio
    async def test_runs_on_first_call_no_prior(self) -> None:
        """Runs consolidation when _last_consolidation_at is None."""
        handler = _make_handler()
        handler._last_consolidation_at = None  # type: ignore[attr-defined]

        cfg = handler.config.maintenance  # type: ignore[attr-defined]

        call_count = 0

        async def fake_sleep(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with patch("neural_memory.mcp.scheduled_consolidation_handler.asyncio.sleep", fake_sleep):
            with patch.object(
                handler, "_run_scheduled_consolidation", new_callable=AsyncMock
            ) as mock_run:
                try:
                    await handler._scheduled_consolidation_loop(cfg)
                except asyncio.CancelledError:
                    pass

                mock_run.assert_called_once_with(cfg)


class TestRunScheduledConsolidation:
    """Tests for _run_scheduled_consolidation."""

    @pytest.mark.asyncio
    async def test_calls_run_with_delta(self) -> None:
        """Runs consolidation with configured strategies."""
        handler = _make_handler(cfg=_make_cfg(strategies=("prune", "enrich")))

        mock_delta = MagicMock()
        mock_delta.report.summary.return_value = "OK"
        mock_delta.purity_delta = 1.5

        with patch(
            "neural_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
            return_value=mock_delta,
        ) as mock_rwd:
            cfg = handler.config.maintenance  # type: ignore[attr-defined]
            await handler._run_scheduled_consolidation(cfg)

            mock_rwd.assert_called_once()
            call_args = mock_rwd.call_args
            # Verify strategies passed
            strategies = call_args[1].get("strategies") or call_args[0][2]
            strategy_values = [s.value for s in strategies]
            assert "prune" in strategy_values
            assert "enrich" in strategy_values

    @pytest.mark.asyncio
    async def test_updates_last_consolidation_at(self) -> None:
        """Sets _last_consolidation_at after run."""
        handler = _make_handler()
        assert handler._last_consolidation_at is None  # type: ignore[attr-defined]

        mock_delta = MagicMock()
        mock_delta.report.summary.return_value = "OK"
        mock_delta.purity_delta = 0.0

        with patch(
            "neural_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
            return_value=mock_delta,
        ):
            cfg = handler.config.maintenance  # type: ignore[attr-defined]
            await handler._run_scheduled_consolidation(cfg)

        assert handler._last_consolidation_at is not None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self) -> None:
        """Logs error but does not raise on consolidation failure."""
        handler = _make_handler()
        handler.get_storage = AsyncMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("db error")
        )

        cfg = handler.config.maintenance  # type: ignore[attr-defined]
        # Should not raise
        await handler._run_scheduled_consolidation(cfg)


class TestCancelScheduledConsolidation:
    """Tests for cancel_scheduled_consolidation."""

    @pytest.mark.asyncio
    async def test_cancels_running_task(self) -> None:
        """Cancels the background task."""
        handler = _make_handler()
        handler._scheduled_consolidation_task = asyncio.create_task(asyncio.sleep(10))

        handler.cancel_scheduled_consolidation()

        try:
            await handler._scheduled_consolidation_task
        except asyncio.CancelledError:
            pass

        assert (
            handler._scheduled_consolidation_task.cancelled()
            or handler._scheduled_consolidation_task.done()
        )

    def test_noop_when_no_task(self) -> None:
        """No error when no task is running."""
        handler = _make_handler()
        handler._scheduled_consolidation_task = None
        handler.cancel_scheduled_consolidation()  # Should not raise

    def test_noop_when_task_done(self) -> None:
        """No error when task is already done."""
        handler = _make_handler()
        handler._scheduled_consolidation_task = MagicMock()
        handler._scheduled_consolidation_task.done.return_value = True
        handler.cancel_scheduled_consolidation()  # Should not raise


class TestScheduledConsolidationConfig:
    """Tests for config fields roundtrip."""

    def test_defaults(self) -> None:
        """Default values are correct."""
        cfg = MaintenanceConfig()
        assert cfg.scheduled_consolidation_enabled is True
        assert cfg.scheduled_consolidation_interval_hours == 24
        assert cfg.scheduled_consolidation_strategies == (
            "prune",
            "merge",
            "enrich",
        )

    def test_roundtrip(self) -> None:
        """to_dict/from_dict preserves scheduled consolidation fields."""
        cfg = MaintenanceConfig(
            scheduled_consolidation_enabled=False,
            scheduled_consolidation_interval_hours=6,
            scheduled_consolidation_strategies=("prune",),
        )
        data = cfg.to_dict()
        restored = MaintenanceConfig.from_dict(data)

        assert restored.scheduled_consolidation_enabled is False
        assert restored.scheduled_consolidation_interval_hours == 6
        assert restored.scheduled_consolidation_strategies == ("prune",)

    def test_from_dict_defaults(self) -> None:
        """Missing keys use defaults."""
        restored = MaintenanceConfig.from_dict({})
        assert restored.scheduled_consolidation_enabled is True
        assert restored.scheduled_consolidation_interval_hours == 24
        assert restored.scheduled_consolidation_strategies == (
            "prune",
            "merge",
            "enrich",
        )
