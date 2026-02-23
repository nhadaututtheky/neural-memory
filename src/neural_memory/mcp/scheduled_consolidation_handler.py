"""Background scheduled consolidation handler for MCP server.

Runs consolidation on a fixed interval (default 24h) as a background
asyncio loop. Shares ``_last_consolidation_at`` with MaintenanceHandler
to prevent overlap with op-triggered auto-consolidation.

Starts via ``maybe_start_scheduled_consolidation()`` at server startup.
Cancelled via ``cancel_scheduled_consolidation()`` in server shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from neural_memory.utils.timeutils import utcnow
from neural_memory.mcp.tool_handlers import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import MaintenanceConfig, UnifiedConfig

logger = logging.getLogger(__name__)


class ScheduledConsolidationHandler:
    """Mixin: scheduled background consolidation for MCP server.

    Runs a periodic consolidation loop independent of health-pulse
    triggers. Shares ``_last_consolidation_at`` with
    ``MaintenanceHandler`` so the two do not overlap.
    """

    _scheduled_consolidation_task: asyncio.Task[None] | None = None

    if TYPE_CHECKING:
        config: UnifiedConfig
        _last_consolidation_at: datetime | None

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def maybe_start_scheduled_consolidation(
        self,
    ) -> asyncio.Task[None] | None:
        """Start the background consolidation loop if configured.

        Called once at server startup. Guards against double-start.
        Returns the background task or None.
        """
        if (
            self._scheduled_consolidation_task is not None
            and not self._scheduled_consolidation_task.done()
        ):
            return self._scheduled_consolidation_task

        cfg: MaintenanceConfig = self.config.maintenance
        if not cfg.enabled or not cfg.scheduled_consolidation_enabled:
            return None

        task = asyncio.create_task(self._scheduled_consolidation_loop(cfg))
        task.add_done_callback(_log_scheduled_exception)
        self._scheduled_consolidation_task = task
        logger.info(
            "Scheduled consolidation started: every %dh, strategies=%s",
            cfg.scheduled_consolidation_interval_hours,
            cfg.scheduled_consolidation_strategies,
        )
        return self._scheduled_consolidation_task

    async def _scheduled_consolidation_loop(self, cfg: MaintenanceConfig) -> None:
        """Background loop: sleep for interval, then consolidate.

        First run waits one full interval to avoid triggering on every
        server restart. Subsequent runs wait the full interval again.
        Checks ``_last_consolidation_at`` before running to prevent
        overlap with MaintenanceHandler's op-triggered consolidation.
        """
        interval_seconds = cfg.scheduled_consolidation_interval_hours * 3600

        while True:
            await asyncio.sleep(interval_seconds)

            # Skip if MaintenanceHandler ran consolidation recently
            now = utcnow()
            if self._last_consolidation_at is not None:
                elapsed = now - self._last_consolidation_at
                # Skip if less than half the interval has passed since last run
                half_interval = timedelta(
                    hours=cfg.scheduled_consolidation_interval_hours / 2,
                )
                if elapsed < half_interval:
                    logger.debug(
                        "Scheduled consolidation skipped: last run %.1fh ago",
                        elapsed.total_seconds() / 3600,
                    )
                    continue

            await self._run_scheduled_consolidation(cfg)

    async def _run_scheduled_consolidation(self, cfg: MaintenanceConfig) -> None:
        """Execute one scheduled consolidation run."""
        from neural_memory.engine.consolidation import ConsolidationStrategy
        from neural_memory.engine.consolidation_delta import run_with_delta

        try:
            storage = await self.get_storage()
            brain_id = _require_brain_id(storage)
            strategies = [ConsolidationStrategy(s) for s in cfg.scheduled_consolidation_strategies]

            self._last_consolidation_at = utcnow()

            delta = await run_with_delta(storage, brain_id, strategies=strategies)
            logger.info(
                "Scheduled consolidation complete (strategies=%s): %s | purity delta: %+.1f",
                cfg.scheduled_consolidation_strategies,
                delta.report.summary(),
                delta.purity_delta,
            )
        except Exception:
            logger.error("Scheduled consolidation failed", exc_info=True)

    def cancel_scheduled_consolidation(self) -> None:
        """Cancel the background consolidation task if running."""
        if (
            self._scheduled_consolidation_task is not None
            and not self._scheduled_consolidation_task.done()
        ):
            self._scheduled_consolidation_task.cancel()
            logger.debug("Scheduled consolidation task cancelled")


def _log_scheduled_exception(task: asyncio.Task[None]) -> None:
    """Log unhandled exceptions from the scheduled consolidation task."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Scheduled consolidation task raised unhandled exception: %s", exc)
