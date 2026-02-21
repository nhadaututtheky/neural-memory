"""Background expiry cleanup handler for MCP server.

Periodically scans for expired TypedMemory records and deletes both
the typed memory and its underlying fiber. Fires MEMORY_EXPIRED hooks
and logs all deletions for auditability.

Piggybacks on _check_maintenance() via _maybe_run_expiry_cleanup().
Runs as a fire-and-forget background task to avoid blocking tool responses.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.engine.hooks import HookRegistry
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import MaintenanceConfig, UnifiedConfig

logger = logging.getLogger(__name__)


class ExpiryCleanupHandler:
    """Mixin: background expiry cleanup for MCP server.

    Triggered from _check_maintenance() after each health pulse.
    Checks whether the cleanup interval has elapsed and, if so,
    launches a fire-and-forget background task.
    """

    _last_expiry_cleanup_at: datetime | None = None
    _expiry_cleanup_task: asyncio.Task[int] | None = None

    if TYPE_CHECKING:
        config: UnifiedConfig
        hooks: HookRegistry

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _maybe_run_expiry_cleanup(self) -> int:
        """Check if expiry cleanup is due and launch it if so.

        Called from _check_maintenance() after health pulse.
        Returns 0 immediately; actual work happens in background.
        """
        cfg: MaintenanceConfig = self.config.maintenance
        if not cfg.enabled or not cfg.expiry_cleanup_enabled:
            return 0

        now = utcnow()
        if self._last_expiry_cleanup_at is not None:
            interval = timedelta(hours=cfg.expiry_cleanup_interval_hours)
            if now - self._last_expiry_cleanup_at < interval:
                return 0

        # Skip if cleanup already running
        if self._expiry_cleanup_task is not None and not self._expiry_cleanup_task.done():
            return 0

        self._last_expiry_cleanup_at = now
        self._expiry_cleanup_task = asyncio.create_task(self._run_expiry_cleanup(cfg))
        self._expiry_cleanup_task.add_done_callback(_log_cleanup_exception)
        return 0

    async def _run_expiry_cleanup(self, cfg: MaintenanceConfig) -> int:
        """Execute one expiry cleanup run.

        Deletes up to cfg.expiry_cleanup_max_per_run expired memories.
        For each deletion: remove typed_memory, remove fiber, fire hook, log.
        """
        from neural_memory.engine.hooks import HookEvent

        try:
            storage = await self.get_storage()
        except Exception:
            logger.error("Expiry cleanup: get_storage failed", exc_info=True)
            return 0

        try:
            expired_memories = await storage.get_expired_memories(
                limit=cfg.expiry_cleanup_max_per_run,
            )
        except Exception:
            logger.error("Expiry cleanup: get_expired_memories failed", exc_info=True)
            return 0

        if not expired_memories:
            return 0

        deleted_count = 0

        for mem in expired_memories:
            try:
                await storage.delete_typed_memory(mem.fiber_id)
                await storage.delete_fiber(mem.fiber_id)
                deleted_count += 1

                await self.hooks.emit(
                    HookEvent.MEMORY_EXPIRED,
                    {
                        "fiber_id": mem.fiber_id,
                        "memory_type": mem.memory_type.value,
                        "priority": mem.priority.value,
                        "expired_at": mem.expires_at.isoformat() if mem.expires_at else None,
                    },
                )

                logger.info(
                    "Expired memory cleaned up: fiber_id=%s, type=%s, priority=%d",
                    mem.fiber_id,
                    mem.memory_type.value,
                    mem.priority.value,
                )
            except Exception:
                logger.error(
                    "Expiry cleanup: failed to delete fiber_id=%s",
                    mem.fiber_id,
                    exc_info=True,
                )

        if deleted_count > 0:
            logger.info(
                "Expiry cleanup complete: %d/%d expired memories deleted",
                deleted_count,
                len(expired_memories),
            )

        return deleted_count

    def cancel_expiry_cleanup(self) -> None:
        """Cancel any running cleanup task."""
        if self._expiry_cleanup_task is not None and not self._expiry_cleanup_task.done():
            self._expiry_cleanup_task.cancel()
            logger.debug("Expiry cleanup task cancelled")


def _log_cleanup_exception(task: asyncio.Task[int]) -> None:
    """Log unhandled exceptions from cleanup task."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Expiry cleanup task raised unhandled exception: %s", exc)
