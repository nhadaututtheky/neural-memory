"""Background auto-sync handler for Mem0 integration.

Detects Mem0 presence (via MEM0_API_KEY env var or self_hosted config)
and runs a background sync on MCP server startup. Respects cooldown
periods and persists sync state across restarts.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta

from neural_memory.utils.timeutils import utcnow
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.integration.adapters.mem0_adapter import _BaseMem0Adapter
    from neural_memory.unified_config import Mem0SyncConfig, UnifiedConfig

logger = logging.getLogger(__name__)


class Mem0SyncHandler:
    """Mixin: background auto-sync from Mem0 on MCP server startup."""

    _mem0_sync_task: asyncio.Task[None] | None = None

    async def maybe_start_mem0_sync(self) -> asyncio.Task[None] | None:
        """Check config and environment, start background sync if appropriate.

        Called once at server startup. Returns the background task or None.
        Guards against double-start by checking if a task is already running.

        Detection logic:
            MEM0_API_KEY set + not self_hosted -> Mem0Adapter (Platform)
            MEM0_API_KEY set + self_hosted     -> Mem0SelfHostedAdapter
            no API key     + self_hosted       -> Mem0SelfHostedAdapter
            no API key     + not self_hosted   -> skip (no sync)
        """
        # Guard: prevent duplicate background tasks
        if self._mem0_sync_task is not None and not self._mem0_sync_task.done():
            return self._mem0_sync_task

        config: UnifiedConfig = self.config  # type: ignore[attr-defined]
        cfg: Mem0SyncConfig = config.mem0_sync

        if not cfg.enabled or not cfg.sync_on_startup:
            return None

        has_api_key = bool(os.environ.get("MEM0_API_KEY"))

        if not has_api_key and not cfg.self_hosted:
            return None

        task = asyncio.create_task(self._run_mem0_sync(has_api_key, cfg))
        task.add_done_callback(_log_task_exception)
        self._mem0_sync_task = task
        return self._mem0_sync_task

    async def _run_mem0_sync(self, has_api_key: bool, cfg: Mem0SyncConfig) -> None:
        """Background task: sync from Mem0 with cooldown + state persistence.

        Steps:
            1. Get storage + brain
            2. Load persisted SyncState -> check cooldown
            3. Create adapter (Platform vs Self-hosted)
            4. Run SyncEngine.sync()
            5. Persist updated SyncState
            6. Log results
        """
        try:
            storage = await self.get_storage()  # type: ignore[attr-defined]
            brain = await storage.get_brain(storage._current_brain_id)
            if not brain:
                logger.warning("Mem0 auto-sync: no brain configured, skipping")
                return

            # Determine source name and load persisted state
            use_self_hosted = cfg.self_hosted
            source_name = "mem0_self_hosted" if use_self_hosted else "mem0"
            collection = cfg.user_id or cfg.agent_id or "default"

            sync_state = await storage.get_sync_state(source_name, collection)

            # Check cooldown
            if sync_state and sync_state.last_sync_at:
                cooldown = timedelta(minutes=cfg.cooldown_minutes)
                # Normalize to naive UTC for comparison
                last_sync = sync_state.last_sync_at
                if last_sync.tzinfo is not None:
                    last_sync = last_sync.replace(tzinfo=None)
                elapsed = utcnow() - last_sync
                if elapsed < cooldown:
                    remaining = cooldown - elapsed
                    logger.info(
                        "Mem0 auto-sync: cooldown active (%d min remaining), skipping",
                        int(remaining.total_seconds() / 60),
                    )
                    return

            # Create adapter
            adapter = self._create_mem0_adapter(has_api_key, use_self_hosted, cfg)

            # Run sync (SyncEngine manages its own disable/enable_auto_save)
            from neural_memory.integration.sync_engine import SyncEngine

            engine = SyncEngine(storage, brain.config)
            result, updated_state = await engine.sync(
                adapter=adapter,
                collection=collection,
                sync_state=sync_state,
                limit=cfg.limit,
            )

            # Persist updated sync state
            await storage.save_sync_state(updated_state, brain_id=brain.id)

            logger.info(
                "Mem0 auto-sync complete: %d imported, %d skipped from %s/%s (%.1fs)",
                result.records_imported,
                result.records_skipped,
                source_name,
                collection,
                result.duration_seconds,
            )

        except ImportError:
            logger.debug("Mem0 auto-sync: mem0ai package not installed, skipping")
        except Exception:
            logger.error("Mem0 auto-sync failed", exc_info=True)

    @staticmethod
    def _create_mem0_adapter(
        has_api_key: bool,
        use_self_hosted: bool,
        cfg: Mem0SyncConfig,
    ) -> _BaseMem0Adapter:
        """Create the appropriate Mem0 adapter based on config."""
        kwargs: dict[str, Any] = {}
        if cfg.user_id:
            kwargs["user_id"] = cfg.user_id
        if cfg.agent_id:
            kwargs["agent_id"] = cfg.agent_id

        if use_self_hosted:
            from neural_memory.integration.adapters.mem0_adapter import (
                Mem0SelfHostedAdapter,
            )

            return Mem0SelfHostedAdapter(**kwargs)

        from neural_memory.integration.adapters.mem0_adapter import Mem0Adapter

        api_key = os.environ.get("MEM0_API_KEY", "")
        return Mem0Adapter(api_key=api_key, **kwargs)

    def cancel_mem0_sync(self) -> None:
        """Cancel the background sync task if running."""
        if self._mem0_sync_task is not None and not self._mem0_sync_task.done():
            self._mem0_sync_task.cancel()
            logger.debug("Mem0 auto-sync task cancelled")


def _log_task_exception(task: asyncio.Task[None]) -> None:
    """Callback to log unhandled exceptions from the background sync task."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Mem0 auto-sync task raised unhandled exception: %s", exc)
