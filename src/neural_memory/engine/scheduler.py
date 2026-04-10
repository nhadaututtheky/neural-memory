"""Unified scheduler core for MCP and FastAPI.

Provides a single scheduler engine that both server implementations use,
eliminating scope fragmentation where features exist in only one scope.

Supports four trigger types:
- INTERVAL: Fixed-period background tasks (consolidation, decay, version check)
- OP_COUNTER: Fires after N operations (health pulse, expiry cleanup)
- EVENT: Named manual triggers (session_end)
- MANUAL: Explicitly triggered (not auto-scheduled)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class TriggerType(StrEnum):
    """How a scheduled task gets triggered."""

    INTERVAL = "interval"  # Fixed period background loop
    OP_COUNTER = "op_counter"  # After N operations via tick()
    EVENT = "event"  # Named trigger (e.g. "session_end")
    MANUAL = "manual"  # Explicitly called, not auto-scheduled


# Type alias for async task functions
TaskFn = Callable[[], Coroutine[Any, Any, None]]


@dataclass
class ScheduledTask:
    """A registered task with its trigger configuration."""

    name: str
    fn: TaskFn
    trigger_type: TriggerType
    interval_seconds: float = 0.0  # For INTERVAL type
    op_threshold: int = 0  # For OP_COUNTER type
    event_name: str = ""  # For EVENT type
    enabled: bool = True

    # Runtime state
    _op_count: int = field(default=0, repr=False)
    _running: bool = field(default=False, repr=False)
    _last_run: float = field(default=0.0, repr=False)


class SchedulerCore:
    """Unified scheduler that both MCP and FastAPI use.

    Usage:
        scheduler = SchedulerCore()
        scheduler.register("consolidation", my_fn, TriggerType.INTERVAL, interval_seconds=86400)
        scheduler.register("health_pulse", pulse_fn, TriggerType.OP_COUNTER, op_threshold=50)
        scheduler.register("session_end", end_fn, TriggerType.EVENT, event_name="session_end")

        await scheduler.start()  # Starts INTERVAL tasks as background loops
        scheduler.tick()          # Called per-operation, triggers OP_COUNTER tasks
        await scheduler.trigger("session_end")  # Fires EVENT tasks
        await scheduler.stop()   # Graceful shutdown
    """

    def __init__(self) -> None:
        self._tasks: dict[str, ScheduledTask] = {}
        self._background_tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._lock = asyncio.Lock()

    def register(
        self,
        name: str,
        fn: TaskFn,
        trigger_type: TriggerType,
        interval_seconds: float = 0.0,
        op_threshold: int = 0,
        event_name: str = "",
        enabled: bool = True,
    ) -> None:
        """Register a task with the scheduler.

        Args:
            name: Unique task name.
            fn: Async callable to execute.
            trigger_type: How this task gets triggered.
            interval_seconds: Period for INTERVAL tasks.
            op_threshold: Operation count for OP_COUNTER tasks.
            event_name: Event name for EVENT tasks.
            enabled: Whether task is active.
        """
        self._tasks[name] = ScheduledTask(
            name=name,
            fn=fn,
            trigger_type=trigger_type,
            interval_seconds=interval_seconds,
            op_threshold=op_threshold,
            event_name=event_name or name,
            enabled=enabled,
        )

    def unregister(self, name: str) -> None:
        """Remove a registered task."""
        self._tasks.pop(name, None)

    async def start(self) -> None:
        """Start all INTERVAL tasks as background loops."""
        if self._running:
            return
        self._running = True

        for task in self._tasks.values():
            if task.trigger_type == TriggerType.INTERVAL and task.enabled:
                bg = asyncio.create_task(
                    self._interval_loop(task),
                    name=f"scheduler:{task.name}",
                )
                self._background_tasks.append(bg)

        logger.info(
            "Scheduler started: %d tasks registered (%d interval loops)",
            len(self._tasks),
            len(self._background_tasks),
        )

    async def stop(self) -> None:
        """Stop all background tasks gracefully."""
        self._running = False
        for bg in self._background_tasks:
            bg.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        logger.info("Scheduler stopped")

    def tick(self) -> None:
        """Called per-operation. Triggers OP_COUNTER tasks that hit threshold.

        This is synchronous — it fires-and-forgets async tasks.
        Safe to call from sync or async code.
        """
        for task in self._tasks.values():
            if task.trigger_type != TriggerType.OP_COUNTER or not task.enabled:
                continue
            task._op_count += 1
            if task._op_count >= task.op_threshold and not task._running:
                task._op_count = 0
                bg = asyncio.ensure_future(self._run_task(task))
                self._background_tasks.append(bg)

    async def trigger(self, event_name: str) -> None:
        """Fire all EVENT tasks matching the given event name."""
        for task in self._tasks.values():
            if task.trigger_type == TriggerType.EVENT and task.event_name == event_name:
                if task.enabled and not task._running:
                    await self._run_task(task)

    async def run(self, name: str) -> None:
        """Manually run a specific task by name."""
        task = self._tasks.get(name)
        if task is not None and task.enabled:
            await self._run_task(task)

    @property
    def task_names(self) -> list[str]:
        """List all registered task names."""
        return list(self._tasks.keys())

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is active."""
        return self._running

    def get_task_info(self) -> list[dict[str, Any]]:
        """Get info about all registered tasks."""
        return [
            {
                "name": t.name,
                "trigger_type": t.trigger_type.value,
                "enabled": t.enabled,
                "running": t._running,
                "interval_seconds": t.interval_seconds,
                "op_threshold": t.op_threshold,
                "event_name": t.event_name,
            }
            for t in self._tasks.values()
        ]

    # ── Internal ──

    async def _run_task(self, task: ScheduledTask) -> None:
        """Execute a task with error handling and re-entrancy guard."""
        if task._running:
            return
        task._running = True
        try:
            await task.fn()
            task._last_run = utcnow().timestamp()
        except Exception:
            logger.error("Scheduler task %s failed", task.name, exc_info=True)
        finally:
            task._running = False

    async def _interval_loop(self, task: ScheduledTask) -> None:
        """Background loop for INTERVAL tasks."""
        # Initial delay to avoid thundering herd at startup
        await asyncio.sleep(min(5.0, task.interval_seconds * 0.1))

        while self._running:
            try:
                await self._run_task(task)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("Interval loop %s crashed, retrying", task.name, exc_info=True)

            try:
                await asyncio.sleep(task.interval_seconds)
            except asyncio.CancelledError:
                break
