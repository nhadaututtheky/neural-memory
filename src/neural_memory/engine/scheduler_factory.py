"""Scheduler task factory for MCP and FastAPI.

Creates a pre-configured SchedulerCore with all background tasks registered.
Both MCP and FastAPI call `build_scheduler()` with their task implementations
to get a unified scheduler, eliminating scope fragmentation.

Usage (MCP):
    scheduler = build_scheduler(tasks={
        "consolidation": my_consolidation_fn,
        "health_pulse": my_health_fn,
        ...
    }, config=maintenance_config)
    await scheduler.start()

Usage (FastAPI):
    scheduler = build_scheduler(tasks={
        "consolidation": consolidation_fn,
        "decay": decay_fn,
        ...
    }, config=maintenance_config)
    await scheduler.start()
"""

from __future__ import annotations

import logging
from typing import Any

from neural_memory.engine.scheduler import SchedulerCore, TaskFn, TriggerType

logger = logging.getLogger(__name__)

# Task definitions: name → (trigger_type, config_key_for_interval_or_threshold)
# These define WHAT tasks exist; actual implementations are provided by callers.
TASK_DEFINITIONS: dict[str, dict[str, Any]] = {
    "consolidation": {
        "trigger_type": TriggerType.INTERVAL,
        "config_key": "scheduled_consolidation_interval_hours",
        "config_enabled_key": "scheduled_consolidation_enabled",
        "unit": "hours",
    },
    "decay": {
        "trigger_type": TriggerType.INTERVAL,
        "config_key": "decay_interval_hours",
        "config_enabled_key": "decay_enabled",
        "unit": "hours",
    },
    "version_check": {
        "trigger_type": TriggerType.INTERVAL,
        "config_key": "version_check_interval_hours",
        "config_enabled_key": "version_check_enabled",
        "unit": "hours",
    },
    "reindex": {
        "trigger_type": TriggerType.INTERVAL,
        "config_key": "reindex_interval_hours",
        "config_enabled_key": "reindex_enabled",
        "unit": "hours",
    },
    "health_pulse": {
        "trigger_type": TriggerType.OP_COUNTER,
        "config_key": "check_interval",
        "config_enabled_key": None,  # Always enabled when maintenance is enabled
    },
    "expiry_cleanup": {
        "trigger_type": TriggerType.OP_COUNTER,
        "config_key": "expiry_cleanup_interval_hours",
        "config_enabled_key": "expiry_cleanup_enabled",
        # Note: in MCP this piggybacks health pulse. As OP_COUNTER with a
        # higher threshold, it effectively runs less often.
    },
    "session_end": {
        "trigger_type": TriggerType.EVENT,
        "event_name": "session_end",
    },
    "notifications": {
        "trigger_type": TriggerType.INTERVAL,
        "config_key": "notifications_interval_hours",
        "config_enabled_key": "notifications_enabled",
        "unit": "hours",
    },
}


def build_scheduler(
    tasks: dict[str, TaskFn],
    config: Any | None = None,
) -> SchedulerCore:
    """Build a SchedulerCore with tasks registered from the provided map.

    Only tasks present in both `tasks` AND `TASK_DEFINITIONS` are registered.
    Callers provide the actual async callables; this function handles trigger
    type, interval, threshold, and enabled-state configuration.

    Args:
        tasks: Map of task_name → async callable. Only names matching
               TASK_DEFINITIONS are registered.
        config: MaintenanceConfig (or similar) with interval/threshold fields.
                If None, all tasks are registered with defaults.

    Returns:
        Configured SchedulerCore (not yet started).
    """
    scheduler = SchedulerCore()
    maintenance_enabled = getattr(config, "enabled", True) if config else True

    for name, fn in tasks.items():
        defn = TASK_DEFINITIONS.get(name)
        if defn is None:
            # Custom task not in definitions — register as MANUAL
            scheduler.register(name, fn, TriggerType.MANUAL)
            continue

        trigger_type: TriggerType = defn["trigger_type"]

        # Check if task is enabled via config
        enabled_key = defn.get("config_enabled_key")
        enabled = maintenance_enabled
        if enabled_key and config:
            enabled = maintenance_enabled and getattr(config, enabled_key, True)

        if trigger_type == TriggerType.INTERVAL:
            config_key = defn.get("config_key", "")
            unit = defn.get("unit", "hours")
            raw_interval = getattr(config, config_key, 24) if config else 24
            interval_seconds = raw_interval * 3600 if unit == "hours" else raw_interval

            scheduler.register(
                name,
                fn,
                TriggerType.INTERVAL,
                interval_seconds=interval_seconds,
                enabled=enabled,
            )

        elif trigger_type == TriggerType.OP_COUNTER:
            config_key = defn.get("config_key", "")
            threshold = getattr(config, config_key, 25) if config else 25
            scheduler.register(
                name,
                fn,
                TriggerType.OP_COUNTER,
                op_threshold=int(threshold),
                enabled=enabled,
            )

        elif trigger_type == TriggerType.EVENT:
            event_name = defn.get("event_name", name)
            scheduler.register(
                name,
                fn,
                TriggerType.EVENT,
                event_name=event_name,
                enabled=enabled,
            )

        else:
            scheduler.register(name, fn, TriggerType.MANUAL, enabled=enabled)

    registered = scheduler.task_names
    logger.info(
        "Scheduler built: %d tasks registered (%s)",
        len(registered),
        ", ".join(registered),
    )
    return scheduler
