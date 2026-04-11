"""Tests for the unified scheduler engine (Phase 1).

Covers:
- Task registration and unregistration
- INTERVAL trigger (background loop)
- OP_COUNTER trigger (tick-based)
- EVENT trigger (named manual)
- MANUAL trigger (explicit run)
- Re-entrancy guard (no double execution)
- Start/stop lifecycle
- Task info introspection
"""

from __future__ import annotations

import asyncio

from neural_memory.engine.scheduler import ScheduledTask, SchedulerCore, TriggerType

# ── Helpers ──


class CallTracker:
    """Track async function calls."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self.call_count = 0

    async def __call__(self) -> None:
        self.call_count += 1
        self.calls.append(asyncio.get_event_loop().time())


class SlowTask:
    """Task that takes time, for testing re-entrancy guard."""

    def __init__(self, delay: float = 0.1) -> None:
        self.call_count = 0
        self.delay = delay

    async def __call__(self) -> None:
        self.call_count += 1
        await asyncio.sleep(self.delay)


class FailingTask:
    """Task that always raises."""

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(self) -> None:
        self.call_count += 1
        raise RuntimeError("intentional failure")


# ── Registration ──


class TestRegistration:
    def test_register_task(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("test", tracker, TriggerType.MANUAL)

        assert "test" in scheduler.task_names
        assert len(scheduler.task_names) == 1

    def test_register_multiple_tasks(self) -> None:
        scheduler = SchedulerCore()
        scheduler.register("a", CallTracker(), TriggerType.MANUAL)
        scheduler.register("b", CallTracker(), TriggerType.MANUAL)
        scheduler.register("c", CallTracker(), TriggerType.MANUAL)

        assert len(scheduler.task_names) == 3

    def test_register_overwrites(self) -> None:
        scheduler = SchedulerCore()
        t1 = CallTracker()
        t2 = CallTracker()
        scheduler.register("same", t1, TriggerType.MANUAL)
        scheduler.register("same", t2, TriggerType.MANUAL)

        assert len(scheduler.task_names) == 1

    def test_unregister(self) -> None:
        scheduler = SchedulerCore()
        scheduler.register("test", CallTracker(), TriggerType.MANUAL)
        scheduler.unregister("test")

        assert "test" not in scheduler.task_names

    def test_unregister_nonexistent(self) -> None:
        scheduler = SchedulerCore()
        scheduler.unregister("ghost")  # Should not raise


# ── MANUAL trigger ──


class TestManualTrigger:
    async def test_run_by_name(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("manual_task", tracker, TriggerType.MANUAL)

        await scheduler.run("manual_task")

        assert tracker.call_count == 1

    async def test_run_nonexistent(self) -> None:
        scheduler = SchedulerCore()
        await scheduler.run("ghost")  # Should not raise

    async def test_run_disabled_task(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("disabled", tracker, TriggerType.MANUAL, enabled=False)

        await scheduler.run("disabled")

        assert tracker.call_count == 0


# ── OP_COUNTER trigger ──


class TestOpCounter:
    async def test_tick_triggers_at_threshold(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("counter_task", tracker, TriggerType.OP_COUNTER, op_threshold=3)

        # Ticks 1, 2 — not enough
        scheduler.tick()
        scheduler.tick()
        await asyncio.sleep(0.02)
        assert tracker.call_count == 0

        # Tick 3 — fires
        scheduler.tick()
        await asyncio.sleep(0.05)
        assert tracker.call_count == 1

    async def test_tick_fires_at_threshold(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("counter_task", tracker, TriggerType.OP_COUNTER, op_threshold=3)

        scheduler.tick()
        scheduler.tick()
        scheduler.tick()

        # Let fire-and-forget complete
        await asyncio.sleep(0.05)

        assert tracker.call_count == 1

    async def test_tick_resets_counter(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("counter_task", tracker, TriggerType.OP_COUNTER, op_threshold=2)

        # First cycle
        scheduler.tick()
        scheduler.tick()
        await asyncio.sleep(0.05)
        assert tracker.call_count == 1

        # Second cycle
        scheduler.tick()
        scheduler.tick()
        await asyncio.sleep(0.05)
        assert tracker.call_count == 2

    async def test_tick_skips_disabled(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register(
            "disabled", tracker, TriggerType.OP_COUNTER, op_threshold=1, enabled=False
        )

        scheduler.tick()
        await asyncio.sleep(0.05)

        assert tracker.call_count == 0

    async def test_tick_ignores_non_counter_tasks(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("interval_task", tracker, TriggerType.INTERVAL, interval_seconds=1.0)

        for _ in range(10):
            scheduler.tick()
        await asyncio.sleep(0.05)

        assert tracker.call_count == 0


# ── EVENT trigger ──


class TestEventTrigger:
    async def test_trigger_fires_matching_event(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("end_task", tracker, TriggerType.EVENT, event_name="session_end")

        await scheduler.trigger("session_end")

        assert tracker.call_count == 1

    async def test_trigger_skips_non_matching(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("end_task", tracker, TriggerType.EVENT, event_name="session_end")

        await scheduler.trigger("other_event")

        assert tracker.call_count == 0

    async def test_trigger_fires_multiple_handlers(self) -> None:
        scheduler = SchedulerCore()
        t1 = CallTracker()
        t2 = CallTracker()
        scheduler.register("handler_a", t1, TriggerType.EVENT, event_name="shutdown")
        scheduler.register("handler_b", t2, TriggerType.EVENT, event_name="shutdown")

        await scheduler.trigger("shutdown")

        assert t1.call_count == 1
        assert t2.call_count == 1

    async def test_trigger_skips_disabled(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("disabled", tracker, TriggerType.EVENT, event_name="test", enabled=False)

        await scheduler.trigger("test")

        assert tracker.call_count == 0


# ── INTERVAL trigger ──


class TestIntervalTrigger:
    async def test_start_creates_background_tasks(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("loop", tracker, TriggerType.INTERVAL, interval_seconds=0.05)

        await scheduler.start()
        assert scheduler.is_running

        # Wait for initial delay + at least one execution
        await asyncio.sleep(0.15)
        await scheduler.stop()

        assert tracker.call_count >= 1

    async def test_start_idempotent(self) -> None:
        scheduler = SchedulerCore()
        scheduler.register("loop", CallTracker(), TriggerType.INTERVAL, interval_seconds=1.0)

        await scheduler.start()
        await scheduler.start()  # Should not double-start

        await scheduler.stop()

    async def test_stop_cancels_all(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register("loop", tracker, TriggerType.INTERVAL, interval_seconds=0.05)

        await scheduler.start()
        await asyncio.sleep(0.1)
        await scheduler.stop()

        count_at_stop = tracker.call_count
        await asyncio.sleep(0.1)

        # No more executions after stop
        assert tracker.call_count == count_at_stop
        assert not scheduler.is_running

    async def test_disabled_interval_not_started(self) -> None:
        scheduler = SchedulerCore()
        tracker = CallTracker()
        scheduler.register(
            "disabled", tracker, TriggerType.INTERVAL, interval_seconds=0.05, enabled=False
        )

        await scheduler.start()
        await asyncio.sleep(0.15)
        await scheduler.stop()

        assert tracker.call_count == 0


# ── Error handling ──


class TestErrorHandling:
    async def test_failing_task_doesnt_crash(self) -> None:
        scheduler = SchedulerCore()
        failing = FailingTask()
        scheduler.register("failing", failing, TriggerType.MANUAL)

        await scheduler.run("failing")

        assert failing.call_count == 1

    async def test_failing_interval_retries(self) -> None:
        scheduler = SchedulerCore()
        failing = FailingTask()
        scheduler.register("failing", failing, TriggerType.INTERVAL, interval_seconds=0.05)

        await scheduler.start()
        await asyncio.sleep(0.2)
        await scheduler.stop()

        # Should have retried multiple times
        assert failing.call_count >= 2

    async def test_failing_op_counter_resets(self) -> None:
        scheduler = SchedulerCore()
        failing = FailingTask()
        scheduler.register("failing", failing, TriggerType.OP_COUNTER, op_threshold=1)

        scheduler.tick()
        await asyncio.sleep(0.05)
        assert failing.call_count == 1

        # Counter should reset even after failure, allowing re-trigger
        scheduler.tick()
        await asyncio.sleep(0.05)
        assert failing.call_count == 2


# ── Re-entrancy guard ──


class TestReentrancyGuard:
    async def test_no_double_execution(self) -> None:
        scheduler = SchedulerCore()
        slow = SlowTask(delay=0.2)
        scheduler.register("slow", slow, TriggerType.MANUAL)

        # Fire two concurrent runs
        await asyncio.gather(scheduler.run("slow"), scheduler.run("slow"))

        # Second should be skipped due to _running guard
        assert slow.call_count == 1

    async def test_op_counter_skips_while_running(self) -> None:
        scheduler = SchedulerCore()
        slow = SlowTask(delay=0.2)
        scheduler.register("slow", slow, TriggerType.OP_COUNTER, op_threshold=1)

        scheduler.tick()  # Start task
        await asyncio.sleep(0.02)
        scheduler.tick()  # Should skip (still running)
        scheduler.tick()  # Should skip

        await asyncio.sleep(0.3)

        assert slow.call_count == 1


# ── Task info ──


class TestTaskInfo:
    def test_get_task_info(self) -> None:
        scheduler = SchedulerCore()
        scheduler.register("a", CallTracker(), TriggerType.INTERVAL, interval_seconds=60)
        scheduler.register("b", CallTracker(), TriggerType.OP_COUNTER, op_threshold=10)

        info = scheduler.get_task_info()
        assert len(info) == 2

        names = {i["name"] for i in info}
        assert names == {"a", "b"}

        # Check fields exist
        for item in info:
            assert "trigger_type" in item
            assert "enabled" in item
            assert "running" in item

    def test_task_info_trigger_types(self) -> None:
        scheduler = SchedulerCore()
        scheduler.register("i", CallTracker(), TriggerType.INTERVAL, interval_seconds=1)
        scheduler.register("o", CallTracker(), TriggerType.OP_COUNTER, op_threshold=5)
        scheduler.register("e", CallTracker(), TriggerType.EVENT, event_name="test")
        scheduler.register("m", CallTracker(), TriggerType.MANUAL)

        info = {i["name"]: i["trigger_type"] for i in scheduler.get_task_info()}
        assert info == {"i": "interval", "o": "op_counter", "e": "event", "m": "manual"}


# ── ScheduledTask dataclass ──


class TestScheduledTaskModel:
    def test_defaults(self) -> None:
        task = ScheduledTask(name="t", fn=CallTracker(), trigger_type=TriggerType.MANUAL)
        assert task.enabled is True
        assert task.interval_seconds == 0.0
        assert task.op_threshold == 0
        assert task.event_name == ""
        assert task._op_count == 0
        assert task._running is False

    def test_event_name_defaults_to_name_on_register(self) -> None:
        """When event_name not provided, register() sets it to name."""
        scheduler = SchedulerCore()
        scheduler.register("my_event", CallTracker(), TriggerType.EVENT)

        info = scheduler.get_task_info()
        assert info[0]["event_name"] == "my_event"
