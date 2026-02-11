"""Tests for proactive brain maintenance handler."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from neural_memory.core.trigger_engine import TriggerType
from neural_memory.mcp.maintenance_handler import (
    HealthPulse,
    MaintenanceHandler,
    _evaluate_thresholds,
)
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.unified_config import (
    MaintenanceConfig,
    UnifiedConfig,
)

# ========== Helpers ==========

BRAIN_ID = "test-brain"


def _make_storage() -> InMemoryStorage:
    """Create an InMemoryStorage with a brain context set."""
    storage = InMemoryStorage()
    storage.set_brain(BRAIN_ID)
    storage.disable_auto_save = lambda: None  # type: ignore[attr-defined]
    storage.enable_auto_save = lambda: None  # type: ignore[attr-defined]

    async def _batch_save() -> None:
        pass

    storage.batch_save = _batch_save  # type: ignore[attr-defined]

    async def _record_action(**kwargs: Any) -> None:
        pass

    storage.record_action = _record_action  # type: ignore[attr-defined]
    return storage


class _FakeServer(MaintenanceHandler):
    """Minimal server stub that provides get_storage() and config."""

    def __init__(
        self,
        storage: InMemoryStorage,
        maintenance_cfg: MaintenanceConfig | None = None,
    ) -> None:
        self._storage = storage
        self.config = UnifiedConfig(
            maintenance=maintenance_cfg or MaintenanceConfig(),
        )

    async def get_storage(self) -> InMemoryStorage:
        return self._storage


# ========== MaintenanceConfig tests ==========


class TestMaintenanceConfig:
    def test_defaults(self) -> None:
        cfg = MaintenanceConfig()
        assert cfg.enabled is True
        assert cfg.check_interval == 25
        assert cfg.fiber_warn_threshold == 500
        assert cfg.neuron_warn_threshold == 2000
        assert cfg.synapse_warn_threshold == 5000
        assert cfg.orphan_ratio_threshold == 0.25
        assert cfg.expired_memory_warn_threshold == 10
        assert cfg.stale_fiber_ratio_threshold == 0.3
        assert cfg.stale_fiber_days == 90
        assert cfg.auto_consolidate is False
        assert cfg.auto_consolidate_strategies == ("prune", "merge")
        assert cfg.consolidate_cooldown_minutes == 60

    def test_immutability(self) -> None:
        cfg = MaintenanceConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cfg.check_interval = 10  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        cfg = MaintenanceConfig(
            enabled=False,
            check_interval=10,
            fiber_warn_threshold=100,
            neuron_warn_threshold=500,
            synapse_warn_threshold=1000,
            orphan_ratio_threshold=0.5,
            expired_memory_warn_threshold=5,
            stale_fiber_ratio_threshold=0.5,
            stale_fiber_days=60,
            auto_consolidate=True,
            auto_consolidate_strategies=("prune",),
            consolidate_cooldown_minutes=30,
        )
        data = cfg.to_dict()
        restored = MaintenanceConfig.from_dict(data)
        assert restored == cfg

    def test_from_dict_defaults(self) -> None:
        cfg = MaintenanceConfig.from_dict({})
        assert cfg == MaintenanceConfig()

    def test_from_dict_list_strategies(self) -> None:
        cfg = MaintenanceConfig.from_dict(
            {"auto_consolidate_strategies": ["prune", "merge", "summarize"]}
        )
        assert cfg.auto_consolidate_strategies == ("prune", "merge", "summarize")


# ========== HealthPulse tests ==========


class TestHealthPulse:
    def test_construction(self) -> None:
        pulse = HealthPulse(
            fiber_count=100,
            neuron_count=500,
            synapse_count=800,
            connectivity=1.6,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=(),
            should_consolidate=False,
        )
        assert pulse.fiber_count == 100
        assert pulse.neuron_count == 500
        assert pulse.expired_memory_count == 0
        assert pulse.stale_fiber_ratio == 0.0
        assert pulse.hints == ()
        assert pulse.should_consolidate is False

    def test_frozen(self) -> None:
        pulse = HealthPulse(
            fiber_count=0,
            neuron_count=0,
            synapse_count=0,
            connectivity=0.0,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=(),
            should_consolidate=False,
        )
        with pytest.raises(AttributeError):
            pulse.fiber_count = 99  # type: ignore[misc]


# ========== Threshold evaluation tests ==========


class TestEvaluateThresholds:
    def test_healthy_brain_no_hints(self) -> None:
        cfg = MaintenanceConfig()
        hints = _evaluate_thresholds(
            fiber_count=50,
            neuron_count=200,
            synapse_count=400,
            connectivity=2.0,
            orphan_ratio=0.1,
            cfg=cfg,
        )
        assert hints == []

    def test_high_neuron_count(self) -> None:
        cfg = MaintenanceConfig(neuron_warn_threshold=100)
        hints = _evaluate_thresholds(
            fiber_count=50,
            neuron_count=150,
            synapse_count=300,
            connectivity=2.0,
            orphan_ratio=0.0,
            cfg=cfg,
        )
        assert len(hints) == 1
        assert "neuron count (150)" in hints[0].lower()
        assert "prune" in hints[0].lower()

    def test_high_fiber_count(self) -> None:
        cfg = MaintenanceConfig(fiber_warn_threshold=100)
        hints = _evaluate_thresholds(
            fiber_count=150,
            neuron_count=50,
            synapse_count=100,
            connectivity=2.0,
            orphan_ratio=0.0,
            cfg=cfg,
        )
        assert len(hints) == 1
        assert "fiber count (150)" in hints[0].lower()
        assert "merge" in hints[0].lower()

    def test_high_synapse_count(self) -> None:
        cfg = MaintenanceConfig(synapse_warn_threshold=100)
        hints = _evaluate_thresholds(
            fiber_count=10,
            neuron_count=20,
            synapse_count=150,
            connectivity=7.5,
            orphan_ratio=0.0,
            cfg=cfg,
        )
        assert any("synapse count (150)" in h.lower() for h in hints)

    def test_low_connectivity(self) -> None:
        cfg = MaintenanceConfig()
        hints = _evaluate_thresholds(
            fiber_count=10,
            neuron_count=50,
            synapse_count=50,
            connectivity=1.0,
            orphan_ratio=0.0,
            cfg=cfg,
        )
        assert any("connectivity" in h.lower() for h in hints)
        assert any("enrich" in h.lower() for h in hints)

    def test_low_connectivity_skipped_small_brain(self) -> None:
        cfg = MaintenanceConfig()
        hints = _evaluate_thresholds(
            fiber_count=1,
            neuron_count=5,
            synapse_count=2,
            connectivity=0.4,
            orphan_ratio=0.0,
            cfg=cfg,
        )
        assert not any("connectivity" in h.lower() for h in hints)

    def test_high_orphan_ratio(self) -> None:
        cfg = MaintenanceConfig(orphan_ratio_threshold=0.2)
        hints = _evaluate_thresholds(
            fiber_count=5,
            neuron_count=100,
            synapse_count=200,
            connectivity=2.0,
            orphan_ratio=0.5,
            cfg=cfg,
        )
        assert any("orphan" in h.lower() for h in hints)

    def test_multiple_issues(self) -> None:
        cfg = MaintenanceConfig(
            neuron_warn_threshold=100,
            fiber_warn_threshold=50,
        )
        hints = _evaluate_thresholds(
            fiber_count=100,
            neuron_count=200,
            synapse_count=100,
            connectivity=0.5,
            orphan_ratio=0.0,
            cfg=cfg,
        )
        assert len(hints) >= 3  # neuron + fiber + connectivity


# ========== Operation counter and interval tests ==========


class TestOperationCounter:
    @pytest.mark.asyncio
    async def test_counter_increments(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        assert server._op_count == 0
        server._increment_op_counter()
        assert server._op_count == 1
        server._increment_op_counter()
        assert server._op_count == 2

    @pytest.mark.asyncio
    async def test_should_check_at_interval(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(check_interval=5)
        server = _FakeServer(storage, cfg)

        checks = []
        for _ in range(15):
            server._increment_op_counter()
            checks.append(server._should_check_health())

        # Should be True at positions 4, 9, 14 (op_count 5, 10, 15)
        assert checks[4] is True
        assert checks[9] is True
        assert checks[14] is True
        # Should be False at other positions
        assert checks[0] is False
        assert checks[3] is False

    @pytest.mark.asyncio
    async def test_disabled_never_checks(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(enabled=False, check_interval=1)
        server = _FakeServer(storage, cfg)

        for _ in range(10):
            server._increment_op_counter()
            assert server._should_check_health() is False


# ========== Health pulse tests ==========


class TestHealthPulseMethod:
    @pytest.mark.asyncio
    async def test_pulse_returns_none_when_disabled(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(enabled=False)
        server = _FakeServer(storage, cfg)
        result = await server._health_pulse()
        assert result is None

    @pytest.mark.asyncio
    async def test_pulse_returns_healthy(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        pulse = await server._health_pulse()
        assert pulse is not None
        assert pulse.fiber_count == 0
        assert pulse.neuron_count == 0
        assert pulse.hints == ()
        assert pulse.should_consolidate is False

    @pytest.mark.asyncio
    async def test_pulse_caches_result(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        assert server._last_pulse is None
        pulse = await server._health_pulse()
        assert server._last_pulse is pulse


# ========== Hint formatting tests ==========


class TestGetMaintenanceHint:
    def test_none_pulse(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        assert server._get_maintenance_hint(None) is None

    def test_no_hints(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        pulse = HealthPulse(
            fiber_count=0,
            neuron_count=0,
            synapse_count=0,
            connectivity=0.0,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=(),
            should_consolidate=False,
        )
        assert server._get_maintenance_hint(pulse) is None

    def test_returns_first_hint(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        pulse = HealthPulse(
            fiber_count=0,
            neuron_count=0,
            synapse_count=0,
            connectivity=0.0,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=("First hint", "Second hint"),
            should_consolidate=False,
        )
        assert server._get_maintenance_hint(pulse) == "First hint"


# ========== Auto-consolidation tests ==========


class TestAutoConsolidation:
    @pytest.mark.asyncio
    async def test_skips_when_disabled(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(auto_consolidate=False)
        server = _FakeServer(storage, cfg)
        pulse = HealthPulse(
            fiber_count=0,
            neuron_count=0,
            synapse_count=0,
            connectivity=0.0,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=("some hint",),
            should_consolidate=False,
        )
        await server._maybe_auto_consolidate(pulse)
        assert server._last_consolidation_at is None

    @pytest.mark.asyncio
    async def test_respects_cooldown(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(auto_consolidate=True, consolidate_cooldown_minutes=60)
        server = _FakeServer(storage, cfg)

        # Set last consolidation to 30 min ago (within cooldown)
        server._last_consolidation_at = datetime.now(UTC) - timedelta(
            minutes=30
        )
        original_time = server._last_consolidation_at

        pulse = HealthPulse(
            fiber_count=0,
            neuron_count=0,
            synapse_count=0,
            connectivity=0.0,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=("hint",),
            should_consolidate=True,
        )
        await server._maybe_auto_consolidate(pulse)
        # Should not have updated the timestamp since cooldown blocked it
        assert server._last_consolidation_at == original_time

    @pytest.mark.asyncio
    async def test_triggers_after_cooldown(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(auto_consolidate=True, consolidate_cooldown_minutes=60)
        server = _FakeServer(storage, cfg)

        # Set last consolidation to 90 min ago (past cooldown)
        server._last_consolidation_at = datetime.now(UTC) - timedelta(
            minutes=90
        )

        pulse = HealthPulse(
            fiber_count=0,
            neuron_count=0,
            synapse_count=0,
            connectivity=0.0,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=("hint",),
            should_consolidate=True,
        )

        with patch.object(
            server, "_run_auto_consolidation", new_callable=AsyncMock
        ) as mock_run:
            await server._maybe_auto_consolidate(pulse)
            # create_task fires the coroutine; give it a tick
            await asyncio.sleep(0.05)
            mock_run.assert_called_once()


# ========== Integration: _check_maintenance ==========


class TestCheckMaintenance:
    @pytest.mark.asyncio
    async def test_returns_none_before_interval(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(check_interval=5)
        server = _FakeServer(storage, cfg)

        # First 4 ops should not trigger a check
        for _ in range(4):
            result = await server._check_maintenance()
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_pulse_at_interval(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(check_interval=3)
        server = _FakeServer(storage, cfg)

        results = []
        for _ in range(6):
            results.append(await server._check_maintenance())

        # At op 3 and 6 we should get a pulse
        assert results[2] is not None
        assert results[5] is not None
        # Others should be None
        assert results[0] is None
        assert results[1] is None
        assert results[3] is None
        assert results[4] is None

    @pytest.mark.asyncio
    async def test_disabled_always_returns_none(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(enabled=False, check_interval=1)
        server = _FakeServer(storage, cfg)

        for _ in range(5):
            result = await server._check_maintenance()
            assert result is None


# ========== Health degradation trigger tests ==========


class TestFireHealthTrigger:
    def test_no_hints_returns_not_triggered(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        pulse = HealthPulse(
            fiber_count=10,
            neuron_count=50,
            synapse_count=100,
            connectivity=2.0,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=(),
            should_consolidate=False,
        )
        result = server._fire_health_trigger(pulse)
        assert result.triggered is False
        assert result.trigger_type is None

    def test_hints_returns_health_degradation(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        pulse = HealthPulse(
            fiber_count=600,
            neuron_count=3000,
            synapse_count=100,
            connectivity=2.0,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=("High neuron count (3000). Consider pruning.",),
            should_consolidate=False,
        )
        result = server._fire_health_trigger(pulse)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.HEALTH_DEGRADATION
        assert "neuron count" in result.message.lower()
        assert result.save_tiers == (3,)

    def test_multiple_hints_uses_first(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        pulse = HealthPulse(
            fiber_count=600,
            neuron_count=3000,
            synapse_count=6000,
            connectivity=2.0,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=(
                "High neuron count (3000).",
                "High synapse count (6000).",
            ),
            should_consolidate=False,
        )
        result = server._fire_health_trigger(pulse)
        assert result.triggered is True
        assert "neuron" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_maintenance_fires_trigger_on_degradation(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(check_interval=1, neuron_warn_threshold=10)
        server = _FakeServer(storage, cfg)

        # Mock get_stats to return high neuron count
        async def _high_stats(brain_id: str) -> dict[str, int]:
            return {"neuron_count": 50, "synapse_count": 100, "fiber_count": 5, "project_count": 0}

        storage.get_stats = _high_stats  # type: ignore[assignment]

        with patch.object(server, "_fire_health_trigger", wraps=server._fire_health_trigger) as mock_fire:
            pulse = await server._check_maintenance()
            assert pulse is not None
            assert pulse.hints  # should have neuron warning
            mock_fire.assert_called_once_with(pulse)


# ========== P2: Expired memory hint tests ==========


class TestExpiredMemoryHint:
    def test_expired_below_threshold_no_hint(self) -> None:
        cfg = MaintenanceConfig(expired_memory_warn_threshold=10)
        hints = _evaluate_thresholds(
            fiber_count=50,
            neuron_count=200,
            synapse_count=400,
            connectivity=2.0,
            orphan_ratio=0.1,
            expired_memory_count=5,
            stale_fiber_ratio=0.0,
            cfg=cfg,
        )
        assert not any("expired" in h.lower() for h in hints)

    def test_expired_above_threshold_hint(self) -> None:
        cfg = MaintenanceConfig(expired_memory_warn_threshold=10)
        hints = _evaluate_thresholds(
            fiber_count=50,
            neuron_count=200,
            synapse_count=400,
            connectivity=2.0,
            orphan_ratio=0.1,
            expired_memory_count=15,
            stale_fiber_ratio=0.0,
            cfg=cfg,
        )
        assert any("expired" in h.lower() for h in hints)
        assert any("15" in h for h in hints)

    @pytest.mark.asyncio
    async def test_pulse_includes_expired_count(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        pulse = await server._health_pulse()
        assert pulse is not None
        assert pulse.expired_memory_count == 0

    @pytest.mark.asyncio
    async def test_pulse_with_expired_memories(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(check_interval=1, expired_memory_warn_threshold=1)
        server = _FakeServer(storage, cfg)

        # Mock get_expired_memory_count to return high count
        async def _mock_expired_count() -> int:
            return 5

        storage.get_expired_memory_count = _mock_expired_count  # type: ignore[attr-defined]

        pulse = await server._health_pulse()
        assert pulse is not None
        assert pulse.expired_memory_count == 5
        assert any("expired" in h.lower() for h in pulse.hints)


# ========== P3: Stale fiber hint tests ==========


class TestStaleFiberHint:
    def test_stale_below_threshold_no_hint(self) -> None:
        cfg = MaintenanceConfig(stale_fiber_ratio_threshold=0.3)
        hints = _evaluate_thresholds(
            fiber_count=100,
            neuron_count=200,
            synapse_count=400,
            connectivity=2.0,
            orphan_ratio=0.1,
            expired_memory_count=0,
            stale_fiber_ratio=0.2,
            cfg=cfg,
        )
        assert not any("stale" in h.lower() for h in hints)

    def test_stale_above_threshold_hint(self) -> None:
        cfg = MaintenanceConfig(stale_fiber_ratio_threshold=0.3, stale_fiber_days=90)
        hints = _evaluate_thresholds(
            fiber_count=100,
            neuron_count=200,
            synapse_count=400,
            connectivity=2.0,
            orphan_ratio=0.1,
            expired_memory_count=0,
            stale_fiber_ratio=0.5,
            cfg=cfg,
        )
        assert any("stale" in h.lower() for h in hints)
        assert any("50%" in h for h in hints)
        assert any("90" in h for h in hints)

    def test_stale_skipped_small_brain(self) -> None:
        cfg = MaintenanceConfig(stale_fiber_ratio_threshold=0.3)
        hints = _evaluate_thresholds(
            fiber_count=5,
            neuron_count=20,
            synapse_count=40,
            connectivity=2.0,
            orphan_ratio=0.0,
            expired_memory_count=0,
            stale_fiber_ratio=0.8,
            cfg=cfg,
        )
        assert not any("stale" in h.lower() for h in hints)

    @pytest.mark.asyncio
    async def test_pulse_includes_stale_ratio(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)
        pulse = await server._health_pulse()
        assert pulse is not None
        assert pulse.stale_fiber_ratio == 0.0

    @pytest.mark.asyncio
    async def test_pulse_with_stale_fibers(self) -> None:
        storage = _make_storage()
        cfg = MaintenanceConfig(
            check_interval=1,
            stale_fiber_ratio_threshold=0.2,
            stale_fiber_days=30,
        )
        server = _FakeServer(storage, cfg)

        # Mock stats and stale count
        async def _stats(brain_id: str) -> dict[str, int]:
            return {"neuron_count": 50, "synapse_count": 100, "fiber_count": 20, "project_count": 0}

        async def _stale_count(brain_id: str, stale_days: int = 90) -> int:
            return 10  # 50% stale

        storage.get_stats = _stats  # type: ignore[assignment]
        storage.get_stale_fiber_count = _stale_count  # type: ignore[attr-defined]

        pulse = await server._health_pulse()
        assert pulse is not None
        assert pulse.stale_fiber_ratio == 0.5
        assert any("stale" in h.lower() for h in pulse.hints)
