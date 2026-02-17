"""Tests for Proactive Alerts Queue (Feature B)."""

from __future__ import annotations

from uuid import uuid4

import pytest
import pytest_asyncio

from neural_memory.core.alert import Alert, AlertStatus, AlertType
from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.mcp.alert_handler import _hint_to_alert_type
from neural_memory.mcp.maintenance_handler import HealthHint, HintSeverity
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow


# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store(tmp_path: object) -> SQLiteStorage:
    """SQLite storage with alerts table, ready for testing."""
    import pathlib

    db_path = pathlib.Path(str(tmp_path)) / "test_alerts.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()

    brain = Brain.create(name="alert-test", config=BrainConfig(), owner_id="test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


def _make_alert(
    alert_type: AlertType = AlertType.HIGH_NEURON_COUNT,
    severity: str = "medium",
    message: str = "Test alert",
    status: AlertStatus = AlertStatus.ACTIVE,
) -> Alert:
    """Create a test alert."""
    return Alert(
        id=uuid4().hex[:16],
        brain_id="placeholder",
        alert_type=alert_type,
        severity=severity,
        message=message,
        status=status,
        created_at=utcnow(),
    )


# ── Alert Model Tests ───────────────────────────────────────────


class TestAlertModel:
    """Alert frozen dataclass tests."""

    def test_frozen(self) -> None:
        alert = _make_alert()
        with pytest.raises(AttributeError):
            alert.status = AlertStatus.RESOLVED  # type: ignore[misc]

    def test_defaults(self) -> None:
        alert = Alert(
            id="a1",
            brain_id="b1",
            alert_type=AlertType.LOW_CONNECTIVITY,
        )
        assert alert.severity == "low"
        assert alert.status == AlertStatus.ACTIVE
        assert alert.seen_at is None
        assert alert.acknowledged_at is None
        assert alert.resolved_at is None
        assert alert.metadata == {}

    def test_alert_types(self) -> None:
        for at in AlertType:
            assert at.value == at.value.lower()

    def test_alert_statuses(self) -> None:
        assert list(AlertStatus) == [
            AlertStatus.ACTIVE,
            AlertStatus.SEEN,
            AlertStatus.ACKNOWLEDGED,
            AlertStatus.RESOLVED,
        ]


# ── Hint-to-AlertType Mapping Tests ─────────────────────────────


class TestHintMapping:
    """Tests for _hint_to_alert_type mapping."""

    def test_high_neuron_count(self) -> None:
        hint = HealthHint(
            message="High neuron count (5000). Consider running consolidation.",
            severity=HintSeverity.MEDIUM,
            recommended_strategy="prune",
        )
        assert _hint_to_alert_type(hint) == AlertType.HIGH_NEURON_COUNT

    def test_high_fiber_count(self) -> None:
        hint = HealthHint(
            message="High fiber count (2000). Consider running consolidation.",
            severity=HintSeverity.MEDIUM,
            recommended_strategy="merge",
        )
        assert _hint_to_alert_type(hint) == AlertType.HIGH_FIBER_COUNT

    def test_low_connectivity(self) -> None:
        hint = HealthHint(
            message="Low connectivity (0.8 synapses/neuron). Consider enrich.",
            severity=HintSeverity.LOW,
            recommended_strategy="enrich",
        )
        assert _hint_to_alert_type(hint) == AlertType.LOW_CONNECTIVITY

    def test_expired_memories(self) -> None:
        hint = HealthHint(
            message="15 expired memories found. Consider cleanup.",
            severity=HintSeverity.LOW,
            recommended_strategy="prune",
        )
        assert _hint_to_alert_type(hint) == AlertType.EXPIRED_MEMORIES

    def test_stale_fibers(self) -> None:
        hint = HealthHint(
            message="45% of fibers are stale (>90 days unused).",
            severity=HintSeverity.LOW,
            recommended_strategy="prune",
        )
        assert _hint_to_alert_type(hint) == AlertType.STALE_FIBERS

    def test_unknown_hint(self) -> None:
        hint = HealthHint(
            message="Something unknown happened.",
            severity=HintSeverity.LOW,
            recommended_strategy="none",
        )
        assert _hint_to_alert_type(hint) is None


# ── SQLite Alerts Storage Tests ──────────────────────────────────


class TestSQLiteAlerts:
    """Tests for SQLiteAlertsMixin operations."""

    async def test_record_and_get(self, store: SQLiteStorage) -> None:
        alert = _make_alert()
        result = await store.record_alert(alert)
        assert result == alert.id

        fetched = await store.get_alert(alert.id)
        assert fetched is not None
        assert fetched.alert_type == AlertType.HIGH_NEURON_COUNT
        assert fetched.status == AlertStatus.ACTIVE

    async def test_dedup_cooldown(self, store: SQLiteStorage) -> None:
        alert1 = _make_alert()
        result1 = await store.record_alert(alert1)
        assert result1 != ""

        alert2 = _make_alert()  # Same type, within 6h cooldown
        result2 = await store.record_alert(alert2)
        assert result2 == ""  # Suppressed

    async def test_different_types_not_deduped(self, store: SQLiteStorage) -> None:
        a1 = _make_alert(alert_type=AlertType.HIGH_NEURON_COUNT)
        a2 = _make_alert(alert_type=AlertType.LOW_CONNECTIVITY)
        r1 = await store.record_alert(a1)
        r2 = await store.record_alert(a2)
        assert r1 != ""
        assert r2 != ""

    async def test_get_active_alerts(self, store: SQLiteStorage) -> None:
        a1 = _make_alert(severity="critical")
        a2 = _make_alert(alert_type=AlertType.LOW_CONNECTIVITY, severity="low")
        await store.record_alert(a1)
        await store.record_alert(a2)

        alerts = await store.get_active_alerts()
        assert len(alerts) == 2
        # Critical severity should come first
        assert alerts[0].severity == "critical"

    async def test_count_pending_alerts(self, store: SQLiteStorage) -> None:
        a1 = _make_alert()
        await store.record_alert(a1)

        count = await store.count_pending_alerts()
        assert count == 1

    async def test_mark_seen(self, store: SQLiteStorage) -> None:
        a = _make_alert()
        await store.record_alert(a)

        updated = await store.mark_alerts_seen([a.id])
        assert updated == 1

        fetched = await store.get_alert(a.id)
        assert fetched is not None
        assert fetched.status == AlertStatus.SEEN
        assert fetched.seen_at is not None

    async def test_mark_acknowledged(self, store: SQLiteStorage) -> None:
        a = _make_alert()
        await store.record_alert(a)

        result = await store.mark_alert_acknowledged(a.id)
        assert result is True

        fetched = await store.get_alert(a.id)
        assert fetched is not None
        assert fetched.status == AlertStatus.ACKNOWLEDGED

    async def test_resolve_by_type(self, store: SQLiteStorage) -> None:
        a1 = _make_alert(alert_type=AlertType.HIGH_NEURON_COUNT)
        a2 = _make_alert(alert_type=AlertType.LOW_CONNECTIVITY)
        await store.record_alert(a1)
        await store.record_alert(a2)

        resolved = await store.resolve_alerts_by_type([AlertType.HIGH_NEURON_COUNT.value])
        assert resolved == 1

        remaining = await store.get_active_alerts()
        assert len(remaining) == 1
        assert remaining[0].alert_type == AlertType.LOW_CONNECTIVITY

    async def test_acknowledged_not_resolved_by_type(self, store: SQLiteStorage) -> None:
        a = _make_alert()
        await store.record_alert(a)
        await store.mark_alert_acknowledged(a.id)

        resolved = await store.resolve_alerts_by_type([AlertType.HIGH_NEURON_COUNT.value])
        assert resolved == 0

    async def test_get_nonexistent_alert(self, store: SQLiteStorage) -> None:
        result = await store.get_alert("nonexistent")
        assert result is None
