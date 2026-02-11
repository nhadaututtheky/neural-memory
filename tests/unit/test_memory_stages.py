"""Unit tests for memory maturation lifecycle — STM → Episodic → Semantic."""

from __future__ import annotations

from datetime import datetime, timedelta

from neural_memory.engine.memory_stages import (
    MaturationRecord,
    MemoryStage,
    compute_stage_transition,
    get_decay_multiplier,
)


class TestMemoryStage:
    """Tests for MemoryStage enum and decay multipliers."""

    def test_stage_values(self) -> None:
        """All stages should have string values."""
        assert MemoryStage.SHORT_TERM == "stm"
        assert MemoryStage.WORKING == "working"
        assert MemoryStage.EPISODIC == "episodic"
        assert MemoryStage.SEMANTIC == "semantic"

    def test_stm_decays_fastest(self) -> None:
        """STM should have the highest decay multiplier."""
        assert get_decay_multiplier(MemoryStage.SHORT_TERM) == 5.0

    def test_semantic_decays_slowest(self) -> None:
        """SEMANTIC should have the lowest decay multiplier."""
        assert get_decay_multiplier(MemoryStage.SEMANTIC) == 0.3

    def test_decay_multiplier_ordering(self) -> None:
        """Decay multipliers should decrease with consolidation."""
        stm = get_decay_multiplier(MemoryStage.SHORT_TERM)
        working = get_decay_multiplier(MemoryStage.WORKING)
        episodic = get_decay_multiplier(MemoryStage.EPISODIC)
        semantic = get_decay_multiplier(MemoryStage.SEMANTIC)
        assert stm > working > episodic > semantic


class TestMaturationRecord:
    """Tests for MaturationRecord creation and mutations."""

    def test_default_stage_is_stm(self) -> None:
        """New record should start as SHORT_TERM."""
        record = MaturationRecord(fiber_id="f1", brain_id="b1")
        assert record.stage == MemoryStage.SHORT_TERM
        assert record.rehearsal_count == 0

    def test_rehearse_increments_count(self) -> None:
        """rehearse() should increment count and add timestamp."""
        record = MaturationRecord(fiber_id="f1", brain_id="b1")
        t = datetime(2026, 1, 1, 12, 0, 0)
        rehearsed = record.rehearse(now=t)
        assert rehearsed.rehearsal_count == 1
        assert len(rehearsed.reinforcement_timestamps) == 1
        assert rehearsed.reinforcement_timestamps[0] == t.isoformat()

    def test_rehearse_immutable(self) -> None:
        """rehearse() should not mutate the original record."""
        original = MaturationRecord(fiber_id="f1", brain_id="b1")
        original.rehearse()
        assert original.rehearsal_count == 0

    def test_advance_stage(self) -> None:
        """advance_stage() should create new record in new stage."""
        record = MaturationRecord(fiber_id="f1", brain_id="b1")
        t = datetime(2026, 1, 1, 12, 0, 0)
        advanced = record.advance_stage(MemoryStage.WORKING, now=t)
        assert advanced.stage == MemoryStage.WORKING
        assert advanced.stage_entered_at == t

    def test_distinct_reinforcement_days(self) -> None:
        """Should count distinct calendar days."""
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            reinforcement_timestamps=(
                "2026-01-01T10:00:00",
                "2026-01-01T14:00:00",  # same day
                "2026-01-03T10:00:00",
                "2026-01-05T10:00:00",
            ),
        )
        assert record.distinct_reinforcement_days == 3

    def test_decay_multiplier_property(self) -> None:
        """decay_multiplier property should return stage-specific value."""
        stm = MaturationRecord(fiber_id="f1", brain_id="b1")
        assert stm.decay_multiplier == 5.0

        semantic = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.SEMANTIC,
        )
        assert semantic.decay_multiplier == 0.3

    def test_serialization_roundtrip(self) -> None:
        """to_dict/from_dict should produce equivalent records."""
        original = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=datetime(2026, 1, 1, 12, 0, 0),
            rehearsal_count=5,
            reinforcement_timestamps=("2026-01-01T10:00:00", "2026-01-03T10:00:00"),
        )
        d = original.to_dict()
        restored = MaturationRecord.from_dict(d)
        assert restored.fiber_id == original.fiber_id
        assert restored.brain_id == original.brain_id
        assert restored.stage == original.stage
        assert restored.rehearsal_count == original.rehearsal_count
        assert restored.reinforcement_timestamps == original.reinforcement_timestamps


class TestStageTransitions:
    """Tests for automatic stage transition logic."""

    def test_stm_to_working(self) -> None:
        """STM should advance to WORKING after 30 minutes."""
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.SHORT_TERM,
            stage_entered_at=t0,
        )
        t1 = t0 + timedelta(minutes=31)
        advanced = compute_stage_transition(record, now=t1)
        assert advanced.stage == MemoryStage.WORKING

    def test_stm_stays_if_too_recent(self) -> None:
        """STM should stay if less than 30 minutes old."""
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.SHORT_TERM,
            stage_entered_at=t0,
        )
        t1 = t0 + timedelta(minutes=10)
        result = compute_stage_transition(record, now=t1)
        assert result.stage == MemoryStage.SHORT_TERM

    def test_working_to_episodic(self) -> None:
        """WORKING should advance to EPISODIC after 4 hours."""
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.WORKING,
            stage_entered_at=t0,
        )
        t1 = t0 + timedelta(hours=5)
        advanced = compute_stage_transition(record, now=t1)
        assert advanced.stage == MemoryStage.EPISODIC

    def test_episodic_to_semantic_with_spacing(self) -> None:
        """EPISODIC should advance to SEMANTIC after 7 days with 3 distinct reinforcement days."""
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=t0,
            rehearsal_count=5,
            reinforcement_timestamps=(
                "2026-01-02T10:00:00",
                "2026-01-04T10:00:00",
                "2026-01-06T10:00:00",
            ),
        )
        t1 = t0 + timedelta(days=8)
        advanced = compute_stage_transition(record, now=t1)
        assert advanced.stage == MemoryStage.SEMANTIC

    def test_episodic_stays_without_spacing(self) -> None:
        """EPISODIC should NOT advance without 3 distinct reinforcement days."""
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=t0,
            rehearsal_count=5,
            reinforcement_timestamps=(
                "2026-01-02T10:00:00",
                "2026-01-02T14:00:00",  # same day
            ),
        )
        t1 = t0 + timedelta(days=8)
        result = compute_stage_transition(record, now=t1)
        assert result.stage == MemoryStage.EPISODIC  # stays

    def test_episodic_stays_if_too_recent(self) -> None:
        """EPISODIC should NOT advance before 7 days even with spacing."""
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=t0,
            rehearsal_count=5,
            reinforcement_timestamps=(
                "2026-01-02T10:00:00",
                "2026-01-04T10:00:00",
                "2026-01-06T10:00:00",
            ),
        )
        t1 = t0 + timedelta(days=5)  # only 5 days
        result = compute_stage_transition(record, now=t1)
        assert result.stage == MemoryStage.EPISODIC

    def test_semantic_stays(self) -> None:
        """SEMANTIC is the final stage — should not advance further."""
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.SEMANTIC,
            stage_entered_at=t0,
        )
        t1 = t0 + timedelta(days=100)
        result = compute_stage_transition(record, now=t1)
        assert result.stage == MemoryStage.SEMANTIC
