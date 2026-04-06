"""Tests for consolidation Phase 4: maturation fast-track and config knobs."""

from __future__ import annotations

from datetime import timedelta

from neural_memory.engine.consolidation import ConsolidationConfig
from neural_memory.engine.memory_stages import (
    MaturationRecord,
    MemoryStage,
    compute_stage_transition,
)
from neural_memory.utils.timeutils import utcnow


class TestMaturationFastTrack:
    """High-recall memories advance episodic→semantic faster."""

    def test_fast_track_with_10_rehearsals_after_1_day(self) -> None:
        """Memory with 10+ rehearsals should advance after 1 day (not 3)."""
        now = utcnow()
        entered = now - timedelta(days=1, hours=1)  # 1 day + 1 hour ago

        # Build reinforcement timestamps across 3+ 2h-windows
        timestamps = tuple((entered + timedelta(hours=i * 3)).isoformat() for i in range(10))

        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=10,
            reinforcement_timestamps=timestamps,
        )

        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.SEMANTIC

    def test_no_fast_track_with_9_rehearsals(self) -> None:
        """Memory with 9 rehearsals should NOT fast-track (needs 10)."""
        now = utcnow()
        entered = now - timedelta(days=1, hours=1)

        timestamps = tuple((entered + timedelta(hours=i * 3)).isoformat() for i in range(9))

        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=9,
            reinforcement_timestamps=timestamps,
        )

        result = compute_stage_transition(record, now=now)
        # 9 rehearsals, only 1 day — neither fast-track (needs 10) nor
        # classic (needs 3 days) applies
        assert result.stage == MemoryStage.EPISODIC

    def test_fast_track_still_needs_spacing(self) -> None:
        """Even with 10 rehearsals, still needs temporal spacing for promotion."""
        now = utcnow()
        entered = now - timedelta(days=1, hours=1)

        # All timestamps in the same 2h window — no temporal spread
        timestamps = tuple((entered + timedelta(minutes=i)).isoformat() for i in range(10))

        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=10,
            reinforcement_timestamps=timestamps,
        )

        result = compute_stage_transition(record, now=now)
        # Has 10 rehearsals but all in same window → no spacing → no promotion
        assert result.stage == MemoryStage.EPISODIC

    def test_classic_path_unaffected(self) -> None:
        """Classic 3-day path still works normally (no regression)."""
        now = utcnow()
        entered = now - timedelta(days=4)

        # 2 rehearsals on distinct days
        timestamps = (
            (entered + timedelta(days=1)).isoformat(),
            (entered + timedelta(days=2)).isoformat(),
        )

        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=2,
            reinforcement_timestamps=timestamps,
        )

        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.SEMANTIC

    def test_custom_fast_track_params(self) -> None:
        """Custom fast_track_rehearsals and time work."""
        now = utcnow()
        entered = now - timedelta(hours=13)  # 13 hours ago

        timestamps = tuple((entered + timedelta(hours=i * 3)).isoformat() for i in range(5))

        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=5,
            reinforcement_timestamps=timestamps,
        )

        # Custom: 5 rehearsals needed, 0.5 day time
        result = compute_stage_transition(
            record, now=now, fast_track_rehearsals=5, fast_track_time_days=0.5
        )
        assert result.stage == MemoryStage.SEMANTIC


class TestConsolidationConfigKnobs:
    """Verify new config fields exist with correct defaults."""

    def test_default_prune_semantic_factor(self) -> None:
        config = ConsolidationConfig()
        assert config.prune_semantic_factor == 0.5

    def test_default_bridge_weight_floor(self) -> None:
        config = ConsolidationConfig()
        assert config.bridge_weight_floor == 0.01

    def test_default_surface_regen_prune_threshold(self) -> None:
        config = ConsolidationConfig()
        assert config.surface_regen_prune_threshold == 10

    def test_default_maturation_fast_track_rehearsals(self) -> None:
        config = ConsolidationConfig()
        assert config.maturation_fast_track_rehearsals == 10

    def test_default_maturation_fast_track_time_days(self) -> None:
        config = ConsolidationConfig()
        assert config.maturation_fast_track_time_days == 1.0

    def test_custom_config_values(self) -> None:
        config = ConsolidationConfig(
            prune_semantic_factor=0.3,
            bridge_weight_floor=0.02,
            surface_regen_prune_threshold=5,
            maturation_fast_track_rehearsals=8,
            maturation_fast_track_time_days=0.5,
        )
        assert config.prune_semantic_factor == 0.3
        assert config.bridge_weight_floor == 0.02
        assert config.surface_regen_prune_threshold == 5
        assert config.maturation_fast_track_rehearsals == 8
        assert config.maturation_fast_track_time_days == 0.5


class TestFastTrackSpacingIsolation:
    """Fast-track reduces time gate but does NOT bypass spacing requirement."""

    def test_fast_track_no_spacing_stays_episodic(self) -> None:
        """10+ rehearsals but only 1 distinct window → should NOT promote."""
        now = utcnow()
        entered = now - timedelta(days=1, hours=1)

        # All 10 timestamps in same 2h window AND same day → 1 window, 1 day
        timestamps = tuple((entered + timedelta(minutes=i * 5)).isoformat() for i in range(10))

        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=10,
            reinforcement_timestamps=timestamps,
        )

        result = compute_stage_transition(record, now=now)
        # Time gate satisfied (1 day with fast-track) but spacing NOT met
        assert result.stage == MemoryStage.EPISODIC

    def test_boundary_10_vs_9_rehearsals(self) -> None:
        """Exactly 10 rehearsals should fast-track; 9 should not."""
        now = utcnow()
        entered = now - timedelta(days=1, hours=1)

        # Spread across 3+ windows so spacing is satisfied
        timestamps_10 = tuple((entered + timedelta(hours=i * 3)).isoformat() for i in range(10))
        timestamps_9 = timestamps_10[:9]

        record_10 = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=10,
            reinforcement_timestamps=timestamps_10,
        )
        record_9 = MaturationRecord(
            fiber_id="f2",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=9,
            reinforcement_timestamps=timestamps_9,
        )

        assert compute_stage_transition(record_10, now=now).stage == MemoryStage.SEMANTIC
        assert compute_stage_transition(record_9, now=now).stage == MemoryStage.EPISODIC
