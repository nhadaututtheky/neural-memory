"""Memory maturation lifecycle — STM → Working → Episodic → Semantic.

Implements biologically-inspired memory consolidation stages:
- SHORT_TERM: First 30 minutes, fragile, decays 5x faster
- WORKING: 30 min to 4 hours, still volatile, decays 2x faster
- EPISODIC: 4 hours to 3 days, normal decay
- SEMANTIC: 3+ days with spacing effect, resistant to forgetting (0.3x decay)

The spacing effect requires reinforcement across 2+ distinct days
for promotion from EPISODIC to SEMANTIC, modeling how spaced
repetition strengthens long-term memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from neural_memory.utils.timeutils import utcnow


class MemoryStage(StrEnum):
    """Memory maturation stages in order of consolidation."""

    SHORT_TERM = "stm"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


# Decay multipliers per stage
STAGE_DECAY_MULTIPLIERS: dict[MemoryStage, float] = {
    MemoryStage.SHORT_TERM: 5.0,
    MemoryStage.WORKING: 2.0,
    MemoryStage.EPISODIC: 1.0,
    MemoryStage.SEMANTIC: 0.3,
}

# Time thresholds for automatic stage transitions
_STM_TO_WORKING = timedelta(minutes=30)
_WORKING_TO_EPISODIC = timedelta(hours=4)
_EPISODIC_TO_SEMANTIC = timedelta(days=3)

# Spacing effect: minimum distinct days of reinforcement for semantic promotion
_MIN_DISTINCT_DAYS = 2

# Alternative path for AI agents: high rehearsal count with temporal spread.
# Agents recall the same memory frequently but rarely across many distinct
# calendar days. This path requires 5+ rehearsals spread across 3+ distinct
# 2-hour windows, with the time gate reduced to 3 days.
_MIN_REHEARSAL_COUNT = 5
_MIN_DISTINCT_WINDOWS = 3
_WINDOW_SIZE_HOURS = 2


@dataclass(frozen=True)
class MaturationRecord:
    """Tracks a fiber's maturation through memory stages.

    Attributes:
        fiber_id: ID of the fiber being tracked
        brain_id: ID of the owning brain
        stage: Current memory stage
        stage_entered_at: When the current stage was entered
        rehearsal_count: Total number of rehearsal events
        reinforcement_timestamps: ISO timestamps of reinforcement events
    """

    fiber_id: str
    brain_id: str
    stage: MemoryStage = MemoryStage.SHORT_TERM
    stage_entered_at: datetime = field(default_factory=utcnow)
    rehearsal_count: int = 0
    reinforcement_timestamps: tuple[str, ...] = field(default_factory=tuple)

    def rehearse(self, now: datetime | None = None) -> MaturationRecord:
        """Record a rehearsal event.

        Args:
            now: Timestamp of the rehearsal (default: utcnow)

        Returns:
            New MaturationRecord with updated rehearsal data
        """
        now = now or utcnow()
        return MaturationRecord(
            fiber_id=self.fiber_id,
            brain_id=self.brain_id,
            stage=self.stage,
            stage_entered_at=self.stage_entered_at,
            rehearsal_count=self.rehearsal_count + 1,
            reinforcement_timestamps=(
                *self.reinforcement_timestamps,
                now.isoformat(),
            ),
        )

    def advance_stage(
        self,
        new_stage: MemoryStage,
        now: datetime | None = None,
    ) -> MaturationRecord:
        """Advance to a new memory stage.

        Args:
            new_stage: Target stage
            now: When the transition occurs

        Returns:
            New MaturationRecord in the new stage
        """
        now = now or utcnow()
        return MaturationRecord(
            fiber_id=self.fiber_id,
            brain_id=self.brain_id,
            stage=new_stage,
            stage_entered_at=now,
            rehearsal_count=self.rehearsal_count,
            reinforcement_timestamps=tuple(self.reinforcement_timestamps),
        )

    @property
    def distinct_reinforcement_days(self) -> int:
        """Count distinct calendar days with reinforcement events."""
        days: set[str] = set()
        for ts_str in self.reinforcement_timestamps:
            try:
                dt = datetime.fromisoformat(ts_str)
                days.add(dt.strftime("%Y-%m-%d"))
            except ValueError:
                continue
        return len(days)

    @property
    def distinct_reinforcement_windows(self) -> int:
        """Count distinct 2-hour windows with reinforcement events.

        Groups timestamps into 2-hour buckets (0-1, 2-3, 4-5, ...) per day
        to measure temporal spread without requiring distinct calendar days.
        """
        windows: set[str] = set()
        for ts_str in self.reinforcement_timestamps:
            try:
                dt = datetime.fromisoformat(ts_str)
                bucket = dt.hour // _WINDOW_SIZE_HOURS
                windows.add(f"{dt.strftime('%Y-%m-%d')}:{bucket}")
            except ValueError:
                continue
        return len(windows)

    @property
    def decay_multiplier(self) -> float:
        """Get the decay rate multiplier for the current stage."""
        return STAGE_DECAY_MULTIPLIERS.get(self.stage, 1.0)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dict for storage."""
        return {
            "fiber_id": self.fiber_id,
            "brain_id": self.brain_id,
            "stage": self.stage.value,
            "stage_entered_at": self.stage_entered_at.isoformat(),
            "rehearsal_count": self.rehearsal_count,
            "reinforcement_timestamps": json.dumps(list(self.reinforcement_timestamps)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MaturationRecord:
        """Deserialize from a dict."""
        timestamps_raw = data.get("reinforcement_timestamps", "[]")
        if isinstance(timestamps_raw, str):
            timestamps = tuple(json.loads(timestamps_raw))
        else:
            timestamps = tuple(timestamps_raw) if isinstance(timestamps_raw, (list, tuple)) else ()

        stage_entered_raw = data.get("stage_entered_at", "")
        stage_entered = (
            datetime.fromisoformat(str(stage_entered_raw)) if stage_entered_raw else utcnow()
        )

        return cls(
            fiber_id=str(data["fiber_id"]),
            brain_id=str(data["brain_id"]),
            stage=MemoryStage(str(data.get("stage", "stm"))),
            stage_entered_at=stage_entered,
            rehearsal_count=int(data.get("rehearsal_count", 0)),  # type: ignore[call-overload]
            reinforcement_timestamps=timestamps,
        )


def compute_stage_transition(
    record: MaturationRecord,
    now: datetime | None = None,
    fast_track_rehearsals: int = 10,
    fast_track_time_days: float = 1.0,
) -> MaturationRecord:
    """Compute if a maturation record should advance to the next stage.

    Stage transition rules:
    - STM → Working: time > 30 minutes
    - Working → Episodic: time > 4 hours
    - Episodic → Semantic: time > 3 days AND (2+ distinct days OR 5+ rehearsals with 3+ 2h-windows)
    - Fast-track: 10+ rehearsals reduces episodic→semantic time to 1 day

    Args:
        record: Current maturation record
        now: Reference time for transition checks
        fast_track_rehearsals: Rehearsals needed for fast-track (default 10)
        fast_track_time_days: Reduced time for fast-track (default 1.0 day)

    Returns:
        New MaturationRecord (possibly in a new stage)
    """
    now = now or utcnow()
    time_in_stage = now - record.stage_entered_at

    if record.stage == MemoryStage.SHORT_TERM:
        if time_in_stage >= _STM_TO_WORKING:
            return record.advance_stage(MemoryStage.WORKING, now)

    elif record.stage == MemoryStage.WORKING:
        if time_in_stage >= _WORKING_TO_EPISODIC:
            return record.advance_stage(MemoryStage.EPISODIC, now)

    elif record.stage == MemoryStage.EPISODIC:
        # Fast-track: high-recall memories need less time
        time_threshold = _EPISODIC_TO_SEMANTIC
        if record.rehearsal_count >= fast_track_rehearsals:
            time_threshold = timedelta(days=fast_track_time_days)

        if time_in_stage >= time_threshold and (
            # Classic path: 2+ distinct calendar days (human spaced repetition)
            record.distinct_reinforcement_days >= _MIN_DISTINCT_DAYS
            # Agent path: 5+ rehearsals spread across 3+ distinct 2h windows
            or (
                record.rehearsal_count >= _MIN_REHEARSAL_COUNT
                and record.distinct_reinforcement_windows >= _MIN_DISTINCT_WINDOWS
            )
        ):
            return record.advance_stage(MemoryStage.SEMANTIC, now)

    # No transition — return unchanged
    return record


def get_decay_multiplier(stage: MemoryStage) -> float:
    """Get the decay rate multiplier for a given memory stage.

    Args:
        stage: Memory maturation stage

    Returns:
        Decay multiplier (higher = faster decay)
    """
    return STAGE_DECAY_MULTIPLIERS.get(stage, 1.0)
