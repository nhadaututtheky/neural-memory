"""Memory maturation lifecycle — STM → Working → Episodic → Semantic.

Implements biologically-inspired memory consolidation stages:
- SHORT_TERM: First 30 minutes, fragile, decays 5x faster
- WORKING: 30 min to 4 hours, still volatile, decays 2x faster
- EPISODIC: 4 hours to 7 days, normal decay
- SEMANTIC: 7+ days with spacing effect, resistant to forgetting (0.3x decay)

The spacing effect requires reinforcement across 3+ distinct days
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
_EPISODIC_TO_SEMANTIC = timedelta(days=7)

# Spacing effect: minimum distinct days of reinforcement for semantic promotion
_MIN_DISTINCT_DAYS = 3


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
    reinforcement_timestamps: list[str] = field(default_factory=list)

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
            reinforcement_timestamps=[
                *self.reinforcement_timestamps,
                now.isoformat(),
            ],
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
            reinforcement_timestamps=list(self.reinforcement_timestamps),
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
            "reinforcement_timestamps": json.dumps(self.reinforcement_timestamps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MaturationRecord:
        """Deserialize from a dict."""
        timestamps_raw = data.get("reinforcement_timestamps", "[]")
        if isinstance(timestamps_raw, str):
            timestamps = json.loads(timestamps_raw)
        else:
            timestamps = list(timestamps_raw) if timestamps_raw else []  # type: ignore[call-overload]

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
) -> MaturationRecord:
    """Compute if a maturation record should advance to the next stage.

    Stage transition rules:
    - STM → Working: time > 30 minutes
    - Working → Episodic: time > 4 hours
    - Episodic → Semantic: time > 7 days AND 3+ distinct reinforcement days

    Args:
        record: Current maturation record
        now: Reference time for transition checks

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
        if (
            time_in_stage >= _EPISODIC_TO_SEMANTIC
            and record.distinct_reinforcement_days >= _MIN_DISTINCT_DAYS
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
