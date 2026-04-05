"""Time-travel queries — reconstruct memory state at a past timestamp.

Given an `as_of` datetime, determines what maturation stage a fiber was in
at that point by examining stage_entered_at timestamps in the maturation record.
"""

from __future__ import annotations

from datetime import datetime

from neural_memory.engine.memory_stages import MemoryStage

# Stage progression order — used to walk back to previous stage
_STAGE_ORDER: list[MemoryStage] = [
    MemoryStage.SHORT_TERM,
    MemoryStage.WORKING,
    MemoryStage.EPISODIC,
    MemoryStage.SEMANTIC,
]


def reconstruct_stage(
    current_stage: MemoryStage,
    stage_entered_at: datetime | None,
    as_of: datetime,
) -> MemoryStage:
    """Reconstruct what maturation stage a fiber was in at `as_of`.

    Logic:
    - If stage_entered_at <= as_of, the fiber had already reached current_stage.
    - If stage_entered_at > as_of, the fiber hadn't reached this stage yet —
      return the previous stage in the progression.
    - If no stage_entered_at (legacy data), assume current stage is valid.

    Args:
        current_stage: The fiber's current maturation stage.
        stage_entered_at: When the fiber entered the current stage.
        as_of: The point-in-time to reconstruct state for.

    Returns:
        The stage the fiber was most likely in at `as_of`.
    """
    if stage_entered_at is None:
        return current_stage

    if stage_entered_at <= as_of:
        return current_stage

    # Current stage was entered AFTER as_of — walk back one stage
    try:
        idx = _STAGE_ORDER.index(current_stage)
    except ValueError:
        return current_stage

    if idx <= 0:
        return MemoryStage.SHORT_TERM

    return _STAGE_ORDER[idx - 1]
