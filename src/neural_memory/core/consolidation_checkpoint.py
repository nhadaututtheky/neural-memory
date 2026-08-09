"""Per-strategy consolidation checkpoints — independent of change_log.synced."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from neural_memory.utils.timeutils import utcnow


@dataclass(frozen=True)
class ConsolidationCheckpoint:
    """High-watermark for one consolidation strategy on one brain.

    ``last_sequence`` is a ``change_log.id`` value. The next incremental run
    processes changes with ``id > last_sequence``. Sync's ``synced`` flag is
    never read or written by consolidation.
    """

    brain_id: str
    strategy: str
    last_sequence: int
    updated_at: datetime

    @classmethod
    def create(
        cls,
        *,
        brain_id: str,
        strategy: str,
        last_sequence: int = 0,
        updated_at: datetime | None = None,
    ) -> ConsolidationCheckpoint:
        if last_sequence < 0:
            raise ValueError(f"last_sequence must be >= 0, got {last_sequence}")
        strat = (strategy or "").strip().lower()
        if not strat:
            raise ValueError("strategy must be non-empty")
        return cls(
            brain_id=brain_id,
            strategy=strat,
            last_sequence=int(last_sequence),
            updated_at=updated_at or utcnow(),
        )
