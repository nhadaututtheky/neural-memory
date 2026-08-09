"""Per-strategy consolidation checkpoint storage mixin."""

from __future__ import annotations

import logging
from datetime import datetime

from neural_memory.core.consolidation_checkpoint import ConsolidationCheckpoint
from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _safe_parse_dt(val: object) -> datetime:
    if isinstance(val, datetime):
        if val.tzinfo is not None:
            return val.replace(tzinfo=None)
        return val
    try:
        parsed = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed.replace(tzinfo=None)
        return parsed
    except (TypeError, ValueError):
        return utcnow()


class ConsolidationCheckpointsMixin:
    """get/save/reset consolidation strategy checkpoints."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def get_consolidation_checkpoint(
        self,
        strategy: str,
    ) -> ConsolidationCheckpoint | None:
        d = self._dialect
        brain_id = self._get_brain_id()
        strat = (strategy or "").strip().lower()
        if not strat:
            return None
        row = await d.fetch_one(
            f"""SELECT brain_id, strategy, last_sequence, updated_at
                FROM consolidation_checkpoints
                WHERE brain_id = {d.ph(1)} AND strategy = {d.ph(2)}""",
            [brain_id, strat],
        )
        if not row:
            return None
        return ConsolidationCheckpoint(
            brain_id=str(row["brain_id"]),
            strategy=str(row["strategy"]),
            last_sequence=int(row["last_sequence"] or 0),
            updated_at=_safe_parse_dt(row.get("updated_at")),
        )

    async def save_consolidation_checkpoint(
        self,
        checkpoint: ConsolidationCheckpoint,
    ) -> ConsolidationCheckpoint:
        """Upsert checkpoint. Rejects negative or backward sequence moves."""
        if checkpoint.last_sequence < 0:
            raise ValueError(f"last_sequence must be >= 0, got {checkpoint.last_sequence}")
        d = self._dialect
        brain_id = checkpoint.brain_id or self._get_brain_id()
        strat = checkpoint.strategy.strip().lower()
        if not strat:
            raise ValueError("strategy must be non-empty")

        existing = await self.get_consolidation_checkpoint(strat)
        if existing is not None and checkpoint.last_sequence < existing.last_sequence:
            raise ValueError(
                f"Refusing backward checkpoint for {strat}: "
                f"{checkpoint.last_sequence} < {existing.last_sequence}"
            )

        now = utcnow()
        await d.execute(
            f"""INSERT INTO consolidation_checkpoints
                (brain_id, strategy, last_sequence, updated_at)
                VALUES ({d.phs(4)})
                ON CONFLICT(brain_id, strategy) DO UPDATE SET
                  last_sequence = excluded.last_sequence,
                  updated_at = excluded.updated_at""",
            [
                brain_id,
                strat,
                int(checkpoint.last_sequence),
                d.serialize_dt(now),
            ],
        )
        return ConsolidationCheckpoint(
            brain_id=brain_id,
            strategy=strat,
            last_sequence=int(checkpoint.last_sequence),
            updated_at=now,
        )

    async def reset_consolidation_checkpoint(
        self,
        strategy: str | None = None,
        *,
        audit_reason: str = "",
    ) -> int:
        """Reset one strategy (or all) to sequence 0. Returns rows affected."""
        d = self._dialect
        brain_id = self._get_brain_id()
        if strategy:
            strat = strategy.strip().lower()
            logger.info(
                "Reset consolidation checkpoint brain=%s strategy=%s reason=%s",
                brain_id[:8] if brain_id else "?",
                strat,
                (audit_reason or "explicit")[:200],
            )
            return await d.execute_count(
                f"""DELETE FROM consolidation_checkpoints
                    WHERE brain_id = {d.ph(1)} AND strategy = {d.ph(2)}""",
                [brain_id, strat],
            )
        logger.info(
            "Reset ALL consolidation checkpoints brain=%s reason=%s",
            brain_id[:8] if brain_id else "?",
            (audit_reason or "explicit")[:200],
        )
        return await d.execute_count(
            f"""DELETE FROM consolidation_checkpoints
                WHERE brain_id = {d.ph(1)}""",
            [brain_id],
        )

    async def list_consolidation_checkpoints(self) -> list[ConsolidationCheckpoint]:
        d = self._dialect
        brain_id = self._get_brain_id()
        rows = await d.fetch_all(
            f"""SELECT brain_id, strategy, last_sequence, updated_at
                FROM consolidation_checkpoints
                WHERE brain_id = {d.ph(1)}
                ORDER BY strategy ASC""",
            [brain_id],
        )
        return [
            ConsolidationCheckpoint(
                brain_id=str(r["brain_id"]),
                strategy=str(r["strategy"]),
                last_sequence=int(r["last_sequence"] or 0),
                updated_at=_safe_parse_dt(r.get("updated_at")),
            )
            for r in rows
        ]
