"""SQLite mixin for retrieval sufficiency calibration persistence."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Cap per brain to prevent unbounded growth
_MAX_RECORDS_PER_BRAIN = 10_000


class SQLiteCalibrationMixin:
    """Mixin providing CRUD for the retrieval_calibration table."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def save_calibration_record(
        self,
        gate: str,
        predicted_sufficient: bool,
        actual_confidence: float,
        actual_fibers: int,
        query_intent: str = "",
        metrics_json: dict[str, Any] | None = None,
    ) -> None:
        """Insert a calibration feedback record."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        await conn.execute(
            """INSERT INTO retrieval_calibration
               (brain_id, gate, predicted_sufficient, actual_confidence,
                actual_fibers, query_intent, metrics_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                brain_id,
                gate,
                1 if predicted_sufficient else 0,
                actual_confidence,
                actual_fibers,
                query_intent,
                json.dumps(metrics_json or {}),
                utcnow().isoformat(),
            ),
        )
        await conn.commit()

    async def get_recent_calibration(
        self,
        gate: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch recent calibration records, optionally filtered by gate."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 200)

        if gate:
            cursor = await conn.execute(
                """SELECT * FROM retrieval_calibration
                   WHERE brain_id = ? AND gate = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (brain_id, gate, capped_limit),
            )
        else:
            cursor = await conn.execute(
                """SELECT * FROM retrieval_calibration
                   WHERE brain_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (brain_id, capped_limit),
            )

        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [dict(zip(col_names, r, strict=False)) for r in rows]

    async def prune_old_calibration(self, keep_days: int = 90) -> int:
        """Delete calibration records older than keep_days. Returns count deleted."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        from datetime import timedelta

        cutoff = (utcnow() - timedelta(days=keep_days)).isoformat()

        cursor = await conn.execute(
            "DELETE FROM retrieval_calibration WHERE brain_id = ? AND created_at < ?",
            (brain_id, cutoff),
        )
        await conn.commit()
        return cursor.rowcount

    async def cap_calibration_records(self) -> int:
        """Enforce max record limit per brain. Returns count deleted."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT COUNT(*) FROM retrieval_calibration WHERE brain_id = ?",
            (brain_id,),
        )
        row = await cursor.fetchone()
        count = row[0] if row else 0

        if count <= _MAX_RECORDS_PER_BRAIN:
            return 0

        excess = count - _MAX_RECORDS_PER_BRAIN
        cursor = await conn.execute(
            """DELETE FROM retrieval_calibration WHERE id IN (
                SELECT id FROM retrieval_calibration
                WHERE brain_id = ?
                ORDER BY created_at ASC LIMIT ?
            )""",
            (brain_id, excess),
        )
        await conn.commit()
        return cursor.rowcount
