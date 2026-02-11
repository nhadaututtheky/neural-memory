"""SQLite mixin for co-activation event storage."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite


class SQLiteCoActivationMixin:
    """Co-activation event persistence for SQLiteStorage.

    Stores individual co-activation events with canonical pair ordering
    (neuron_a < neuron_b) for consistent aggregation.
    """

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def record_co_activation(
        self,
        neuron_a: str,
        neuron_b: str,
        binding_strength: float,
        source_anchor: str | None = None,
    ) -> str:
        """Record a co-activation event between two neurons."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        event_id = str(uuid4())

        # Canonical ordering: a < b
        a, b = (neuron_a, neuron_b) if neuron_a < neuron_b else (neuron_b, neuron_a)

        await conn.execute(
            """INSERT INTO co_activation_events
               (id, brain_id, neuron_a, neuron_b, binding_strength, source_anchor, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                brain_id,
                a,
                b,
                binding_strength,
                source_anchor,
                utcnow().isoformat(),
            ),
        )
        await conn.commit()
        return event_id

    async def get_co_activation_counts(
        self,
        since: datetime | None = None,
        min_count: int = 1,
    ) -> list[tuple[str, str, int, float]]:
        """Get aggregated co-activation counts for neuron pairs."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        if since is not None:
            query = """
                SELECT neuron_a, neuron_b, COUNT(*) as cnt, AVG(binding_strength) as avg_bs
                FROM co_activation_events
                WHERE brain_id = ? AND created_at >= ?
                GROUP BY neuron_a, neuron_b
                HAVING cnt >= ?
                ORDER BY cnt DESC
            """
            params = (brain_id, since.isoformat(), min_count)
        else:
            query = """
                SELECT neuron_a, neuron_b, COUNT(*) as cnt, AVG(binding_strength) as avg_bs
                FROM co_activation_events
                WHERE brain_id = ?
                GROUP BY neuron_a, neuron_b
                HAVING cnt >= ?
                ORDER BY cnt DESC
            """
            params = (brain_id, min_count)  # type: ignore[assignment]

        results: list[tuple[str, str, int, float]] = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                results.append((row["neuron_a"], row["neuron_b"], row["cnt"], row["avg_bs"]))

        return results

    async def prune_co_activations(self, older_than: datetime) -> int:
        """Remove co-activation events older than the given time."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM co_activation_events WHERE brain_id = ? AND created_at < ?",
            (brain_id, older_than.isoformat()),
        )
        await conn.commit()
        return int(cursor.rowcount)
