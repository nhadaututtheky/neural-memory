"""Co-activation event storage mixin — dialect-agnostic."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow


class CoActivationMixin:
    """Co-activation event persistence for SQLStorage.

    Stores individual co-activation events with canonical pair ordering
    (neuron_a < neuron_b) for consistent aggregation.
    """

    _dialect: Dialect

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
        d = self._dialect
        brain_id = self._get_brain_id()
        event_id = str(uuid4())

        # Canonical ordering: a < b
        a, b = (neuron_a, neuron_b) if neuron_a < neuron_b else (neuron_b, neuron_a)

        await d.execute(
            f"""INSERT INTO co_activation_events
               (id, brain_id, neuron_a, neuron_b, binding_strength, source_anchor, created_at)
               VALUES ({d.phs(7)})""",
            [
                event_id,
                brain_id,
                a,
                b,
                binding_strength,
                source_anchor,
                d.serialize_dt(utcnow()),
            ],
        )
        return event_id

    async def get_co_activation_counts(
        self,
        since: datetime | None = None,
        min_count: int = 1,
    ) -> list[tuple[str, str, int, float]]:
        """Get aggregated co-activation counts for neuron pairs."""
        d = self._dialect
        brain_id = self._get_brain_id()

        if since is not None:
            rows = await d.fetch_all(
                f"""
                SELECT neuron_a, neuron_b, COUNT(*) as cnt, AVG(binding_strength) as avg_bs
                FROM co_activation_events
                WHERE brain_id = {d.ph(1)} AND created_at >= {d.ph(2)}
                GROUP BY neuron_a, neuron_b
                HAVING COUNT(*) >= {d.ph(3)}
                ORDER BY cnt DESC
                LIMIT 10000
                """,
                [brain_id, d.serialize_dt(since), min_count],
            )
        else:
            rows = await d.fetch_all(
                f"""
                SELECT neuron_a, neuron_b, COUNT(*) as cnt, AVG(binding_strength) as avg_bs
                FROM co_activation_events
                WHERE brain_id = {d.ph(1)}
                GROUP BY neuron_a, neuron_b
                HAVING COUNT(*) >= {d.ph(2)}
                ORDER BY cnt DESC
                LIMIT 10000
                """,
                [brain_id, min_count],
            )

        return [(r["neuron_a"], r["neuron_b"], r["cnt"], r["avg_bs"]) for r in rows]

    async def prune_co_activations(self, older_than: datetime) -> int:
        """Remove co-activation events older than the given time."""
        d = self._dialect
        brain_id = self._get_brain_id()

        return await d.execute_count(
            f"DELETE FROM co_activation_events WHERE brain_id = {d.ph(1)} AND created_at < {d.ph(2)}",
            [brain_id, d.serialize_dt(older_than)],
        )
