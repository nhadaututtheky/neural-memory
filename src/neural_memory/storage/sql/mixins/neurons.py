"""Dialect-agnostic neuron and neuron-state CRUD mixin.

Merges SQLiteNeuronMixin and PostgresNeuronMixin into a single
implementation that delegates all SQL differences to the Dialect
abstraction.  Every placeholder uses ``d.ph(N)``, every IN clause uses
``d.in_clause()``, every datetime is serialised via ``d.serialize_dt()``,
and row parsing goes through the shared row-mapper layer.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.storage.sql.row_mappers import row_to_neuron, row_to_neuron_state
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.neuron_cache import NeuronLookupCache
    from neural_memory.storage.sql.dialect import Dialect

logger = logging.getLogger(__name__)

# Cap suggestion results
_MAX_SUGGEST = 100


# ---------------------------------------------------------------------------
# FTS helpers (SQLite FTS5 specific — guarded by ``d.supports_fts``)
# ---------------------------------------------------------------------------


def _build_fts_query(search_term: str) -> str:
    """Build an FTS5 MATCH expression from a user search string.

    Splits on whitespace, quotes each token to escape FTS5 operators
    (AND, OR, NOT, NEAR, *, etc.), and joins with implicit AND.
    Double quotes within tokens are escaped by doubling them.
    Example: 'API design' -> '"API" "design"'
    """
    tokens = search_term.split()
    if not tokens:
        return '""'
    return " ".join(f'"{token.replace(chr(34), chr(34) + chr(34))}"' for token in tokens)


def _build_fts_prefix_query(prefix: str) -> str:
    """Build FTS5 MATCH with prefix on last token.

    All tokens except the last are quoted (exact match).
    The last token is sanitized and gets a ``*`` suffix (prefix match).
    Example: ``'API des'`` -> ``'"API" des*'``
    """
    tokens = prefix.split()
    if not tokens:
        return '""'
    parts: list[str] = []
    for token in tokens[:-1]:
        escaped = token.replace(chr(34), chr(34) + chr(34))
        parts.append(f'"{escaped}"')
    last = re.sub(r"[^\w]", "", tokens[-1], flags=re.UNICODE)
    if last:
        parts.append(f"{last}*")
    return " ".join(parts) if parts else '""'


class NeuronMixin:
    """Dialect-agnostic neuron and neuron-state CRUD.

    Requires the mixin host to provide:
    - ``_dialect``: :class:`Dialect` instance
    - ``_get_brain_id() -> str``
    - ``_neuron_cache``: :class:`NeuronLookupCache`
    - ``invalidate_merkle_prefix(...)`` (from merkle mixin)
    """

    if TYPE_CHECKING:
        _dialect: Dialect
        _neuron_cache: NeuronLookupCache

        def _get_brain_id(self) -> str: ...

    # ========== Neuron Operations ==========

    async def add_neuron(self, neuron: Neuron) -> str:
        d = self._dialect
        brain_id = self._get_brain_id()

        # Strip embedding from metadata — stored in dedicated column when supported
        meta = {k: v for k, v in neuron.metadata.items() if k != "_embedding"}
        meta_json = json.dumps(meta)
        embedding = neuron.metadata.get("_embedding")

        # Duplicate check
        row = await d.fetch_one(
            f"SELECT id FROM neurons WHERE id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (neuron.id, brain_id),
        )
        if row is not None:
            raise ValueError(f"Neuron {neuron.id} already exists")

        # Build column/value lists dynamically for optional embedding
        cols = [
            "id",
            "brain_id",
            "type",
            "content",
            "metadata",
            "content_hash",
            "created_at",
            "ephemeral",
        ]
        vals: list[Any] = [
            neuron.id,
            brain_id,
            neuron.type.value,
            neuron.content,
            meta_json,
            neuron.content_hash,
            d.serialize_dt(neuron.created_at),
            1 if neuron.ephemeral else 0,
        ]
        if embedding is not None and d.supports_vector:
            cols.append("embedding")
            vals.append(embedding)

        phs = d.phs(len(cols))
        await d.execute(
            f"INSERT INTO neurons ({', '.join(cols)}) VALUES ({phs})",
            vals,
        )

        # Initialise neuron state
        await d.execute(
            f"""INSERT INTO neuron_states
               (neuron_id, brain_id, firing_threshold, refractory_period_ms,
                homeostatic_target, created_at)
               VALUES ({d.phs(6)})""",
            [neuron.id, brain_id, 0.3, 500.0, 0.5, d.serialize_dt(utcnow())],
        )

        # Cache / Merkle invalidation
        self._neuron_cache.invalidate_key(neuron.content, neuron.type.value)
        await self.invalidate_merkle_prefix("neuron", neuron.id, is_pro=True)  # type: ignore[attr-defined]
        return neuron.id

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        d = self._dialect
        brain_id = self._get_brain_id()
        row = await d.fetch_one(
            f"SELECT * FROM neurons WHERE id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (neuron_id, brain_id),
        )
        if row is None:
            return None
        return row_to_neuron(row, d)

    async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        """Fetch multiple neurons in a single SQL query."""
        if not neuron_ids:
            return {}
        d = self._dialect
        brain_id = self._get_brain_id()

        in_sql, in_params = d.in_clause(2, neuron_ids)
        rows = await d.fetch_all(
            f"SELECT * FROM neurons WHERE brain_id = {d.ph(1)} AND id {in_sql}",
            [brain_id, *in_params],
        )
        return {str(r["id"]): row_to_neuron(r, d) for r in rows}

    async def has_neuron_by_content_hash(self, content_hash: int) -> bool:
        """Check if a neuron with this content hash exists (fast indexed lookup)."""
        d = self._dialect
        row = await d.fetch_one(
            f"SELECT 1 FROM neurons WHERE content_hash = {d.ph(1)} LIMIT 1",
            (content_hash,),
        )
        return row is not None

    async def find_neurons_exact_batch(
        self,
        contents: list[str],
        type: NeuronType | None = None,
        ephemeral: bool | None = None,
    ) -> dict[str, Neuron]:
        """Find neurons by exact content for multiple contents in one query."""
        if not contents:
            return {}
        d = self._dialect
        brain_id = self._get_brain_id()

        in_sql, in_params = d.in_clause(2, contents)
        query = f"SELECT * FROM neurons WHERE brain_id = {d.ph(1)} AND content {in_sql}"
        params: list[Any] = [brain_id, *in_params]

        if type is not None:
            idx = len(params) + 1
            query += f" AND type = {d.ph(idx)}"
            params.append(type.value)

        if ephemeral is not None:
            idx = len(params) + 1
            query += f" AND ephemeral = {d.ph(idx)}"
            params.append(1 if ephemeral else 0)

        results: dict[str, Neuron] = {}
        rows = await d.fetch_all(query, params)
        for row in rows:
            neuron = row_to_neuron(row, d)
            # First match per content wins
            if neuron.content not in results:
                results[neuron.content] = neuron
        return results

    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
        offset: int = 0,
        ephemeral: bool | None = None,
    ) -> list[Neuron]:
        # Cache shortcut for exact-match lookups (most repeated pattern)
        if content_exact is not None and content_contains is None and time_range is None:
            type_val = type.value if type is not None else None
            cached = self._neuron_cache.get(content_exact, type_val)
            if cached is not None:
                return cached[offset : offset + limit]

        d = self._dialect
        brain_id = self._get_brain_id()

        # Allow larger batches for full-scan pagination
        full_scan = content_contains is None and content_exact is None
        limit = min(limit, 10000 if full_scan else 1000)

        # ------ Fast exact-content path ------
        if content_exact is not None:
            row = await d.fetch_one(
                f"SELECT * FROM neurons WHERE brain_id = {d.ph(1)} AND content = {d.ph(2)}",
                (brain_id, content_exact),
            )
            if row is None:
                return []
            if type is not None and row["type"] != type.value:
                return []
            result = [row_to_neuron(row, d)]
            # Populate cache
            type_val = type.value if type is not None else None
            self._neuron_cache.put(content_exact, type_val, result)
            return result

        # ------ FTS path ------
        use_fts = d.supports_fts and content_contains is not None

        if use_fts:
            fts_terms = _build_fts_query(content_contains)  # type: ignore[arg-type]
            from_clause, where_clause = d.fts_neuron_query(1, 2)

            query = f"SELECT n.* FROM {from_clause} WHERE {where_clause}"
            params: list[Any] = [fts_terms, brain_id]

            if type is not None:
                idx = len(params) + 1
                query += f" AND n.type = {d.ph(idx)}"
                params.append(type.value)

            if time_range is not None:
                start_dt, end_dt = time_range
                idx = len(params) + 1
                query += f" AND n.created_at >= {d.ph(idx)} AND n.created_at <= {d.ph(idx + 1)}"
                params.extend([d.serialize_dt(start_dt), d.serialize_dt(end_dt)])

            if ephemeral is not None:
                idx = len(params) + 1
                query += f" AND n.ephemeral = {d.ph(idx)}"
                params.append(1 if ephemeral else 0)

            idx = len(params) + 1
            query += f" ORDER BY fts.rank LIMIT {d.ph(idx)} OFFSET {d.ph(idx + 1)}"
            params.extend([limit, offset])
        else:
            # ------ LIKE / ILIKE fallback ------
            query = f"SELECT * FROM neurons WHERE brain_id = {d.ph(1)}"
            params = [brain_id]

            if type is not None:
                idx = len(params) + 1
                query += f" AND type = {d.ph(idx)}"
                params.append(type.value)

            if content_contains is not None:
                if d.supports_ilike:
                    # PostgreSQL: use ILIKE for case-insensitive match
                    safe = (
                        content_contains.replace("\\", "\\\\")
                        .replace("%", "\\%")
                        .replace("_", "\\_")
                    )
                    idx = len(params) + 1
                    query += f" AND content ILIKE {d.ph(idx)} ESCAPE '\\'"
                    params.append(f"%{safe}%")
                else:
                    # SQLite: plain LIKE (case-insensitive for ASCII by default)
                    escaped = (
                        content_contains.replace("\\", "\\\\")
                        .replace("%", "\\%")
                        .replace("_", "\\_")
                    )
                    idx = len(params) + 1
                    query += f" AND content LIKE {d.ph(idx)} ESCAPE '\\'"
                    params.append(f"%{escaped}%")

            if time_range is not None:
                start_dt, end_dt = time_range
                idx = len(params) + 1
                query += f" AND created_at >= {d.ph(idx)} AND created_at <= {d.ph(idx + 1)}"
                params.extend([d.serialize_dt(start_dt), d.serialize_dt(end_dt)])

            if ephemeral is not None:
                idx = len(params) + 1
                query += f" AND ephemeral = {d.ph(idx)}"
                params.append(1 if ephemeral else 0)

            idx = len(params) + 1
            query += f" ORDER BY id LIMIT {d.ph(idx)} OFFSET {d.ph(idx + 1)}"
            params.extend([limit, offset])

        rows = await d.fetch_all(query, params)
        result = [row_to_neuron(row, d) for row in rows]

        # Populate cache for exact-match queries (unreachable here but kept
        # for safety if the flow is refactored later)
        if content_exact is not None and content_contains is None and time_range is None:
            type_val = type.value if type is not None else None
            self._neuron_cache.put(content_exact, type_val, result)

        return result

    async def update_neuron(self, neuron: Neuron) -> None:
        d = self._dialect
        brain_id = self._get_brain_id()

        embedding = neuron.metadata.get("_embedding")
        meta = {k: v for k, v in neuron.metadata.items() if k != "_embedding"}
        meta_json = json.dumps(meta)

        if embedding is not None and d.supports_vector:
            count = await d.execute_count(
                f"""UPDATE neurons
                    SET type = {d.ph(1)}, content = {d.ph(2)}, metadata = {d.ph(3)},
                        content_hash = {d.ph(4)}, embedding = {d.ph(5)}
                    WHERE id = {d.ph(6)} AND brain_id = {d.ph(7)}""",
                (
                    neuron.type.value,
                    neuron.content,
                    meta_json,
                    neuron.content_hash,
                    embedding,
                    neuron.id,
                    brain_id,
                ),
            )
        else:
            count = await d.execute_count(
                f"""UPDATE neurons
                    SET type = {d.ph(1)}, content = {d.ph(2)},
                        metadata = {d.ph(3)}, content_hash = {d.ph(4)}
                    WHERE id = {d.ph(5)} AND brain_id = {d.ph(6)}""",
                (
                    neuron.type.value,
                    neuron.content,
                    meta_json,
                    neuron.content_hash,
                    neuron.id,
                    brain_id,
                ),
            )

        if count == 0:
            raise ValueError(f"Neuron {neuron.id} does not exist")

        self._neuron_cache.invalidate()

    async def delete_neuron(self, neuron_id: str) -> bool:
        d = self._dialect
        brain_id = self._get_brain_id()

        count = await d.execute_count(
            f"DELETE FROM neurons WHERE id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (neuron_id, brain_id),
        )
        self._neuron_cache.invalidate()
        if count > 0:
            await self.invalidate_merkle_prefix("neuron", neuron_id, is_pro=True)  # type: ignore[attr-defined]
        return count > 0

    async def delete_neurons_batch(self, neuron_ids: list[str]) -> int:
        """Delete multiple neurons in batched SQL statements.

        Uses chunked ``DELETE ... WHERE id IN (...)`` for efficiency.
        Returns total number of deleted rows.
        """
        if not neuron_ids:
            return 0

        d = self._dialect
        brain_id = self._get_brain_id()
        deleted = 0
        chunk_size = 500

        for start_idx in range(0, len(neuron_ids), chunk_size):
            chunk = neuron_ids[start_idx : start_idx + chunk_size]
            in_sql, in_params = d.in_clause(2, chunk)
            count = await d.execute_count(
                f"DELETE FROM neurons WHERE brain_id = {d.ph(1)} AND id {in_sql}",
                [brain_id, *in_params],
            )
            deleted += count

        self._neuron_cache.invalidate()
        return deleted

    # ========== Neuron State Operations ==========

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        d = self._dialect
        brain_id = self._get_brain_id()
        row = await d.fetch_one(
            f"SELECT * FROM neuron_states WHERE neuron_id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (neuron_id, brain_id),
        )
        if row is None:
            return None
        return row_to_neuron_state(d, row)

    async def get_neuron_states_batch(self, neuron_ids: list[str]) -> dict[str, NeuronState]:
        """Batch fetch neuron states in a single SQL query."""
        if not neuron_ids:
            return {}
        d = self._dialect
        brain_id = self._get_brain_id()

        in_sql, in_params = d.in_clause(2, neuron_ids)
        rows = await d.fetch_all(
            f"SELECT * FROM neuron_states WHERE brain_id = {d.ph(1)} AND neuron_id {in_sql}",
            [brain_id, *in_params],
        )
        return {str(r["neuron_id"]): row_to_neuron_state(d, r) for r in rows}

    async def update_neuron_state(self, state: NeuronState) -> None:
        d = self._dialect
        brain_id = self._get_brain_id()

        try:
            await d.execute(
                f"""INSERT INTO neuron_states
                   (neuron_id, brain_id, activation_level, access_frequency,
                    last_activated, decay_rate, firing_threshold, refractory_until,
                    refractory_period_ms, homeostatic_target, created_at)
                   VALUES ({d.phs(11)})
                   ON CONFLICT (brain_id, neuron_id) DO UPDATE SET
                     activation_level = EXCLUDED.activation_level,
                     access_frequency = EXCLUDED.access_frequency,
                     last_activated = EXCLUDED.last_activated,
                     decay_rate = EXCLUDED.decay_rate,
                     firing_threshold = EXCLUDED.firing_threshold,
                     refractory_until = EXCLUDED.refractory_until,
                     refractory_period_ms = EXCLUDED.refractory_period_ms,
                     homeostatic_target = EXCLUDED.homeostatic_target""",
                (
                    state.neuron_id,
                    brain_id,
                    state.activation_level,
                    state.access_frequency,
                    d.serialize_dt(state.last_activated),
                    state.decay_rate,
                    state.firing_threshold,
                    d.serialize_dt(state.refractory_until),
                    state.refractory_period_ms,
                    state.homeostatic_target,
                    d.serialize_dt(state.created_at),
                ),
            )
        except Exception:
            # Neuron may have been deleted (e.g., by consolidation pruning)
            # between state read and state write — skip silently.
            logger.debug(
                "Skipping state update for neuron %s (likely deleted)",
                state.neuron_id,
            )

    async def update_neuron_states_batch(self, states: list[NeuronState]) -> None:
        """Update multiple neuron states in one batch."""
        if not states:
            return
        d = self._dialect
        brain_id = self._get_brain_id()

        args_list = [
            (
                s.neuron_id,
                brain_id,
                s.activation_level,
                s.access_frequency,
                d.serialize_dt(s.last_activated),
                s.decay_rate,
                s.firing_threshold,
                d.serialize_dt(s.refractory_until),
                s.refractory_period_ms,
                s.homeostatic_target,
                d.serialize_dt(s.created_at),
            )
            for s in states
        ]
        try:
            await d.execute_many(
                f"""INSERT INTO neuron_states
                   (neuron_id, brain_id, activation_level, access_frequency,
                    last_activated, decay_rate, firing_threshold, refractory_until,
                    refractory_period_ms, homeostatic_target, created_at)
                   VALUES ({d.phs(11)})
                   ON CONFLICT (brain_id, neuron_id) DO UPDATE SET
                     activation_level = EXCLUDED.activation_level,
                     access_frequency = EXCLUDED.access_frequency,
                     last_activated = EXCLUDED.last_activated,
                     decay_rate = EXCLUDED.decay_rate,
                     firing_threshold = EXCLUDED.firing_threshold,
                     refractory_until = EXCLUDED.refractory_until,
                     refractory_period_ms = EXCLUDED.refractory_period_ms,
                     homeostatic_target = EXCLUDED.homeostatic_target""",
                args_list,
            )
        except Exception:
            # Neurons may have been pruned; skip silently.
            logger.debug("Batch state update partially failed (likely pruned neurons)")

    async def get_all_neuron_states(self) -> list[NeuronState]:
        """Get all neuron states for current brain."""
        d = self._dialect
        brain_id = self._get_brain_id()
        rows = await d.fetch_all(
            f"SELECT * FROM neuron_states WHERE brain_id = {d.ph(1)} LIMIT 10000",
            (brain_id,),
        )
        return [row_to_neuron_state(d, row) for row in rows]

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: NeuronType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest neurons matching a prefix, ranked by relevance + frequency."""
        limit = min(limit, _MAX_SUGGEST)
        if not prefix.strip():
            return []

        d = self._dialect
        brain_id = self._get_brain_id()

        if d.supports_fts:
            # FTS path — ranked by BM25 score + activation heuristics
            fts_expr = _build_fts_prefix_query(prefix)
            from_clause, where_clause = d.fts_neuron_query(1, 2)
            query = (
                f"SELECT n.id AS neuron_id, n.content, n.type,"
                f" COALESCE(ns.access_frequency, 0) AS access_frequency,"
                f" COALESCE(ns.activation_level, 0.0) AS activation_level,"
                f" (-fts.rank"
                f"   + COALESCE(ns.access_frequency, 0) * 0.1"
                f"   + COALESCE(ns.activation_level, 0.0) * 0.5) AS score"
                f" FROM {from_clause}"
                f" LEFT JOIN neuron_states ns"
                f"   ON ns.brain_id = n.brain_id AND ns.neuron_id = n.id"
                f" WHERE {where_clause}"
            )
            params: list[Any] = [fts_expr, brain_id]

            if type_filter is not None:
                idx = len(params) + 1
                query += f" AND n.type = {d.ph(idx)}"
                params.append(type_filter.value)

            idx = len(params) + 1
            query += f" ORDER BY score DESC LIMIT {d.ph(idx)}"
            params.append(limit)

        elif d.supports_ilike:
            # PostgreSQL non-FTS path — ILIKE prefix match
            query = (
                "SELECT n.id AS neuron_id, n.content, n.type,"
                " COALESCE(ns.access_frequency, 0) AS access_frequency,"
                " COALESCE(ns.activation_level, 0.0) AS activation_level,"
                " (COALESCE(ns.access_frequency, 0) * 0.1"
                "  + COALESCE(ns.activation_level, 0.0) * 0.5) AS score"
                " FROM neurons n"
                " LEFT JOIN neuron_states ns"
                "   ON n.brain_id = ns.brain_id AND n.id = ns.neuron_id"
                f" WHERE n.brain_id = {d.ph(1)} AND n.content ILIKE {d.ph(2)}"
            )
            params = [brain_id, f"{prefix}%"]

            if type_filter is not None:
                idx = len(params) + 1
                query += f" AND n.type = {d.ph(idx)}"
                params.append(type_filter.value)

            idx = len(params) + 1
            query += f" ORDER BY COALESCE(ns.access_frequency, 0) DESC NULLS LAST LIMIT {d.ph(idx)}"
            params.append(limit)

        else:
            # SQLite non-FTS fallback — LIKE with ESCAPE
            escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            query = (
                "SELECT n.id AS neuron_id, n.content, n.type,"
                " COALESCE(ns.access_frequency, 0) AS access_frequency,"
                " COALESCE(ns.activation_level, 0.0) AS activation_level,"
                " (COALESCE(ns.access_frequency, 0) * 0.1"
                "  + COALESCE(ns.activation_level, 0.0) * 0.5) AS score"
                " FROM neurons n"
                " LEFT JOIN neuron_states ns"
                "   ON ns.brain_id = n.brain_id AND ns.neuron_id = n.id"
                f" WHERE n.brain_id = {d.ph(1)} AND n.content LIKE {d.ph(2)} ESCAPE '\\'"
            )
            params = [brain_id, f"{escaped}%"]

            if type_filter is not None:
                idx = len(params) + 1
                query += f" AND n.type = {d.ph(idx)}"
                params.append(type_filter.value)

            idx = len(params) + 1
            query += f" ORDER BY COALESCE(ns.access_frequency, 0) DESC LIMIT {d.ph(idx)}"
            params.append(limit)

        rows = await d.fetch_all(query, params)
        return [
            {
                "neuron_id": str(row["neuron_id"]),
                "content": str(row["content"]),
                "type": str(row["type"]),
                "access_frequency": int(row["access_frequency"] or 0),
                "activation_level": float(row["activation_level"] or 0.0),
                "score": float(row["score"] or 0.0),
            }
            for row in rows
        ]

    # ========== Access Tracking ==========

    async def batch_update_last_accessed(self, neuron_ids: list[str]) -> None:
        """Update last_accessed_at for neurons in batch using a single SQL UPDATE."""
        if not neuron_ids:
            return
        d = self._dialect
        brain_id = self._get_brain_id()
        now_val = d.serialize_dt(utcnow())

        in_sql, in_params = d.in_clause(3, neuron_ids)
        await d.execute(
            f"UPDATE neurons SET last_accessed_at = {d.ph(1)}"
            f" WHERE brain_id = {d.ph(2)} AND id {in_sql}",
            [now_val, brain_id, *in_params],
        )

    # ========== Lifecycle State ==========

    async def update_neuron_lifecycle(self, neuron_id: str, lifecycle_state: str) -> None:
        """Update the lifecycle_state column for a neuron."""
        d = self._dialect
        brain_id = self._get_brain_id()
        await d.execute(
            f"UPDATE neurons SET lifecycle_state = {d.ph(1)}"
            f" WHERE id = {d.ph(2)} AND brain_id = {d.ph(3)}",
            (lifecycle_state, neuron_id, brain_id),
        )

    async def update_neuron_frozen(self, neuron_id: str, frozen: bool) -> None:
        """Set or clear the frozen flag for a neuron."""
        d = self._dialect
        brain_id = self._get_brain_id()
        await d.execute(
            f"UPDATE neurons SET frozen = {d.ph(1)} WHERE id = {d.ph(2)} AND brain_id = {d.ph(3)}",
            (1 if frozen else 0, neuron_id, brain_id),
        )

    async def update_neuron_ephemeral(self, neuron_id: str, ephemeral: bool) -> None:
        """Set or clear the ephemeral flag for a neuron."""
        d = self._dialect
        brain_id = self._get_brain_id()
        await d.execute(
            f"UPDATE neurons SET ephemeral = {d.ph(1)}"
            f" WHERE id = {d.ph(2)} AND brain_id = {d.ph(3)}",
            (1 if ephemeral else 0, neuron_id, brain_id),
        )
        self._neuron_cache.invalidate()

    async def update_neurons_ephemeral_batch(
        self,
        neuron_ids: list[str],
        ephemeral: bool,
    ) -> None:
        """Batch-set ephemeral flag for multiple neurons."""
        if not neuron_ids:
            return
        d = self._dialect
        brain_id = self._get_brain_id()
        chunk_size = 500

        for start_idx in range(0, len(neuron_ids), chunk_size):
            chunk = neuron_ids[start_idx : start_idx + chunk_size]
            in_sql, in_params = d.in_clause(3, chunk)
            await d.execute(
                f"UPDATE neurons SET ephemeral = {d.ph(1)}"
                f" WHERE brain_id = {d.ph(2)} AND id {in_sql}",
                [1 if ephemeral else 0, brain_id, *in_params],
            )
        self._neuron_cache.invalidate()

    async def get_lifecycle_distribution(self) -> dict[str, int]:
        """Return count of neurons by lifecycle_state for the current brain."""
        d = self._dialect
        brain_id = self._get_brain_id()
        rows = await d.fetch_all(
            "SELECT COALESCE(lifecycle_state, 'active') AS state, COUNT(*) AS cnt"
            f" FROM neurons WHERE brain_id = {d.ph(1)}"
            " GROUP BY lifecycle_state",
            (brain_id,),
        )
        return {str(row["state"]): int(row["cnt"]) for row in rows}

    # ========== Neuron Snapshots (Tier 3-4 recovery) ==========

    async def save_neuron_snapshot(
        self,
        neuron_id: str,
        brain_id: str,
        original_content: str,
        compressed_at: str,
        tier: int,
    ) -> None:
        """Save (upsert) a pre-compression content snapshot for a neuron."""
        d = self._dialect
        await d.execute(
            f"""INSERT INTO neuron_snapshots
                (neuron_id, brain_id, original_content, compressed_at, tier)
               VALUES ({d.phs(5)})
               ON CONFLICT(brain_id, neuron_id) DO UPDATE SET
                   original_content = excluded.original_content,
                   compressed_at = excluded.compressed_at,
                   tier = excluded.tier""",
            (neuron_id, brain_id, original_content, compressed_at, tier),
        )

    async def get_neuron_snapshot(self, neuron_id: str) -> dict[str, Any] | None:
        """Retrieve the snapshot for a neuron, if any."""
        d = self._dialect
        brain_id = self._get_brain_id()
        row = await d.fetch_one(
            f"SELECT neuron_id, brain_id, original_content, compressed_at, tier"
            f" FROM neuron_snapshots"
            f" WHERE neuron_id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (neuron_id, brain_id),
        )
        if row is None:
            return None
        return {
            "neuron_id": row["neuron_id"],
            "brain_id": row["brain_id"],
            "original_content": row["original_content"],
            "compressed_at": row["compressed_at"],
            "tier": row["tier"],
        }

    async def delete_neuron_snapshot(self, neuron_id: str) -> bool:
        """Delete the snapshot for a neuron."""
        d = self._dialect
        brain_id = self._get_brain_id()
        count = await d.execute_count(
            f"DELETE FROM neuron_snapshots WHERE neuron_id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            (neuron_id, brain_id),
        )
        return count > 0

    async def cleanup_ephemeral_neurons(self, max_age_hours: float = 24.0) -> int:
        """Delete ephemeral neurons older than *max_age_hours*."""
        d = self._dialect
        brain_id = self._get_brain_id()
        cutoff = d.serialize_dt(utcnow() - timedelta(hours=max_age_hours))

        count = await d.execute_count(
            f"DELETE FROM neurons"
            f" WHERE brain_id = {d.ph(1)} AND ephemeral = 1 AND created_at < {d.ph(2)}",
            (brain_id, cutoff),
        )
        if count > 0:
            self._neuron_cache.invalidate()
        return count

    # ========== Vector Search (PostgreSQL pgvector) ==========

    async def find_neurons_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 10,
        type_filter: NeuronType | None = None,
    ) -> list[tuple[Neuron, float]]:
        """Find neurons by vector similarity (cosine distance).

        Only available when the dialect supports vector search.
        Returns list of ``(neuron, score)`` tuples.
        """
        d = self._dialect
        if not d.supports_vector:
            return []

        brain_id = self._get_brain_id()
        limit = min(limit, 100)

        try:
            if type_filter is not None:
                rows = await d.fetch_all(
                    f"""SELECT n.*, 1 - (n.embedding <=> {d.ph(1)}::vector) AS score
                       FROM neurons n
                       WHERE n.brain_id = {d.ph(2)}
                         AND n.embedding IS NOT NULL
                         AND n.type = {d.ph(3)}
                       ORDER BY n.embedding <=> {d.ph(1)}::vector
                       LIMIT {d.ph(4)}""",
                    (query_embedding, brain_id, type_filter.value, limit),
                )
            else:
                rows = await d.fetch_all(
                    f"""SELECT n.*, 1 - (n.embedding <=> {d.ph(1)}::vector) AS score
                       FROM neurons n
                       WHERE n.brain_id = {d.ph(2)}
                         AND n.embedding IS NOT NULL
                       ORDER BY n.embedding <=> {d.ph(1)}::vector
                       LIMIT {d.ph(3)}""",
                    (query_embedding, brain_id, limit),
                )
            return [(row_to_neuron(r, d), float(r["score"])) for r in rows]
        except Exception:
            logger.error("Embedding similarity search failed", exc_info=True)
            return []
