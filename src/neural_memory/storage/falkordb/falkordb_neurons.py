"""FalkorDB neuron and neuron-state CRUD operations."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.storage.falkordb.falkordb_base import FalkorDBBaseMixin


def _fts_escape(term: str) -> str:
    """Escape a search term for FalkorDB fulltext query."""
    tokens = term.split()
    if not tokens:
        return ""
    safe = []
    for t in tokens:
        cleaned = re.sub(r"[^\w]", "", t, flags=re.UNICODE)
        if cleaned:
            safe.append(cleaned)
    return " ".join(safe)


def _fts_prefix(prefix: str) -> str:
    """Build a prefix query for FalkorDB fulltext."""
    tokens = prefix.split()
    if not tokens:
        return ""
    parts: list[str] = []
    for t in tokens[:-1]:
        cleaned = re.sub(r"[^\w]", "", t, flags=re.UNICODE)
        if cleaned:
            parts.append(cleaned)
    last = re.sub(r"[^\w]", "", tokens[-1], flags=re.UNICODE)
    if last:
        parts.append(f"{last}*")
    return " ".join(parts)


class FalkorDBNeuronMixin(FalkorDBBaseMixin):
    """FalkorDB implementation of neuron and neuron-state operations."""

    # ========== Neuron CRUD ==========

    async def add_neuron(self, neuron: Neuron) -> str:
        existing = await self._query_ro(
            "MATCH (n:Neuron {id: $id}) RETURN n.id",
            {"id": neuron.id},
        )
        if existing:
            raise ValueError(f"Neuron {neuron.id} already exists")

        await self._query(
            """
            CREATE (n:Neuron {
                id: $id,
                type: $type,
                content: $content,
                metadata: $metadata,
                content_hash: $content_hash,
                created_at: $created_at,
                activation_level: 0.0,
                access_frequency: 0,
                last_activated: $null_val,
                decay_rate: 0.1,
                firing_threshold: 0.3,
                refractory_until: $null_val,
                refractory_period_ms: 500.0,
                homeostatic_target: 0.5
            })
            """,
            {
                "id": neuron.id,
                "type": neuron.type.value,
                "content": neuron.content,
                "metadata": self._serialize_metadata(neuron.metadata),
                "content_hash": neuron.content_hash,
                "created_at": self._dt_to_str(neuron.created_at),
                "null_val": None,
            },
        )
        return neuron.id

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        rows = await self._query_ro(
            """
            MATCH (n:Neuron {id: $id})
            RETURN n.id, n.type, n.content, n.metadata,
                   n.content_hash, n.created_at
            """,
            {"id": neuron_id},
        )
        if not rows:
            return None
        return self._row_to_neuron(rows[0])

    async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        if not neuron_ids:
            return {}
        rows = await self._query_ro(
            """
            MATCH (n:Neuron)
            WHERE n.id IN $ids
            RETURN n.id, n.type, n.content, n.metadata,
                   n.content_hash, n.created_at
            """,
            {"ids": neuron_ids},
        )
        result: dict[str, Neuron] = {}
        for row in rows:
            neuron = self._row_to_neuron(row)
            result[neuron.id] = neuron
        return result

    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Neuron]:
        full_scan = content_contains is None and content_exact is None
        limit = min(limit, 10000 if full_scan else 1000)

        # FTS path for content_contains
        if content_contains and not content_exact:
            fts_term = _fts_escape(content_contains)
            if fts_term:
                return await self._find_neurons_fts(
                    fts_term, type=type, time_range=time_range, limit=limit
                )

        # Cypher filter path
        conditions: list[str] = []
        params: dict[str, Any] = {}

        if type is not None:
            conditions.append("n.type = $type")
            params["type"] = type.value
        if content_exact is not None:
            conditions.append("n.content = $content_exact")
            params["content_exact"] = content_exact
        if content_contains and not content_exact:
            conditions.append("n.content CONTAINS $content_sub")
            params["content_sub"] = content_contains
        if time_range is not None:
            conditions.append("n.created_at >= $t_start AND n.created_at <= $t_end")
            params["t_start"] = self._dt_to_str(time_range[0])
            params["t_end"] = self._dt_to_str(time_range[1])

        where = " AND ".join(conditions) if conditions else "true"
        rows = await self._query_ro(
            f"""
            MATCH (n:Neuron)
            WHERE {where}
            RETURN n.id, n.type, n.content, n.metadata,
                   n.content_hash, n.created_at
            ORDER BY n.id
            SKIP $offset
            LIMIT $limit
            """,
            {**params, "limit": limit, "offset": offset},
        )
        return [self._row_to_neuron(r) for r in rows]

    async def _find_neurons_fts(
        self,
        fts_query: str,
        type: NeuronType | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
    ) -> list[Neuron]:
        """Use FalkorDB fulltext index for content search."""
        rows = await self._query_ro(
            """
            CALL db.idx.fulltext.queryNodes('Neuron', $query)
            YIELD node
            RETURN node.id, node.type, node.content, node.metadata,
                   node.content_hash, node.created_at
            LIMIT $limit
            """,
            {"query": fts_query, "limit": limit},
        )
        results = [self._row_to_neuron(r) for r in rows]

        # Post-filter type and time_range (FTS does not support these natively)
        if type is not None:
            results = [n for n in results if n.type == type]
        if time_range is not None:
            t_start, t_end = time_range
            results = [n for n in results if t_start <= n.created_at <= t_end]

        return results[:limit]

    async def find_neurons_exact_batch(
        self,
        contents: list[str],
        type: NeuronType | None = None,
    ) -> dict[str, Neuron]:
        if not contents:
            return {}

        params: dict[str, Any] = {"contents": contents}
        type_filter = ""
        if type is not None:
            type_filter = "AND n.type = $type"
            params["type"] = type.value

        rows = await self._query_ro(
            f"""
            MATCH (n:Neuron)
            WHERE n.content IN $contents {type_filter}
            RETURN n.id, n.type, n.content, n.metadata,
                   n.content_hash, n.created_at
            """,
            params,
        )
        result: dict[str, Neuron] = {}
        for row in rows:
            neuron = self._row_to_neuron(row)
            if neuron.content not in result:
                result[neuron.content] = neuron
        return result

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: NeuronType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        limit = min(limit, 50)
        fts_term = _fts_prefix(prefix)

        if fts_term:
            rows = await self._query_ro(
                """
                CALL db.idx.fulltext.queryNodes('Neuron', $query)
                YIELD node
                RETURN node.id, node.content, node.type,
                       node.access_frequency, node.activation_level
                LIMIT $limit
                """,
                {"query": fts_term, "limit": limit},
            )
        else:
            rows = await self._query_ro(
                """
                MATCH (n:Neuron)
                WHERE n.content STARTS WITH $prefix
                RETURN n.id, n.content, n.type,
                       n.access_frequency, n.activation_level
                ORDER BY n.access_frequency DESC
                LIMIT $limit
                """,
                {"prefix": prefix, "limit": limit},
            )

        suggestions: list[dict[str, Any]] = []
        for row in rows:
            ntype = row[2] if len(row) > 2 else ""
            if type_filter and ntype != type_filter.value:
                continue
            freq = row[3] if len(row) > 3 else 0
            act = row[4] if len(row) > 4 else 0.0
            suggestions.append(
                {
                    "neuron_id": row[0],
                    "content": row[1],
                    "type": ntype,
                    "access_frequency": freq,
                    "activation_level": act,
                    "score": (freq or 0) * 0.7 + (act or 0.0) * 0.3,
                }
            )
        return suggestions[:limit]

    async def update_neuron(self, neuron: Neuron) -> None:
        rows = await self._query(
            """
            MATCH (n:Neuron {id: $id})
            SET n.type = $type,
                n.content = $content,
                n.metadata = $metadata,
                n.content_hash = $content_hash,
                n.created_at = $created_at
            RETURN n.id
            """,
            {
                "id": neuron.id,
                "type": neuron.type.value,
                "content": neuron.content,
                "metadata": self._serialize_metadata(neuron.metadata),
                "content_hash": neuron.content_hash,
                "created_at": self._dt_to_str(neuron.created_at),
            },
        )
        if not rows:
            raise ValueError(f"Neuron {neuron.id} not found")

    async def delete_neuron(self, neuron_id: str) -> bool:
        # Check existence first, then DETACH DELETE
        existing = await self._query_ro(
            "MATCH (n:Neuron {id: $id}) RETURN n.id",
            {"id": neuron_id},
        )
        if not existing:
            return False
        await self._query(
            "MATCH (n:Neuron {id: $id}) DETACH DELETE n",
            {"id": neuron_id},
        )
        return True

    # ========== Neuron State ==========

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        rows = await self._query_ro(
            """
            MATCH (n:Neuron {id: $id})
            RETURN n.id,
                   n.activation_level, n.access_frequency,
                   n.last_activated, n.decay_rate, n.created_at,
                   n.firing_threshold, n.refractory_until,
                   n.refractory_period_ms, n.homeostatic_target
            """,
            {"id": neuron_id},
        )
        if not rows:
            return None
        return self._row_to_neuron_state(rows[0])

    async def update_neuron_state(self, state: NeuronState) -> None:
        await self._query(
            """
            MATCH (n:Neuron {id: $neuron_id})
            SET n.activation_level = $activation_level,
                n.access_frequency = $access_frequency,
                n.last_activated = $last_activated,
                n.decay_rate = $decay_rate,
                n.firing_threshold = $firing_threshold,
                n.refractory_until = $refractory_until,
                n.refractory_period_ms = $refractory_period_ms,
                n.homeostatic_target = $homeostatic_target
            """,
            {
                "neuron_id": state.neuron_id,
                "activation_level": state.activation_level,
                "access_frequency": state.access_frequency,
                "last_activated": self._dt_to_str(state.last_activated),
                "decay_rate": state.decay_rate,
                "firing_threshold": state.firing_threshold,
                "refractory_until": self._dt_to_str(state.refractory_until),
                "refractory_period_ms": state.refractory_period_ms,
                "homeostatic_target": state.homeostatic_target,
            },
        )

    async def get_neuron_states_batch(self, neuron_ids: list[str]) -> dict[str, NeuronState]:
        if not neuron_ids:
            return {}
        rows = await self._query_ro(
            """
            MATCH (n:Neuron)
            WHERE n.id IN $ids
            RETURN n.id,
                   n.activation_level, n.access_frequency,
                   n.last_activated, n.decay_rate, n.created_at,
                   n.firing_threshold, n.refractory_until,
                   n.refractory_period_ms, n.homeostatic_target
            """,
            {"ids": neuron_ids},
        )
        result: dict[str, NeuronState] = {}
        for row in rows:
            state = self._row_to_neuron_state(row)
            result[state.neuron_id] = state
        return result

    async def get_all_neuron_states(self) -> list[NeuronState]:
        rows = await self._query_ro(
            """
            MATCH (n:Neuron)
            RETURN n.id,
                   n.activation_level, n.access_frequency,
                   n.last_activated, n.decay_rate, n.created_at,
                   n.firing_threshold, n.refractory_until,
                   n.refractory_period_ms, n.homeostatic_target
            """
        )
        return [self._row_to_neuron_state(r) for r in rows]

    # ========== Row Mappers ==========

    def _row_to_neuron(self, row: list[Any]) -> Neuron:
        """Convert FalkorDB result row to Neuron dataclass.

        Expected columns: id, type, content, metadata, content_hash, created_at
        """
        return Neuron(
            id=row[0],
            type=NeuronType(row[1]),
            content=row[2],
            metadata=self._deserialize_metadata(row[3]),
            content_hash=row[4] if row[4] is not None else 0,
            created_at=self._str_to_dt(row[5]) or datetime.min,
        )

    def _row_to_neuron_state(self, row: list[Any]) -> NeuronState:
        """Convert FalkorDB result row to NeuronState dataclass.

        Expected columns: neuron_id, activation_level, access_frequency,
                         last_activated, decay_rate, created_at,
                         firing_threshold, refractory_until,
                         refractory_period_ms, homeostatic_target
        """
        return NeuronState(
            neuron_id=row[0],
            activation_level=row[1] if row[1] is not None else 0.0,
            access_frequency=row[2] if row[2] is not None else 0,
            last_activated=self._str_to_dt(row[3]),
            decay_rate=row[4] if row[4] is not None else 0.1,
            created_at=self._str_to_dt(row[5]) or datetime.min,
            firing_threshold=row[6] if row[6] is not None else 0.3,
            refractory_until=self._str_to_dt(row[7]),
            refractory_period_ms=row[8] if row[8] is not None else 500.0,
            homeostatic_target=row[9] if row[9] is not None else 0.5,
        )
