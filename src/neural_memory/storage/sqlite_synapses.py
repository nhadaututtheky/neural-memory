"""SQLite synapse operations and graph traversal mixin."""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from neural_memory.core.neuron import Neuron
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.sqlite_row_mappers import row_to_neuron, row_to_synapse

if TYPE_CHECKING:
    import aiosqlite


class SQLiteSynapseMixin:
    """Mixin providing synapse CRUD and graph traversal operations."""

    def _ensure_conn(self) -> aiosqlite.Connection: ...
    def _get_brain_id(self) -> str: ...

    # ========== Synapse Operations ==========

    async def add_synapse(self, synapse: Synapse) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        # Verify neurons exist
        async with conn.execute(
            "SELECT id FROM neurons WHERE id IN (?, ?) AND brain_id = ?",
            (synapse.source_id, synapse.target_id, brain_id),
        ) as cursor:
            rows = await cursor.fetchall()
            found_ids = {row["id"] for row in rows}

        if synapse.source_id not in found_ids:
            raise ValueError(f"Source neuron {synapse.source_id} does not exist")
        if synapse.target_id not in found_ids:
            raise ValueError(f"Target neuron {synapse.target_id} does not exist")

        try:
            await conn.execute(
                """INSERT INTO synapses
                   (id, brain_id, source_id, target_id, type, weight, direction,
                    metadata, reinforced_count, last_activated, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    synapse.id,
                    brain_id,
                    synapse.source_id,
                    synapse.target_id,
                    synapse.type.value,
                    synapse.weight,
                    synapse.direction.value,
                    json.dumps(synapse.metadata),
                    synapse.reinforced_count,
                    synapse.last_activated.isoformat() if synapse.last_activated else None,
                    synapse.created_at.isoformat(),
                ),
            )
            await conn.commit()
            return synapse.id
        except sqlite3.IntegrityError:
            raise ValueError(f"Synapse {synapse.id} already exists")

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM synapses WHERE id = ? AND brain_id = ?",
            (synapse_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return row_to_synapse(row)

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
    ) -> list[Synapse]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        query = "SELECT * FROM synapses WHERE brain_id = ?"
        params: list[Any] = [brain_id]

        if source_id is not None:
            query += " AND source_id = ?"
            params.append(source_id)

        if target_id is not None:
            query += " AND target_id = ?"
            params.append(target_id)

        if type is not None:
            query += " AND type = ?"
            params.append(type.value)

        if min_weight is not None:
            query += " AND weight >= ?"
            params.append(min_weight)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [row_to_synapse(row) for row in rows]

    async def get_all_synapses(self) -> list[Synapse]:
        """Get all synapses for current brain."""
        return await self.get_synapses()

    async def update_synapse(self, synapse: Synapse) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE synapses SET type = ?, weight = ?, direction = ?,
               metadata = ?, reinforced_count = ?, last_activated = ?
               WHERE id = ? AND brain_id = ?""",
            (
                synapse.type.value,
                synapse.weight,
                synapse.direction.value,
                json.dumps(synapse.metadata),
                synapse.reinforced_count,
                synapse.last_activated.isoformat() if synapse.last_activated else None,
                synapse.id,
                brain_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Synapse {synapse.id} does not exist")

        await conn.commit()

    async def delete_synapse(self, synapse_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM synapses WHERE id = ? AND brain_id = ?",
            (synapse_id, brain_id),
        )
        await conn.commit()

        return cursor.rowcount > 0

    async def get_synapses_for_neurons(
        self,
        neuron_ids: list[str],
        direction: str = "out",
    ) -> dict[str, list[Synapse]]:
        """Batch fetch synapses for multiple neurons in a single SQL query."""
        if not neuron_ids:
            return {}

        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        col = "source_id" if direction == "out" else "target_id"
        placeholders = ",".join("?" for _ in neuron_ids)
        query = f"SELECT * FROM synapses WHERE brain_id = ? AND {col} IN ({placeholders})"
        params: list[Any] = [brain_id, *neuron_ids]

        result: dict[str, list[Synapse]] = {nid: [] for nid in neuron_ids}
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                synapse = row_to_synapse(row)
                key = row[col]
                result[key].append(synapse)

        return result

    # ========== Graph Traversal ==========

    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        results: list[tuple[Neuron, Synapse]] = []

        # Build parameterized filter clauses
        extra_conditions: list[str] = []
        extra_params: list[Any] = []

        if synapse_types:
            placeholders = ",".join("?" for _ in synapse_types)
            extra_conditions.append(f"s.type IN ({placeholders})")
            extra_params.extend(t.value for t in synapse_types)

        if min_weight is not None:
            extra_conditions.append("s.weight >= ?")
            extra_params.append(min_weight)

        if direction in ("out", "both"):
            results.extend(
                await self._fetch_outgoing(
                    conn, neuron_id, brain_id, extra_conditions, extra_params
                )
            )

        if direction in ("in", "both"):
            incoming = await self._fetch_incoming(
                conn, neuron_id, brain_id, extra_conditions, extra_params
            )
            for pair in incoming:
                if direction == "in" and not pair[1].is_bidirectional:
                    continue
                if pair not in results:
                    results.append(pair)

        return results

    async def _fetch_outgoing(
        self,
        conn: aiosqlite.Connection,
        neuron_id: str,
        brain_id: str,
        extra_conditions: list[str],
        extra_params: list[Any],
    ) -> list[tuple[Neuron, Synapse]]:
        """Fetch outgoing neighbor neurons and their synapses."""
        where_clause = "s.source_id = ? AND s.brain_id = ?"
        params: list[Any] = [neuron_id, brain_id]

        for condition in extra_conditions:
            where_clause += f" AND {condition}"
        params.extend(extra_params)

        query = f"""
            SELECT n.*, s.id as s_id, s.source_id, s.target_id, s.type as s_type,
                   s.weight, s.direction, s.metadata as s_metadata,
                   s.reinforced_count, s.last_activated as s_last_activated,
                   s.created_at as s_created_at
            FROM synapses s
            JOIN neurons n ON s.target_id = n.id AND s.brain_id = n.brain_id
            WHERE {where_clause}
        """
        results: list[tuple[Neuron, Synapse]] = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                results.append((row_to_neuron(row), _row_to_joined_synapse(row)))
        return results

    async def _fetch_incoming(
        self,
        conn: aiosqlite.Connection,
        neuron_id: str,
        brain_id: str,
        extra_conditions: list[str],
        extra_params: list[Any],
    ) -> list[tuple[Neuron, Synapse]]:
        """Fetch incoming neighbor neurons and their synapses."""
        where_clause = "s.target_id = ? AND s.brain_id = ?"
        params: list[Any] = [neuron_id, brain_id]

        for condition in extra_conditions:
            where_clause += f" AND {condition}"
        params.extend(extra_params)

        query = f"""
            SELECT n.*, s.id as s_id, s.source_id, s.target_id, s.type as s_type,
                   s.weight, s.direction, s.metadata as s_metadata,
                   s.reinforced_count, s.last_activated as s_last_activated,
                   s.created_at as s_created_at
            FROM synapses s
            JOIN neurons n ON s.source_id = n.id AND s.brain_id = n.brain_id
            WHERE {where_clause}
        """
        results: list[tuple[Neuron, Synapse]] = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                results.append((row_to_neuron(row), _row_to_joined_synapse(row)))
        return results

    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
    ) -> list[tuple[Neuron, Synapse]] | None:
        """Find shortest path using BFS."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT id FROM neurons WHERE id IN (?, ?) AND brain_id = ?",
            (source_id, target_id, brain_id),
        ) as cursor:
            rows = await cursor.fetchall()
            if len(rows) < 2:
                return None

        visited = {source_id}
        queue: deque[tuple[str, list[tuple[str, str]]]] = deque([(source_id, [])])

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_hops:
                continue

            async with conn.execute(
                """SELECT id, target_id FROM synapses
                   WHERE source_id = ? AND brain_id = ?""",
                (current_id, brain_id),
            ) as cursor:
                async for row in cursor:
                    next_id = row["target_id"]
                    synapse_id = row["id"]

                    if next_id == target_id:
                        full_path = [*path, (next_id, synapse_id)]
                        return await self._build_path_result(full_path)

                    if next_id not in visited:
                        visited.add(next_id)
                        queue.append((next_id, [*path, (next_id, synapse_id)]))

        return None

    async def _build_path_result(self, path: list[tuple[str, str]]) -> list[tuple[Neuron, Synapse]]:
        """Build path result from neuron/synapse IDs using batch fetches."""
        if not path:
            return []
        neuron_ids = [nid for nid, _ in path]
        synapse_ids = [sid for _, sid in path]
        neurons = await self.get_neurons_batch(neuron_ids)
        synapses_map = await self._get_synapses_batch(synapse_ids)
        result: list[tuple[Neuron, Synapse]] = []
        for neuron_id, synapse_id in path:
            neuron = neurons.get(neuron_id)
            synapse = synapses_map.get(synapse_id)
            if neuron and synapse:
                result.append((neuron, synapse))
        return result

    async def _get_synapses_batch(self, synapse_ids: list[str]) -> dict[str, Synapse]:
        """Batch fetch synapses by ID list."""
        if not synapse_ids:
            return {}
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        placeholders = ",".join("?" for _ in synapse_ids)
        query = f"SELECT * FROM synapses WHERE brain_id = ? AND id IN ({placeholders})"
        params: list[Any] = [brain_id, *synapse_ids]
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return {row["id"]: row_to_synapse(row) for row in rows}


def _row_to_joined_synapse(row: Any) -> Synapse:
    """Convert a joined row (with s_ prefixed columns) to a Synapse."""
    return Synapse(
        id=row["s_id"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        type=SynapseType(row["s_type"]),
        weight=row["weight"],
        direction=Direction(row["direction"]),
        metadata=json.loads(row["s_metadata"]),
        reinforced_count=row["reinforced_count"],
        last_activated=(
            datetime.fromisoformat(row["s_last_activated"]) if row["s_last_activated"] else None
        ),
        created_at=datetime.fromisoformat(row["s_created_at"]),
    )
