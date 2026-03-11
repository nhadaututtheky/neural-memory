"""PostgreSQL synapse operations and graph traversal."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from typing import Any, Literal

from neural_memory.core.neuron import Neuron
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.postgres.postgres_base import PostgresBaseMixin
from neural_memory.storage.postgres.postgres_row_mappers import (
    row_to_neuron,
    row_to_synapse,
)
from neural_memory.utils.timeutils import utcnow as _utcnow


def _row_to_joined_synapse(row: Any) -> Synapse:
    """Convert joined row (s_ prefixed columns) to Synapse."""
    s_created = row.get("s_created_at") or row.get("s_created")
    created = s_created
    if created is not None and not hasattr(created, "isoformat"):
        created = datetime.fromisoformat(str(created))
    return Synapse(
        id=str(row["s_id"]),
        source_id=str(row["source_id"]),
        target_id=str(row["target_id"]),
        type=SynapseType(row["s_type"]),
        weight=float(row["weight"]),
        direction=Direction(row.get("direction", "uni")),
        metadata=json.loads(row["s_metadata"]) if row.get("s_metadata") else {},
        reinforced_count=int(row.get("reinforced_count", 0)),
        last_activated=row.get("s_last_activated"),
        created_at=created or _utcnow(),
    )


class PostgresSynapseMixin(PostgresBaseMixin):
    """PostgreSQL synapse CRUD and graph traversal."""

    async def add_synapse(self, synapse: Synapse) -> str:
        brain_id = self._get_brain_id()

        rows = await self._query_ro(
            "SELECT id FROM neurons WHERE brain_id = $1 AND id = ANY($2::text[])",
            brain_id,
            [synapse.source_id, synapse.target_id],
        )
        found = {str(r["id"]) for r in rows}
        if synapse.source_id not in found:
            raise ValueError(f"Source neuron {synapse.source_id} does not exist")
        if synapse.target_id not in found:
            raise ValueError(f"Target neuron {synapse.target_id} does not exist")

        try:
            await self._query(
                """INSERT INTO synapses
                   (id, brain_id, source_id, target_id, type, weight, direction,
                    metadata, reinforced_count, last_activated, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
                synapse.id,
                brain_id,
                synapse.source_id,
                synapse.target_id,
                synapse.type.value,
                synapse.weight,
                synapse.direction.value,
                json.dumps(synapse.metadata),
                synapse.reinforced_count,
                synapse.last_activated,
                synapse.created_at,
            )
            return synapse.id
        except Exception as e:
            from asyncpg.exceptions import UniqueViolationError

            if isinstance(e, UniqueViolationError):
                raise ValueError(f"Synapse {synapse.id} already exists") from e
            raise

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        brain_id = self._get_brain_id()
        row = await self._query_one(
            "SELECT * FROM synapses WHERE brain_id = $1 AND id = $2",
            brain_id,
            synapse_id,
        )
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
        brain_id = self._get_brain_id()
        query = "SELECT * FROM synapses WHERE brain_id = $1"
        params: list[Any] = [brain_id]

        if source_id is not None:
            params.append(source_id)
            query += f" AND source_id = ${len(params)}"
        if target_id is not None:
            params.append(target_id)
            query += f" AND target_id = ${len(params)}"
        if type is not None:
            params.append(type.value)
            query += f" AND type = ${len(params)}"
        if min_weight is not None:
            params.append(min_weight)
            query += f" AND weight >= ${len(params)}"

        rows = await self._query_ro(query, *params)
        return [row_to_synapse(r) for r in rows]

    async def get_all_synapses(self) -> list[Synapse]:
        return await self.get_synapses()

    async def get_synapses_for_neurons(
        self,
        neuron_ids: list[str],
        direction: str = "out",
    ) -> dict[str, list[Synapse]]:
        if not neuron_ids:
            return {}
        brain_id = self._get_brain_id()
        result: dict[str, list[Synapse]] = {nid: [] for nid in neuron_ids}
        nid_set = set(neuron_ids)

        if direction == "out":
            rows = await self._query_ro(
                "SELECT * FROM synapses WHERE brain_id = $1 AND source_id = ANY($2::text[])",
                brain_id,
                neuron_ids,
            )
            for r in rows:
                syn = row_to_synapse(r)
                if syn.source_id in nid_set:
                    result[syn.source_id].append(syn)
        elif direction == "in":
            rows = await self._query_ro(
                "SELECT * FROM synapses WHERE brain_id = $1 AND target_id = ANY($2::text[])",
                brain_id,
                neuron_ids,
            )
            for r in rows:
                syn = row_to_synapse(r)
                if syn.target_id in nid_set:
                    result[syn.target_id].append(syn)
        else:
            rows = await self._query_ro(
                """SELECT * FROM synapses WHERE brain_id = $1
                   AND (source_id = ANY($2::text[]) OR target_id = ANY($2::text[]))""",
                brain_id,
                neuron_ids,
            )
            for r in rows:
                syn = row_to_synapse(r)
                if syn.source_id in nid_set:
                    result[syn.source_id].append(syn)
                if syn.target_id in nid_set:
                    result[syn.target_id].append(syn)

        return result

    async def update_synapse(self, synapse: Synapse) -> None:
        brain_id = self._get_brain_id()
        r = await self._query(
            """UPDATE synapses SET type = $1, weight = $2, direction = $3,
               metadata = $4, reinforced_count = $5, last_activated = $6
               WHERE brain_id = $7 AND id = $8""",
            synapse.type.value,
            synapse.weight,
            synapse.direction.value,
            json.dumps(synapse.metadata),
            synapse.reinforced_count,
            synapse.last_activated,
            brain_id,
            synapse.id,
        )
        if r == "UPDATE 0":
            raise ValueError(f"Synapse {synapse.id} does not exist")

    async def update_synapses_batch(self, synapses: list[Synapse]) -> None:
        """Update multiple synapses in one batch."""
        if not synapses:
            return
        brain_id = self._get_brain_id()
        args_list = [
            (
                s.type.value,
                s.weight,
                s.direction.value,
                json.dumps(s.metadata),
                s.reinforced_count,
                s.last_activated,
                brain_id,
                s.id,
            )
            for s in synapses
        ]
        await self._executemany(
            """UPDATE synapses SET type = $1, weight = $2, direction = $3,
               metadata = $4, reinforced_count = $5, last_activated = $6
               WHERE brain_id = $7 AND id = $8""",
            args_list,
        )

    async def delete_synapse(self, synapse_id: str) -> bool:
        brain_id = self._get_brain_id()
        r = await self._query(
            "DELETE FROM synapses WHERE brain_id = $1 AND id = $2",
            brain_id,
            synapse_id,
        )
        return bool(r != "DELETE 0")

    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        brain_id = self._get_brain_id()
        results: list[tuple[Neuron, Synapse]] = []
        if synapse_types and min_weight is not None:
            where_extra = " AND s.type = ANY($3::text[]) AND s.weight >= $4"
            params: list[Any] = [brain_id, neuron_id, [t.value for t in synapse_types], min_weight]
        elif synapse_types:
            where_extra = " AND s.type = ANY($3::text[])"
            params = [brain_id, neuron_id, [t.value for t in synapse_types]]
        elif min_weight is not None:
            where_extra = " AND s.weight >= $3"
            params = [brain_id, neuron_id, min_weight]
        else:
            where_extra = ""
            params = [brain_id, neuron_id]

        if direction in ("out", "both"):
            base = (
                "SELECT n.*, s.id as s_id, s.source_id, s.target_id, s.type as s_type, "
                "s.weight, s.direction, s.metadata as s_metadata, "
                "s.reinforced_count, s.last_activated as s_last_activated, "
                "s.created_at as s_created_at "
                "FROM synapses s "
                "JOIN neurons n ON s.target_id = n.id AND s.brain_id = n.brain_id "
                "WHERE s.brain_id = $1 AND s.source_id = $2"
            )
            q = base + where_extra
            rows = await self._query_ro(q, *params)
            results.extend((row_to_neuron(r), _row_to_joined_synapse(r)) for r in rows)

        if direction in ("in", "both"):
            base = (
                "SELECT n.*, s.id as s_id, s.source_id, s.target_id, s.type as s_type, "
                "s.weight, s.direction, s.metadata as s_metadata, "
                "s.reinforced_count, s.last_activated as s_last_activated, "
                "s.created_at as s_created_at "
                "FROM synapses s "
                "JOIN neurons n ON s.source_id = n.id AND s.brain_id = n.brain_id "
                "WHERE s.brain_id = $1 AND s.target_id = $2"
            )
            q = base + where_extra
            rows = await self._query_ro(q, *params)
            seen_sids = {s.id for _, s in results}
            for r in rows:
                syn = _row_to_joined_synapse(r)
                if syn.id not in seen_sids:
                    seen_sids.add(syn.id)
                    results.append((row_to_neuron(r), syn))

        return results

    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
        bidirectional: bool = False,
    ) -> list[tuple[Neuron, Synapse]] | None:
        brain_id = self._get_brain_id()
        max_hops = min(max_hops, 10)
        check = await self._query_ro(
            "SELECT id FROM neurons WHERE brain_id = $1 AND id = ANY($2::text[])",
            brain_id,
            [source_id, target_id],
        )
        if len(check) < 2:
            return None

        visited = {source_id}
        queue: deque[tuple[str, list[tuple[str, str]]]] = deque([(source_id, [])])

        if bidirectional:
            edge_sql = """
                SELECT id, target_id AS next_id FROM synapses
                WHERE source_id = $1 AND brain_id = $2
                UNION ALL
                SELECT id, source_id AS next_id FROM synapses
                WHERE target_id = $1 AND brain_id = $2
            """
        else:
            edge_sql = """
                SELECT id, target_id AS next_id FROM synapses
                WHERE source_id = $1 AND brain_id = $2
            """

        while queue:
            current_id, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            rows = await self._query_ro(edge_sql, current_id, brain_id)

            for row in rows:
                next_id = str(row["next_id"])
                synapse_id = str(row["id"])
                if next_id == target_id:
                    full_path = path + [(next_id, synapse_id)]
                    return await self._build_path_result(full_path)
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [(next_id, synapse_id)]))

        return None

    async def _build_path_result(self, path: list[tuple[str, str]]) -> list[tuple[Neuron, Synapse]]:
        """Build path from IDs using batch fetches."""
        if not path:
            return []
        neuron_ids = [pid for pid, _ in path]
        synapse_ids = [sid for _, sid in path]
        neurons = await self.get_neurons_batch(neuron_ids)  # type: ignore[attr-defined]
        syn_map = await self._get_synapses_batch(synapse_ids)
        result: list[tuple[Neuron, Synapse]] = []
        for nid, sid in path:
            neuron = neurons.get(nid)
            syn = syn_map.get(sid)
            if neuron and syn:
                result.append((neuron, syn))
        return result

    async def _get_synapses_batch(self, synapse_ids: list[str]) -> dict[str, Synapse]:
        if not synapse_ids:
            return {}
        brain_id = self._get_brain_id()
        rows = await self._query_ro(
            "SELECT * FROM synapses WHERE brain_id = $1 AND id = ANY($2::text[])",
            brain_id,
            synapse_ids,
        )
        return {str(r["id"]): row_to_synapse(r) for r in rows}
