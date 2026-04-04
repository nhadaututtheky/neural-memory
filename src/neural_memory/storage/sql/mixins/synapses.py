"""Dialect-agnostic synapse operations and graph traversal mixin.

Merges SQLiteSynapseMixin and PostgresSynapseMixin into a single
implementation that uses the Dialect abstraction for all SQL generation.

Rules enforced:
- ZERO bare ``?`` or ``$N`` — every placeholder uses ``d.ph(N)``
- ZERO ``.isoformat()`` — uses ``d.serialize_dt(dt)``
- ZERO ``fromisoformat`` — uses ``d.normalize_dt()`` or row_mappers
- ZERO ``conn.execute`` / ``conn.commit`` — uses dialect methods only
- IN clauses via ``d.in_clause()``
"""

from __future__ import annotations

import json
from collections import deque
from typing import TYPE_CHECKING, Any, Literal

from neural_memory.core.neuron import Neuron
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.storage.sql.row_mappers import (
    _row_to_joined_synapse,
    row_to_neuron,
    row_to_synapse,
)

if TYPE_CHECKING:
    from neural_memory.storage.sql.dialect import Dialect


class SynapseMixin:
    """Dialect-agnostic synapse CRUD and graph traversal.

    Requires the mixin host to provide:
    - ``_dialect``: :class:`Dialect` instance
    - ``_get_brain_id() -> str``
    - ``get_neurons_batch(neuron_ids: list[str]) -> dict[str, Neuron]``
    - ``invalidate_merkle_prefix(kind, id, *, is_pro)`` (optional)
    """

    if TYPE_CHECKING:
        _dialect: Dialect

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_brain_id(self) -> str:  # pragma: no cover — provided by host
        raise NotImplementedError

    async def get_neurons_batch(
        self, neuron_ids: list[str]
    ) -> dict[str, Neuron]:  # pragma: no cover
        raise NotImplementedError

    # ========== Synapse CRUD ==========

    async def add_synapse(self, synapse: Synapse) -> str:
        d = self._dialect
        brain_id = self._get_brain_id()

        # Verify both neurons exist
        in_clause, in_params = d.in_clause(2, [synapse.source_id, synapse.target_id])
        rows = await d.fetch_all(
            f"SELECT id FROM neurons WHERE brain_id = {d.ph(1)} AND id {in_clause}",
            [brain_id, *in_params],
        )
        found_ids = {str(r["id"]) for r in rows}

        if synapse.source_id not in found_ids:
            raise ValueError(f"Source neuron {synapse.source_id} does not exist")
        if synapse.target_id not in found_ids:
            raise ValueError(f"Target neuron {synapse.target_id} does not exist")

        try:
            await d.execute(
                f"""INSERT INTO synapses
                   (id, brain_id, source_id, target_id, type, weight, direction,
                    metadata, reinforced_count, last_activated, created_at)
                   VALUES ({d.phs(11)})""",
                [
                    synapse.id,
                    brain_id,
                    synapse.source_id,
                    synapse.target_id,
                    synapse.type.value,
                    synapse.weight,
                    synapse.direction.value,
                    json.dumps(synapse.metadata),
                    synapse.reinforced_count,
                    d.serialize_dt(synapse.last_activated),
                    d.serialize_dt(synapse.created_at),
                ],
            )
            # Invalidate Merkle hash cache
            await self.invalidate_merkle_prefix("synapse", synapse.id, is_pro=True)  # type: ignore[attr-defined]
            return synapse.id
        except Exception as e:
            err = str(e).upper()
            if "UNIQUE" in err or "IntegrityError" in type(e).__name__:
                raise ValueError(f"Synapse {synapse.id} already exists") from e
            if "FOREIGN KEY" in err:
                raise ValueError(
                    f"Source or target neuron does not exist for synapse {synapse.id}"
                ) from e
            raise

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM synapses WHERE id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            [synapse_id, brain_id],
        )
        if row is None:
            return None
        return row_to_synapse(d, row)

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
    ) -> list[Synapse]:
        d = self._dialect
        brain_id = self._get_brain_id()

        query = f"SELECT * FROM synapses WHERE brain_id = {d.ph(1)}"
        params: list[Any] = [brain_id]
        idx = 1

        if source_id is not None:
            idx += 1
            query += f" AND source_id = {d.ph(idx)}"
            params.append(source_id)

        if target_id is not None:
            idx += 1
            query += f" AND target_id = {d.ph(idx)}"
            params.append(target_id)

        if type is not None:
            idx += 1
            query += f" AND type = {d.ph(idx)}"
            params.append(type.value)

        if min_weight is not None:
            idx += 1
            query += f" AND weight >= {d.ph(idx)}"
            params.append(min_weight)

        query += " LIMIT 10000"
        rows = await d.fetch_all(query, params)
        return [row_to_synapse(d, row) for row in rows]

    async def get_all_synapses(self) -> list[Synapse]:
        """Get all synapses for current brain."""
        return await self.get_synapses()

    async def update_synapse(self, synapse: Synapse) -> None:
        d = self._dialect
        brain_id = self._get_brain_id()

        count = await d.execute_count(
            f"""UPDATE synapses
                SET type = {d.ph(1)}, weight = {d.ph(2)}, direction = {d.ph(3)},
                    metadata = {d.ph(4)}, reinforced_count = {d.ph(5)},
                    last_activated = {d.ph(6)}
                WHERE id = {d.ph(7)} AND brain_id = {d.ph(8)}""",
            [
                synapse.type.value,
                synapse.weight,
                synapse.direction.value,
                json.dumps(synapse.metadata),
                synapse.reinforced_count,
                d.serialize_dt(synapse.last_activated),
                synapse.id,
                brain_id,
            ],
        )
        if count == 0:
            raise ValueError(f"Synapse {synapse.id} does not exist")

    async def update_synapses_batch(self, synapses: list[Synapse]) -> None:
        """Update multiple synapses in one batch."""
        if not synapses:
            return
        d = self._dialect
        brain_id = self._get_brain_id()

        args_list = [
            [
                s.type.value,
                s.weight,
                s.direction.value,
                json.dumps(s.metadata),
                s.reinforced_count,
                d.serialize_dt(s.last_activated),
                s.id,
                brain_id,
            ]
            for s in synapses
        ]
        await d.execute_many(
            f"""UPDATE synapses
                SET type = {d.ph(1)}, weight = {d.ph(2)}, direction = {d.ph(3)},
                    metadata = {d.ph(4)}, reinforced_count = {d.ph(5)},
                    last_activated = {d.ph(6)}
                WHERE id = {d.ph(7)} AND brain_id = {d.ph(8)}""",
            args_list,
        )

    async def delete_synapse(self, synapse_id: str) -> bool:
        d = self._dialect
        brain_id = self._get_brain_id()

        count = await d.execute_count(
            f"DELETE FROM synapses WHERE id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            [synapse_id, brain_id],
        )
        if count > 0:
            await self.invalidate_merkle_prefix("synapse", synapse_id, is_pro=True)  # type: ignore[attr-defined]
        return count > 0

    async def delete_synapses_batch(self, synapse_ids: set[str]) -> int:
        """Delete multiple synapses in batched SQL statements.

        Uses chunked ``DELETE ... WHERE id IN (...)`` for efficiency.
        Returns total number of deleted rows.
        """
        if not synapse_ids:
            return 0

        d = self._dialect
        brain_id = self._get_brain_id()
        deleted = 0
        chunk_size = 500
        ids_list = list(synapse_ids)

        for start in range(0, len(ids_list), chunk_size):
            chunk = ids_list[start : start + chunk_size]
            in_clause, in_params = d.in_clause(2, chunk)
            count = await d.execute_count(
                f"DELETE FROM synapses WHERE brain_id = {d.ph(1)} AND id {in_clause}",
                [brain_id, *in_params],
            )
            deleted += count

        return deleted

    async def get_synapses_for_neurons(
        self,
        neuron_ids: list[str],
        direction: str = "out",
    ) -> dict[str, list[Synapse]]:
        """Batch fetch synapses for multiple neurons in a single SQL query."""
        if not neuron_ids:
            return {}

        d = self._dialect
        brain_id = self._get_brain_id()
        result: dict[str, list[Synapse]] = {nid: [] for nid in neuron_ids}
        nid_set = set(neuron_ids)

        direction_cols = {"out": "source_id", "in": "target_id"}

        if direction in direction_cols:
            col = direction_cols[direction]
            in_clause, in_params = d.in_clause(2, neuron_ids)
            rows = await d.fetch_all(
                f"SELECT * FROM synapses WHERE brain_id = {d.ph(1)} AND {col} {in_clause}",
                [brain_id, *in_params],
            )
            for r in rows:
                syn = row_to_synapse(d, r)
                key = str(r[col])
                if key in nid_set:
                    result[key].append(syn)
        elif direction == "both":
            in_clause, in_params = d.in_clause(2, neuron_ids)
            rows = await d.fetch_all(
                f"SELECT * FROM synapses WHERE brain_id = {d.ph(1)} "
                f"AND (source_id {in_clause} OR target_id {in_clause})",
                [brain_id, *in_params, *in_params],
            )
            for r in rows:
                syn = row_to_synapse(d, r)
                if syn.source_id in nid_set:
                    result[syn.source_id].append(syn)
                if syn.target_id in nid_set:
                    result[syn.target_id].append(syn)
        else:
            raise ValueError(f"Invalid direction: {direction!r}. Must be 'out', 'in', or 'both'.")

        return result

    # ========== Graph Traversal ==========

    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        d = self._dialect
        brain_id = self._get_brain_id()
        results: list[tuple[Neuron, Synapse]] = []

        # Base params: ph(1) = brain_id, ph(2) = neuron_id
        # Extra filters start at ph(3)
        extra_conditions: list[str] = []
        extra_params: list[Any] = []
        next_idx = 3

        if synapse_types:
            in_clause, in_p = d.in_clause(next_idx, [t.value for t in synapse_types])
            extra_conditions.append(f"s.type {in_clause}")
            extra_params.extend(in_p)
            # Advance index past the params consumed by in_clause.
            # For SQLite, in_clause consumes len(values) params.
            # For PG, it consumes 1 param (the array).
            next_idx += len(in_p)

        if min_weight is not None:
            extra_conditions.append(f"s.weight >= {d.ph(next_idx)}")
            extra_params.append(min_weight)
            next_idx += 1

        extra_sql = ""
        if extra_conditions:
            extra_sql = " AND " + " AND ".join(extra_conditions)

        if direction in ("out", "both"):
            results.extend(
                await self._fetch_outgoing(d, neuron_id, brain_id, extra_sql, extra_params)
            )

        if direction in ("in", "both"):
            incoming = await self._fetch_incoming(d, neuron_id, brain_id, extra_sql, extra_params)
            seen_ids = {s.id for _, s in results}
            for neuron, syn in incoming:
                if direction == "in" and not syn.is_bidirectional:
                    continue
                if syn.id not in seen_ids:
                    seen_ids.add(syn.id)
                    results.append((neuron, syn))

        return results

    async def _fetch_outgoing(
        self,
        d: Dialect,
        neuron_id: str,
        brain_id: str,
        extra_sql: str,
        extra_params: list[Any],
    ) -> list[tuple[Neuron, Synapse]]:
        """Fetch outgoing neighbor neurons and their synapses."""
        query = f"""
            SELECT n.*, s.id AS s_id, s.source_id, s.target_id, s.type AS s_type,
                   s.weight, s.direction, s.metadata AS s_metadata,
                   s.reinforced_count, s.last_activated AS s_last_activated,
                   s.created_at AS s_created_at
            FROM synapses s
            JOIN neurons n ON s.target_id = n.id AND s.brain_id = n.brain_id
            WHERE s.brain_id = {d.ph(1)} AND s.source_id = {d.ph(2)}{extra_sql}
        """
        params: list[Any] = [brain_id, neuron_id, *extra_params]
        rows = await d.fetch_all(query, params)
        return [(row_to_neuron(d, row), _row_to_joined_synapse(d, row)) for row in rows]

    async def _fetch_incoming(
        self,
        d: Dialect,
        neuron_id: str,
        brain_id: str,
        extra_sql: str,
        extra_params: list[Any],
    ) -> list[tuple[Neuron, Synapse]]:
        """Fetch incoming neighbor neurons and their synapses."""
        query = f"""
            SELECT n.*, s.id AS s_id, s.source_id, s.target_id, s.type AS s_type,
                   s.weight, s.direction, s.metadata AS s_metadata,
                   s.reinforced_count, s.last_activated AS s_last_activated,
                   s.created_at AS s_created_at
            FROM synapses s
            JOIN neurons n ON s.source_id = n.id AND s.brain_id = n.brain_id
            WHERE s.brain_id = {d.ph(1)} AND s.target_id = {d.ph(2)}{extra_sql}
        """
        params: list[Any] = [brain_id, neuron_id, *extra_params]
        rows = await d.fetch_all(query, params)
        return [(row_to_neuron(d, row), _row_to_joined_synapse(d, row)) for row in rows]

    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
        bidirectional: bool = False,
    ) -> list[tuple[Neuron, Synapse]] | None:
        """Find shortest path between two neurons using BFS.

        Args:
            source_id: Starting neuron ID.
            target_id: Destination neuron ID.
            max_hops: Maximum path length (capped at 10).
            bidirectional: If True, traverse both outgoing and incoming
                edges (treats the graph as undirected).
        """
        max_hops = min(max_hops, 10)
        d = self._dialect
        brain_id = self._get_brain_id()

        # Verify both endpoints exist
        in_clause, in_params = d.in_clause(2, [source_id, target_id])
        rows = await d.fetch_all(
            f"SELECT id FROM neurons WHERE brain_id = {d.ph(1)} AND id {in_clause}",
            [brain_id, *in_params],
        )
        if len(rows) < 2:
            return None

        visited: set[str] = {source_id}
        queue: deque[tuple[str, list[tuple[str, str]]]] = deque([(source_id, [])])

        # Build edge SQL using dialect placeholders.
        # ph(1) = current_id, ph(2) = brain_id
        if bidirectional:
            edge_sql = (
                f"SELECT id, target_id AS next_id FROM synapses"
                f" WHERE source_id = {d.ph(1)} AND brain_id = {d.ph(2)}"
                f" UNION ALL"
                f" SELECT id, source_id AS next_id FROM synapses"
                f" WHERE target_id = {d.ph(3)} AND brain_id = {d.ph(4)}"
            )
        else:
            edge_sql = (
                f"SELECT id, target_id AS next_id FROM synapses"
                f" WHERE source_id = {d.ph(1)} AND brain_id = {d.ph(2)}"
            )

        while queue:
            current_id, path = queue.popleft()

            if len(path) >= max_hops:
                continue

            if bidirectional:
                edge_params: list[Any] = [current_id, brain_id, current_id, brain_id]
            else:
                edge_params = [current_id, brain_id]

            edge_rows = await d.fetch_all(edge_sql, edge_params)
            for row in edge_rows:
                next_id = str(row["next_id"])
                synapse_id = str(row["id"])

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
        syn_map = await self._get_synapses_batch(synapse_ids)
        result: list[tuple[Neuron, Synapse]] = []
        for neuron_id, synapse_id in path:
            neuron = neurons.get(neuron_id)
            syn = syn_map.get(synapse_id)
            if neuron and syn:
                result.append((neuron, syn))
        return result

    async def _get_synapses_batch(self, synapse_ids: list[str]) -> dict[str, Synapse]:
        """Batch fetch synapses by ID list."""
        if not synapse_ids:
            return {}
        d = self._dialect
        brain_id = self._get_brain_id()
        in_clause, in_params = d.in_clause(2, synapse_ids)
        rows = await d.fetch_all(
            f"SELECT * FROM synapses WHERE brain_id = {d.ph(1)} AND id {in_clause}",
            [brain_id, *in_params],
        )
        return {str(r["id"]): row_to_synapse(d, r) for r in rows}
