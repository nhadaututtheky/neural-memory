"""SQLite brain operations mixin (CRUD, export, import)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.core.brain import Brain, BrainConfig, BrainSnapshot
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import (
    Confidence,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
)
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.project import Project
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.sqlite_row_mappers import row_to_brain

if TYPE_CHECKING:
    import aiosqlite


class SQLiteBrainMixin:
    """Mixin providing brain CRUD, export, and import operations."""

    def _ensure_conn(self) -> aiosqlite.Connection: ...
    def _get_brain_id(self) -> str: ...

    _current_brain_id: str | None

    def set_brain(self, brain_id: str) -> None: ...

    # These are provided by other mixins
    async def save_brain(self, brain: Brain) -> None:
        conn = self._ensure_conn()

        await conn.execute(
            """INSERT OR REPLACE INTO brains
               (id, name, config, owner_id, is_public, shared_with, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                brain.id,
                brain.name,
                json.dumps(
                    {
                        "decay_rate": brain.config.decay_rate,
                        "reinforcement_delta": brain.config.reinforcement_delta,
                        "activation_threshold": brain.config.activation_threshold,
                        "max_spread_hops": brain.config.max_spread_hops,
                        "max_context_tokens": brain.config.max_context_tokens,
                        "default_synapse_weight": brain.config.default_synapse_weight,
                        "hebbian_delta": brain.config.hebbian_delta,
                        "hebbian_threshold": brain.config.hebbian_threshold,
                        "hebbian_initial_weight": brain.config.hebbian_initial_weight,
                    }
                ),
                brain.owner_id,
                1 if brain.is_public else 0,
                json.dumps(brain.shared_with),
                brain.created_at.isoformat(),
                brain.updated_at.isoformat(),
            ),
        )
        await conn.commit()

    async def get_brain(self, brain_id: str) -> Brain | None:
        conn = self._ensure_conn()

        async with conn.execute("SELECT * FROM brains WHERE id = ?", (brain_id,)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return row_to_brain(row)

    async def find_brain_by_name(self, name: str) -> Brain | None:
        conn = self._ensure_conn()

        async with conn.execute("SELECT * FROM brains WHERE name = ?", (name,)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return row_to_brain(row)

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        conn = self._ensure_conn()

        brain = await self.get_brain(brain_id)
        if brain is None:
            raise ValueError(f"Brain {brain_id} does not exist")

        neurons = await _export_neurons(conn, brain_id)
        synapses = await _export_synapses(conn, brain_id)
        fibers = await _export_fibers(conn, brain_id)
        typed_memories = await _export_typed_memories(conn, brain_id)
        projects = await _export_projects(conn, brain_id)

        return BrainSnapshot(
            brain_id=brain_id,
            brain_name=brain.name,
            exported_at=datetime.utcnow(),
            version="0.1.0",
            neurons=neurons,
            synapses=synapses,
            fibers=fibers,
            config={
                "decay_rate": brain.config.decay_rate,
                "reinforcement_delta": brain.config.reinforcement_delta,
                "activation_threshold": brain.config.activation_threshold,
                "max_spread_hops": brain.config.max_spread_hops,
                "max_context_tokens": brain.config.max_context_tokens,
                "default_synapse_weight": brain.config.default_synapse_weight,
                "hebbian_delta": brain.config.hebbian_delta,
                "hebbian_threshold": brain.config.hebbian_threshold,
                "hebbian_initial_weight": brain.config.hebbian_initial_weight,
            },
            metadata={
                "typed_memories": typed_memories,
                "projects": projects,
            },
        )

    async def import_brain(
        self,
        snapshot: BrainSnapshot,
        target_brain_id: str | None = None,
    ) -> str:
        brain_id = target_brain_id or snapshot.brain_id

        config = BrainConfig(**snapshot.config)
        brain = Brain.create(
            name=snapshot.brain_name,
            config=config,
            brain_id=brain_id,
        )
        await self.save_brain(brain)

        old_brain_id = self._current_brain_id
        self.set_brain(brain_id)

        try:
            await self._import_neurons(snapshot.neurons)
            await self._import_synapses(snapshot.synapses)
            await self._import_fibers(snapshot.fibers)
            await self._import_projects(snapshot.metadata.get("projects", []))
            await self._import_typed_memories(snapshot.metadata.get("typed_memories", []))
        finally:
            self._current_brain_id = old_brain_id

        return brain_id

    async def _import_neurons(self, neurons_data: list[dict]) -> None:
        for n_data in neurons_data:
            neuron = Neuron(
                id=n_data["id"],
                type=NeuronType(n_data["type"]),
                content=n_data["content"],
                metadata=n_data.get("metadata", {}),
                created_at=datetime.fromisoformat(n_data["created_at"]),
            )
            await self.add_neuron(neuron)

    async def _import_synapses(self, synapses_data: list[dict]) -> None:
        for s_data in synapses_data:
            synapse = Synapse(
                id=s_data["id"],
                source_id=s_data["source_id"],
                target_id=s_data["target_id"],
                type=SynapseType(s_data["type"]),
                weight=s_data["weight"],
                direction=Direction(s_data["direction"]),
                metadata=s_data.get("metadata", {}),
                reinforced_count=s_data.get("reinforced_count", 0),
                created_at=datetime.fromisoformat(s_data["created_at"]),
            )
            await self.add_synapse(synapse)

    async def _import_fibers(self, fibers_data: list[dict]) -> None:
        for f_data in fibers_data:
            fiber = Fiber(
                id=f_data["id"],
                neuron_ids=set(f_data["neuron_ids"]),
                synapse_ids=set(f_data["synapse_ids"]),
                anchor_neuron_id=f_data["anchor_neuron_id"],
                time_start=(
                    datetime.fromisoformat(f_data["time_start"])
                    if f_data.get("time_start")
                    else None
                ),
                time_end=(
                    datetime.fromisoformat(f_data["time_end"]) if f_data.get("time_end") else None
                ),
                coherence=f_data.get("coherence", 0.0),
                salience=f_data.get("salience", 0.0),
                frequency=f_data.get("frequency", 0),
                summary=f_data.get("summary"),
                tags=set(f_data.get("tags", [])),
                metadata=f_data.get("metadata", {}),
                created_at=datetime.fromisoformat(f_data["created_at"]),
            )
            await self.add_fiber(fiber)

    async def _import_projects(self, projects_data: list[dict]) -> None:
        for p_data in projects_data:
            project = Project(
                id=p_data["id"],
                name=p_data["name"],
                description=p_data.get("description", ""),
                start_date=datetime.fromisoformat(p_data["start_date"]),
                end_date=(
                    datetime.fromisoformat(p_data["end_date"]) if p_data.get("end_date") else None
                ),
                tags=frozenset(p_data.get("tags", [])),
                priority=p_data.get("priority", 1.0),
                metadata=p_data.get("metadata", {}),
                created_at=datetime.fromisoformat(p_data["created_at"]),
            )
            await self.add_project(project)

    async def _import_typed_memories(self, typed_memories_data: list[dict]) -> None:
        for tm_data in typed_memories_data:
            prov_data = tm_data.get("provenance", {})
            provenance = Provenance(
                source=prov_data.get("source", "import"),
                confidence=Confidence(prov_data.get("confidence", "medium")),
                verified=prov_data.get("verified", False),
                verified_at=(
                    datetime.fromisoformat(prov_data["verified_at"])
                    if prov_data.get("verified_at")
                    else None
                ),
                created_by=prov_data.get("created_by", "import"),
                last_confirmed=(
                    datetime.fromisoformat(prov_data["last_confirmed"])
                    if prov_data.get("last_confirmed")
                    else None
                ),
            )

            typed_memory = TypedMemory(
                fiber_id=tm_data["fiber_id"],
                memory_type=MemoryType(tm_data["memory_type"]),
                priority=Priority(tm_data["priority"]),
                provenance=provenance,
                expires_at=(
                    datetime.fromisoformat(tm_data["expires_at"])
                    if tm_data.get("expires_at")
                    else None
                ),
                project_id=tm_data.get("project_id"),
                tags=frozenset(tm_data.get("tags", [])),
                metadata=tm_data.get("metadata", {}),
                created_at=datetime.fromisoformat(tm_data["created_at"]),
            )
            await self.add_typed_memory(typed_memory)


# ========== Export helpers (module-level) ==========


async def _export_neurons(conn: aiosqlite.Connection, brain_id: str) -> list[dict]:
    neurons: list[dict] = []
    async with conn.execute("SELECT * FROM neurons WHERE brain_id = ?", (brain_id,)) as cursor:
        async for row in cursor:
            neurons.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                }
            )
    return neurons


async def _export_synapses(conn: aiosqlite.Connection, brain_id: str) -> list[dict]:
    synapses: list[dict] = []
    async with conn.execute("SELECT * FROM synapses WHERE brain_id = ?", (brain_id,)) as cursor:
        async for row in cursor:
            synapses.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "type": row["type"],
                    "weight": row["weight"],
                    "direction": row["direction"],
                    "metadata": json.loads(row["metadata"]),
                    "reinforced_count": row["reinforced_count"],
                    "created_at": row["created_at"],
                }
            )
    return synapses


async def _export_fibers(conn: aiosqlite.Connection, brain_id: str) -> list[dict]:
    fibers: list[dict] = []
    async with conn.execute("SELECT * FROM fibers WHERE brain_id = ?", (brain_id,)) as cursor:
        async for row in cursor:
            fibers.append(
                {
                    "id": row["id"],
                    "neuron_ids": json.loads(row["neuron_ids"]),
                    "synapse_ids": json.loads(row["synapse_ids"]),
                    "anchor_neuron_id": row["anchor_neuron_id"],
                    "time_start": row["time_start"],
                    "time_end": row["time_end"],
                    "coherence": row["coherence"],
                    "salience": row["salience"],
                    "frequency": row["frequency"],
                    "summary": row["summary"],
                    "tags": json.loads(row["tags"]),
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                }
            )
    return fibers


async def _export_typed_memories(conn: aiosqlite.Connection, brain_id: str) -> list[dict]:
    typed_memories: list[dict] = []
    async with conn.execute(
        "SELECT * FROM typed_memories WHERE brain_id = ?", (brain_id,)
    ) as cursor:
        async for row in cursor:
            typed_memories.append(
                {
                    "fiber_id": row["fiber_id"],
                    "memory_type": row["memory_type"],
                    "priority": row["priority"],
                    "provenance": json.loads(row["provenance"]),
                    "expires_at": row["expires_at"],
                    "project_id": row["project_id"],
                    "tags": json.loads(row["tags"]),
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                }
            )
    return typed_memories


async def _export_projects(conn: aiosqlite.Connection, brain_id: str) -> list[dict]:
    projects: list[dict] = []
    async with conn.execute("SELECT * FROM projects WHERE brain_id = ?", (brain_id,)) as cursor:
        async for row in cursor:
            projects.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "tags": json.loads(row["tags"]),
                    "priority": row["priority"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                }
            )
    return projects
