"""In-memory brain operations mixin (CRUD, export, import)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

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


class InMemoryBrainMixin:
    """Mixin providing brain CRUD, export, and import for in-memory storage."""

    _brains: dict[str, Brain]
    _neurons: dict[str, dict[str, Neuron]]
    _synapses: dict[str, dict[str, Synapse]]
    _fibers: dict[str, dict[str, Fiber]]
    _typed_memories: dict[str, dict[str, TypedMemory]]
    _projects: dict[str, dict[str, Project]]
    _current_brain_id: str | None

    def set_brain(self, brain_id: str) -> None: ...
    def _get_brain_id(self) -> str: ...
    async def add_neuron(self, neuron: Neuron) -> str: ...
    async def add_synapse(self, synapse: Synapse) -> str: ...
    async def add_fiber(self, fiber: Fiber) -> str: ...

    async def save_brain(self, brain: Brain) -> None:
        self._brains[brain.id] = brain

    async def get_brain(self, brain_id: str) -> Brain | None:
        return self._brains.get(brain_id)

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        brain = self._brains.get(brain_id)
        if brain is None:
            raise ValueError(f"Brain {brain_id} does not exist")

        neurons = [
            {
                "id": n.id,
                "type": n.type.value,
                "content": n.content,
                "metadata": n.metadata,
                "created_at": n.created_at.isoformat(),
            }
            for n in self._neurons[brain_id].values()
        ]

        synapses = [
            {
                "id": s.id,
                "source_id": s.source_id,
                "target_id": s.target_id,
                "type": s.type.value,
                "weight": s.weight,
                "direction": s.direction.value,
                "metadata": s.metadata,
                "reinforced_count": s.reinforced_count,
                "created_at": s.created_at.isoformat(),
            }
            for s in self._synapses[brain_id].values()
        ]

        fibers = [
            {
                "id": f.id,
                "neuron_ids": list(f.neuron_ids),
                "synapse_ids": list(f.synapse_ids),
                "anchor_neuron_id": f.anchor_neuron_id,
                "pathway": f.pathway,
                "conductivity": f.conductivity,
                "last_conducted": f.last_conducted.isoformat() if f.last_conducted else None,
                "time_start": f.time_start.isoformat() if f.time_start else None,
                "time_end": f.time_end.isoformat() if f.time_end else None,
                "coherence": f.coherence,
                "salience": f.salience,
                "frequency": f.frequency,
                "summary": f.summary,
                "tags": list(f.tags),
                "metadata": f.metadata,
                "created_at": f.created_at.isoformat(),
            }
            for f in self._fibers[brain_id].values()
        ]

        typed_memories = [
            {
                "fiber_id": tm.fiber_id,
                "memory_type": tm.memory_type.value,
                "priority": tm.priority.value,
                "provenance": {
                    "source": tm.provenance.source,
                    "confidence": tm.provenance.confidence.value,
                    "verified": tm.provenance.verified,
                    "verified_at": (
                        tm.provenance.verified_at.isoformat() if tm.provenance.verified_at else None
                    ),
                    "created_by": tm.provenance.created_by,
                    "last_confirmed": (
                        tm.provenance.last_confirmed.isoformat()
                        if tm.provenance.last_confirmed
                        else None
                    ),
                },
                "expires_at": tm.expires_at.isoformat() if tm.expires_at else None,
                "project_id": tm.project_id,
                "tags": list(tm.tags),
                "metadata": tm.metadata,
                "created_at": tm.created_at.isoformat(),
            }
            for tm in self._typed_memories[brain_id].values()
        ]

        projects = [p.to_dict() for p in self._projects[brain_id].values()]

        return BrainSnapshot(
            brain_id=brain_id,
            brain_name=brain.name,
            exported_at=datetime.utcnow(),
            version="0.1.0",
            neurons=neurons,
            synapses=synapses,
            fibers=fibers,
            config=asdict(brain.config),
            metadata={"typed_memories": typed_memories, "projects": projects},
        )

    async def import_brain(
        self,
        snapshot: BrainSnapshot,
        target_brain_id: str | None = None,
    ) -> str:
        brain_id = target_brain_id or snapshot.brain_id

        config = BrainConfig(**snapshot.config)
        brain = Brain.create(name=snapshot.brain_name, config=config, brain_id=brain_id)
        await self.save_brain(brain)

        old_brain_id = self._current_brain_id
        self.set_brain(brain_id)

        try:
            await self._import_neurons(brain_id, snapshot.neurons)
            await self._import_synapses(brain_id, snapshot.synapses)
            await self._import_fibers(brain_id, snapshot.fibers)
            await self._import_typed_memories(brain_id, snapshot.metadata.get("typed_memories", []))
            self._import_projects(brain_id, snapshot.metadata.get("projects", []))
        finally:
            self._current_brain_id = old_brain_id

        return brain_id

    async def _import_neurons(self, brain_id: str, neurons_data: list[dict]) -> None:
        for n_data in neurons_data:
            neuron = Neuron(
                id=n_data["id"],
                type=NeuronType(n_data["type"]),
                content=n_data["content"],
                metadata=n_data.get("metadata", {}),
                created_at=datetime.fromisoformat(n_data["created_at"]),
            )
            await self.add_neuron(neuron)

    async def _import_synapses(self, brain_id: str, synapses_data: list[dict]) -> None:
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

    async def _import_fibers(self, brain_id: str, fibers_data: list[dict]) -> None:
        for f_data in fibers_data:
            fiber = Fiber(
                id=f_data["id"],
                neuron_ids=set(f_data["neuron_ids"]),
                synapse_ids=set(f_data["synapse_ids"]),
                anchor_neuron_id=f_data["anchor_neuron_id"],
                pathway=f_data.get("pathway", []),
                conductivity=f_data.get("conductivity", 1.0),
                last_conducted=(
                    datetime.fromisoformat(f_data["last_conducted"])
                    if f_data.get("last_conducted")
                    else None
                ),
                time_start=(
                    datetime.fromisoformat(f_data["time_start"])
                    if f_data.get("time_start")
                    else None
                ),
                time_end=(
                    datetime.fromisoformat(f_data["time_end"])
                    if f_data.get("time_end")
                    else None
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

    async def _import_typed_memories(
        self, brain_id: str, typed_memories_data: list[dict]
    ) -> None:
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
            if typed_memory.fiber_id in self._fibers[brain_id]:
                self._typed_memories[brain_id][typed_memory.fiber_id] = typed_memory

    def _import_projects(self, brain_id: str, projects_data: list[dict]) -> None:
        for p_data in projects_data:
            project = Project.from_dict(p_data)
            self._projects[brain_id][project.id] = project
