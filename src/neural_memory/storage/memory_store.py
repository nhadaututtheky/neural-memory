"""In-memory storage backend using NetworkX."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import networkx as nx

from neural_memory.core.brain import Brain, BrainSnapshot
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.storage.base import NeuralStorage

if TYPE_CHECKING:
    pass


class InMemoryStorage(NeuralStorage):
    """
    NetworkX-based in-memory storage for development and testing.

    This storage backend keeps everything in memory using a NetworkX
    MultiDiGraph. Good for development, testing, and small deployments.

    Data is lost when the process exits unless explicitly exported.
    """

    def __init__(self) -> None:
        """Initialize empty storage."""
        # Main graph structure
        self._graph = nx.MultiDiGraph()

        # Indexed data stores (by brain_id)
        self._neurons: dict[str, dict[str, Neuron]] = defaultdict(dict)
        self._synapses: dict[str, dict[str, Synapse]] = defaultdict(dict)
        self._fibers: dict[str, dict[str, Fiber]] = defaultdict(dict)
        self._states: dict[str, dict[str, NeuronState]] = defaultdict(dict)
        self._brains: dict[str, Brain] = {}

        # Current brain context
        self._current_brain_id: str | None = None

    def set_brain(self, brain_id: str) -> None:
        """Set the current brain context for operations."""
        self._current_brain_id = brain_id

    def _get_brain_id(self) -> str:
        """Get current brain ID or raise error."""
        if self._current_brain_id is None:
            raise ValueError("No brain context set. Call set_brain() first.")
        return self._current_brain_id

    # ========== Neuron Operations ==========

    async def add_neuron(self, neuron: Neuron) -> str:
        brain_id = self._get_brain_id()

        if neuron.id in self._neurons[brain_id]:
            raise ValueError(f"Neuron {neuron.id} already exists")

        self._neurons[brain_id][neuron.id] = neuron
        self._graph.add_node(
            neuron.id,
            brain_id=brain_id,
            type=neuron.type,
            content=neuron.content,
        )

        # Initialize state
        self._states[brain_id][neuron.id] = NeuronState(neuron_id=neuron.id)

        return neuron.id

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        brain_id = self._get_brain_id()
        return self._neurons[brain_id].get(neuron_id)

    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
    ) -> list[Neuron]:
        brain_id = self._get_brain_id()
        results: list[Neuron] = []

        for neuron in self._neurons[brain_id].values():
            # Type filter
            if type is not None and neuron.type != type:
                continue

            # Content contains filter (case-insensitive)
            if content_contains is not None:
                if content_contains.lower() not in neuron.content.lower():
                    continue

            # Content exact filter
            if content_exact is not None and neuron.content != content_exact:
                continue

            # Time range filter
            if time_range is not None:
                start, end = time_range
                if not (start <= neuron.created_at <= end):
                    continue

            results.append(neuron)

            if len(results) >= limit:
                break

        return results

    async def update_neuron(self, neuron: Neuron) -> None:
        brain_id = self._get_brain_id()

        if neuron.id not in self._neurons[brain_id]:
            raise ValueError(f"Neuron {neuron.id} does not exist")

        self._neurons[brain_id][neuron.id] = neuron
        self._graph.nodes[neuron.id].update(
            type=neuron.type,
            content=neuron.content,
        )

    async def delete_neuron(self, neuron_id: str) -> bool:
        brain_id = self._get_brain_id()

        if neuron_id not in self._neurons[brain_id]:
            return False

        # Delete connected synapses
        synapses_to_delete = [
            s.id
            for s in self._synapses[brain_id].values()
            if s.source_id == neuron_id or s.target_id == neuron_id
        ]
        for synapse_id in synapses_to_delete:
            await self.delete_synapse(synapse_id)

        # Delete from graph
        if self._graph.has_node(neuron_id):
            self._graph.remove_node(neuron_id)

        # Delete neuron and state
        del self._neurons[brain_id][neuron_id]
        self._states[brain_id].pop(neuron_id, None)

        return True

    # ========== Neuron State Operations ==========

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        brain_id = self._get_brain_id()
        return self._states[brain_id].get(neuron_id)

    async def update_neuron_state(self, state: NeuronState) -> None:
        brain_id = self._get_brain_id()
        self._states[brain_id][state.neuron_id] = state

    # ========== Synapse Operations ==========

    async def add_synapse(self, synapse: Synapse) -> str:
        brain_id = self._get_brain_id()

        if synapse.id in self._synapses[brain_id]:
            raise ValueError(f"Synapse {synapse.id} already exists")

        # Verify neurons exist
        if synapse.source_id not in self._neurons[brain_id]:
            raise ValueError(f"Source neuron {synapse.source_id} does not exist")
        if synapse.target_id not in self._neurons[brain_id]:
            raise ValueError(f"Target neuron {synapse.target_id} does not exist")

        self._synapses[brain_id][synapse.id] = synapse
        self._graph.add_edge(
            synapse.source_id,
            synapse.target_id,
            key=synapse.id,
            type=synapse.type,
            weight=synapse.weight,
        )

        return synapse.id

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        brain_id = self._get_brain_id()
        return self._synapses[brain_id].get(synapse_id)

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
    ) -> list[Synapse]:
        brain_id = self._get_brain_id()
        results: list[Synapse] = []

        for synapse in self._synapses[brain_id].values():
            if source_id is not None and synapse.source_id != source_id:
                continue
            if target_id is not None and synapse.target_id != target_id:
                continue
            if type is not None and synapse.type != type:
                continue
            if min_weight is not None and synapse.weight < min_weight:
                continue

            results.append(synapse)

        return results

    async def update_synapse(self, synapse: Synapse) -> None:
        brain_id = self._get_brain_id()

        if synapse.id not in self._synapses[brain_id]:
            raise ValueError(f"Synapse {synapse.id} does not exist")

        old_synapse = self._synapses[brain_id][synapse.id]
        self._synapses[brain_id][synapse.id] = synapse

        # Update graph edge
        if self._graph.has_edge(old_synapse.source_id, old_synapse.target_id, key=synapse.id):
            self._graph[old_synapse.source_id][old_synapse.target_id][synapse.id].update(
                type=synapse.type,
                weight=synapse.weight,
            )

    async def delete_synapse(self, synapse_id: str) -> bool:
        brain_id = self._get_brain_id()

        if synapse_id not in self._synapses[brain_id]:
            return False

        synapse = self._synapses[brain_id][synapse_id]

        # Remove from graph
        if self._graph.has_edge(synapse.source_id, synapse.target_id, key=synapse_id):
            self._graph.remove_edge(synapse.source_id, synapse.target_id, key=synapse_id)

        del self._synapses[brain_id][synapse_id]
        return True

    # ========== Graph Traversal ==========

    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        brain_id = self._get_brain_id()
        results: list[tuple[Neuron, Synapse]] = []

        if neuron_id not in self._neurons[brain_id]:
            return results

        # Get outgoing connections
        if direction in ("out", "both") and self._graph.has_node(neuron_id):
            for _, target_id, edge_key in self._graph.out_edges(neuron_id, keys=True):
                synapse = self._synapses[brain_id].get(edge_key)
                if synapse is None:
                    continue
                if synapse_types and synapse.type not in synapse_types:
                    continue
                if min_weight is not None and synapse.weight < min_weight:
                    continue

                neighbor = self._neurons[brain_id].get(target_id)
                if neighbor:
                    results.append((neighbor, synapse))

        # Get incoming connections
        if direction in ("in", "both") and self._graph.has_node(neuron_id):
            for source_id, _, edge_key in self._graph.in_edges(neuron_id, keys=True):
                synapse = self._synapses[brain_id].get(edge_key)
                if synapse is None:
                    continue
                if synapse_types and synapse.type not in synapse_types:
                    continue
                if min_weight is not None and synapse.weight < min_weight:
                    continue

                # For incoming, also check if bidirectional
                if direction == "in" and not synapse.is_bidirectional:
                    continue

                neighbor = self._neurons[brain_id].get(source_id)
                if neighbor and (neighbor, synapse) not in results:
                    results.append((neighbor, synapse))

        return results

    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
    ) -> list[tuple[Neuron, Synapse]] | None:
        brain_id = self._get_brain_id()

        if source_id not in self._neurons[brain_id]:
            return None
        if target_id not in self._neurons[brain_id]:
            return None

        try:
            # Use NetworkX shortest path
            path_nodes = nx.shortest_path(
                self._graph,
                source_id,
                target_id,
                weight=None,
            )
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

        if len(path_nodes) - 1 > max_hops:
            return None

        # Build result with neurons and synapses
        result: list[tuple[Neuron, Synapse]] = []

        for i in range(len(path_nodes) - 1):
            from_id = path_nodes[i]
            to_id = path_nodes[i + 1]

            neuron = self._neurons[brain_id].get(to_id)
            if not neuron:
                return None

            # Find the connecting synapse
            edge_data = self._graph.get_edge_data(from_id, to_id)
            if not edge_data:
                return None

            # Get first synapse (highest weight if multiple)
            synapse_id = max(edge_data.keys(), key=lambda k: edge_data[k].get("weight", 0))
            synapse = self._synapses[brain_id].get(synapse_id)
            if not synapse:
                return None

            result.append((neuron, synapse))

        return result

    # ========== Fiber Operations ==========

    async def add_fiber(self, fiber: Fiber) -> str:
        brain_id = self._get_brain_id()

        if fiber.id in self._fibers[brain_id]:
            raise ValueError(f"Fiber {fiber.id} already exists")

        self._fibers[brain_id][fiber.id] = fiber
        return fiber.id

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        brain_id = self._get_brain_id()
        return self._fibers[brain_id].get(fiber_id)

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        limit: int = 100,
    ) -> list[Fiber]:
        brain_id = self._get_brain_id()
        results: list[Fiber] = []

        for fiber in self._fibers[brain_id].values():
            # Contains neuron filter
            if contains_neuron is not None:
                if contains_neuron not in fiber.neuron_ids:
                    continue

            # Time overlap filter
            if time_overlaps is not None:
                start, end = time_overlaps
                if not fiber.overlaps_time(start, end):
                    continue

            # Tags filter (must have ALL specified tags)
            if tags is not None:
                if not tags.issubset(fiber.tags):
                    continue

            # Salience filter
            if min_salience is not None:
                if fiber.salience < min_salience:
                    continue

            results.append(fiber)

            if len(results) >= limit:
                break

        # Sort by salience descending
        results.sort(key=lambda f: f.salience, reverse=True)
        return results

    async def update_fiber(self, fiber: Fiber) -> None:
        brain_id = self._get_brain_id()

        if fiber.id not in self._fibers[brain_id]:
            raise ValueError(f"Fiber {fiber.id} does not exist")

        self._fibers[brain_id][fiber.id] = fiber

    async def delete_fiber(self, fiber_id: str) -> bool:
        brain_id = self._get_brain_id()

        if fiber_id not in self._fibers[brain_id]:
            return False

        del self._fibers[brain_id][fiber_id]
        return True

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        brain_id = self._get_brain_id()
        fibers = list(self._fibers[brain_id].values())

        # Sort by the specified field
        if order_by == "created_at":
            fibers.sort(key=lambda f: f.created_at, reverse=descending)
        elif order_by == "salience":
            fibers.sort(key=lambda f: f.salience, reverse=descending)
        elif order_by == "frequency":
            fibers.sort(key=lambda f: f.frequency, reverse=descending)

        return fibers[:limit]

    # ========== Brain Operations ==========

    async def save_brain(self, brain: Brain) -> None:
        self._brains[brain.id] = brain

    async def get_brain(self, brain_id: str) -> Brain | None:
        return self._brains.get(brain_id)

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        brain = self._brains.get(brain_id)
        if brain is None:
            raise ValueError(f"Brain {brain_id} does not exist")

        from dataclasses import asdict

        # Serialize neurons
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

        # Serialize synapses
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

        # Serialize fibers
        fibers = [
            {
                "id": f.id,
                "neuron_ids": list(f.neuron_ids),
                "synapse_ids": list(f.synapse_ids),
                "anchor_neuron_id": f.anchor_neuron_id,
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

        return BrainSnapshot(
            brain_id=brain_id,
            brain_name=brain.name,
            exported_at=datetime.utcnow(),
            version="0.1.0",
            neurons=neurons,
            synapses=synapses,
            fibers=fibers,
            config=asdict(brain.config),
        )

    async def import_brain(
        self,
        snapshot: BrainSnapshot,
        target_brain_id: str | None = None,
    ) -> str:
        from neural_memory.core.brain import BrainConfig

        brain_id = target_brain_id or snapshot.brain_id

        # Create brain
        config = BrainConfig(**snapshot.config)
        brain = Brain.create(
            name=snapshot.brain_name,
            config=config,
            brain_id=brain_id,
        )
        await self.save_brain(brain)

        # Set context
        old_brain_id = self._current_brain_id
        self.set_brain(brain_id)

        try:
            # Import neurons
            for n_data in snapshot.neurons:
                neuron = Neuron(
                    id=n_data["id"],
                    type=NeuronType(n_data["type"]),
                    content=n_data["content"],
                    metadata=n_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(n_data["created_at"]),
                )
                await self.add_neuron(neuron)

            # Import synapses
            from neural_memory.core.synapse import Direction

            for s_data in snapshot.synapses:
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

            # Import fibers
            for f_data in snapshot.fibers:
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

        finally:
            # Restore context
            self._current_brain_id = old_brain_id

        return brain_id

    # ========== Statistics ==========

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        return {
            "neuron_count": len(self._neurons[brain_id]),
            "synapse_count": len(self._synapses[brain_id]),
            "fiber_count": len(self._fibers[brain_id]),
        }

    # ========== Cleanup ==========

    async def clear(self, brain_id: str) -> None:
        # Remove nodes from graph
        nodes_to_remove = [
            n for n in self._graph.nodes() if self._graph.nodes[n].get("brain_id") == brain_id
        ]
        self._graph.remove_nodes_from(nodes_to_remove)

        # Clear data stores
        self._neurons[brain_id].clear()
        self._synapses[brain_id].clear()
        self._fibers[brain_id].clear()
        self._states[brain_id].clear()
        self._brains.pop(brain_id, None)
