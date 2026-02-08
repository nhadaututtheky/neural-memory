"""In-memory storage backend using NetworkX."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

import networkx as nx

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import TypedMemory
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.project import Project
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.brain_versioning import BrainVersion
from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.memory_brain_ops import InMemoryBrainMixin
from neural_memory.storage.memory_collections import InMemoryCollectionsMixin


class InMemoryStorage(InMemoryCollectionsMixin, InMemoryBrainMixin, NeuralStorage):
    """NetworkX-based in-memory storage for development and testing.

    Data is lost when the process exits unless explicitly exported.
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()
        self._neurons: dict[str, dict[str, Neuron]] = defaultdict(dict)
        self._synapses: dict[str, dict[str, Synapse]] = defaultdict(dict)
        self._fibers: dict[str, dict[str, Fiber]] = defaultdict(dict)
        self._states: dict[str, dict[str, NeuronState]] = defaultdict(dict)
        self._typed_memories: dict[str, dict[str, TypedMemory]] = defaultdict(dict)
        self._projects: dict[str, dict[str, Project]] = defaultdict(dict)
        self._brains: dict[str, Brain] = {}
        self._co_activations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._action_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._versions: dict[str, dict[str, tuple[BrainVersion, str]]] = defaultdict(dict)
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
        self._states[brain_id][neuron.id] = NeuronState(neuron_id=neuron.id)
        return neuron.id

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        brain_id = self._get_brain_id()
        return self._neurons[brain_id].get(neuron_id)

    async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        """Batch fetch neurons from in-memory store."""
        brain_id = self._get_brain_id()
        brain_neurons = self._neurons[brain_id]
        return {nid: brain_neurons[nid] for nid in neuron_ids if nid in brain_neurons}

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
            if type is not None and neuron.type != type:
                continue
            if content_contains is not None:
                if content_contains.lower() not in neuron.content.lower():
                    continue
            if content_exact is not None and neuron.content != content_exact:
                continue
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
        self._graph.nodes[neuron.id].update(type=neuron.type, content=neuron.content)

    async def delete_neuron(self, neuron_id: str) -> bool:
        brain_id = self._get_brain_id()

        if neuron_id not in self._neurons[brain_id]:
            return False

        synapses_to_delete = [
            s.id
            for s in self._synapses[brain_id].values()
            if s.source_id == neuron_id or s.target_id == neuron_id
        ]
        for synapse_id in synapses_to_delete:
            await self.delete_synapse(synapse_id)

        if self._graph.has_node(neuron_id):
            self._graph.remove_node(neuron_id)

        del self._neurons[brain_id][neuron_id]
        self._states[brain_id].pop(neuron_id, None)
        return True

    # ========== Neuron State Operations ==========

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        brain_id = self._get_brain_id()
        return self._states[brain_id].get(neuron_id)

    async def get_neuron_states_batch(self, neuron_ids: list[str]) -> dict[str, NeuronState]:
        """Batch fetch neuron states from in-memory store."""
        brain_id = self._get_brain_id()
        brain_states = self._states[brain_id]
        return {nid: brain_states[nid] for nid in neuron_ids if nid in brain_states}

    async def update_neuron_state(self, state: NeuronState) -> None:
        brain_id = self._get_brain_id()
        self._states[brain_id][state.neuron_id] = state

    async def get_all_neuron_states(self) -> list[NeuronState]:
        brain_id = self._get_brain_id()
        return list(self._states[brain_id].values())

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: NeuronType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest neurons matching a prefix, ranked by relevance + frequency."""
        if not prefix.strip():
            return []

        brain_id = self._get_brain_id()
        prefix_lower = prefix.lower()
        scored: list[dict[str, Any]] = []

        for neuron in self._neurons[brain_id].values():
            if prefix_lower not in neuron.content.lower():
                continue
            if type_filter is not None and neuron.type != type_filter:
                continue
            state = self._states[brain_id].get(neuron.id)
            freq = state.access_frequency if state else 0
            activation = state.activation_level if state else 0.0
            score = freq * 0.1 + activation * 0.5
            scored.append(
                {
                    "neuron_id": neuron.id,
                    "content": neuron.content,
                    "type": neuron.type.value,
                    "access_frequency": freq,
                    "activation_level": activation,
                    "score": score,
                }
            )

        scored.sort(key=lambda s: s["score"], reverse=True)
        return scored[:limit]

    # ========== Synapse Operations ==========

    async def add_synapse(self, synapse: Synapse) -> str:
        brain_id = self._get_brain_id()

        if synapse.id in self._synapses[brain_id]:
            raise ValueError(f"Synapse {synapse.id} already exists")
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

    async def get_synapses_for_neurons(
        self,
        neuron_ids: list[str],
        direction: str = "out",
    ) -> dict[str, list[Synapse]]:
        """Batch fetch synapses for multiple neurons from in-memory store."""
        brain_id = self._get_brain_id()
        result: dict[str, list[Synapse]] = {nid: [] for nid in neuron_ids}
        nid_set = set(neuron_ids)

        for synapse in self._synapses[brain_id].values():
            key = synapse.source_id if direction == "out" else synapse.target_id
            if key in nid_set:
                result[key].append(synapse)

        return result

    async def get_all_synapses(self) -> list[Synapse]:
        return await self.get_synapses()

    async def update_synapse(self, synapse: Synapse) -> None:
        brain_id = self._get_brain_id()

        if synapse.id not in self._synapses[brain_id]:
            raise ValueError(f"Synapse {synapse.id} does not exist")

        old_synapse = self._synapses[brain_id][synapse.id]
        self._synapses[brain_id][synapse.id] = synapse

        if self._graph.has_edge(old_synapse.source_id, old_synapse.target_id, key=synapse.id):
            self._graph[old_synapse.source_id][old_synapse.target_id][synapse.id].update(
                type=synapse.type, weight=synapse.weight
            )

    async def delete_synapse(self, synapse_id: str) -> bool:
        brain_id = self._get_brain_id()

        if synapse_id not in self._synapses[brain_id]:
            return False

        synapse = self._synapses[brain_id][synapse_id]
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

        if direction in ("in", "both") and self._graph.has_node(neuron_id):
            for source_id, _, edge_key in self._graph.in_edges(neuron_id, keys=True):
                synapse = self._synapses[brain_id].get(edge_key)
                if synapse is None:
                    continue
                if synapse_types and synapse.type not in synapse_types:
                    continue
                if min_weight is not None and synapse.weight < min_weight:
                    continue
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
            path_nodes = nx.shortest_path(self._graph, source_id, target_id, weight=None)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        if len(path_nodes) - 1 > max_hops:
            return None

        result: list[tuple[Neuron, Synapse]] = []
        for i in range(len(path_nodes) - 1):
            from_id = path_nodes[i]
            to_id = path_nodes[i + 1]

            neuron = self._neurons[brain_id].get(to_id)
            if not neuron:
                return None

            edge_data = self._graph.get_edge_data(from_id, to_id)
            if not edge_data:
                return None

            synapse_id = max(edge_data.keys(), key=lambda k: edge_data[k].get("weight", 0))
            synapse = self._synapses[brain_id].get(synapse_id)
            if not synapse:
                return None

            result.append((neuron, synapse))

        return result

    # ========== Statistics ==========

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        return {
            "neuron_count": len(self._neurons[brain_id]),
            "synapse_count": len(self._synapses[brain_id]),
            "fiber_count": len(self._fibers[brain_id]),
            "project_count": len(self._projects[brain_id]),
        }

    async def get_enhanced_stats(self, brain_id: str) -> dict[str, Any]:
        basic_stats = await self.get_stats(brain_id)

        # Hot neurons by access frequency
        hot_neurons: list[dict[str, Any]] = []
        states = sorted(
            self._states[brain_id].values(),
            key=lambda s: s.access_frequency,
            reverse=True,
        )
        for state in states[:10]:
            neuron = self._neurons[brain_id].get(state.neuron_id)
            if neuron:
                hot_neurons.append(
                    {
                        "neuron_id": state.neuron_id,
                        "content": neuron.content,
                        "type": neuron.type.value,
                        "activation_level": state.activation_level,
                        "access_frequency": state.access_frequency,
                    }
                )

        # Today's fibers
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_fibers_count = sum(
            1 for f in self._fibers[brain_id].values() if f.created_at >= today
        )

        # Neuron type breakdown
        neuron_type_breakdown: dict[str, int] = {}
        for neuron in self._neurons[brain_id].values():
            t = neuron.type.value
            neuron_type_breakdown[t] = neuron_type_breakdown.get(t, 0) + 1

        # Synapse stats
        synapse_stats: dict[str, Any] = {
            "avg_weight": 0.0,
            "total_reinforcements": 0,
            "by_type": {},
        }
        by_type: dict[str, list[float]] = {}
        total_reinforcements = 0
        for synapse in self._synapses[brain_id].values():
            t = synapse.type.value
            by_type.setdefault(t, []).append(synapse.weight)
            total_reinforcements += synapse.reinforced_count

        all_weights: list[float] = []
        for t, weights in by_type.items():
            avg_w = sum(weights) / len(weights) if weights else 0.0
            synapse_stats["by_type"][t] = {
                "count": len(weights),
                "avg_weight": round(avg_w, 4),
                "total_reinforcements": 0,
            }
            all_weights.extend(weights)

        if all_weights:
            synapse_stats["avg_weight"] = round(sum(all_weights) / len(all_weights), 4)
        synapse_stats["total_reinforcements"] = total_reinforcements

        # Memory time range
        fibers = list(self._fibers[brain_id].values())
        oldest_memory: str | None = None
        newest_memory: str | None = None
        if fibers:
            dates = [f.created_at for f in fibers]
            oldest_memory = min(dates).isoformat()
            newest_memory = max(dates).isoformat()

        return {
            **basic_stats,
            "db_size_bytes": 0,
            "hot_neurons": hot_neurons,
            "today_fibers_count": today_fibers_count,
            "synapse_stats": synapse_stats,
            "neuron_type_breakdown": neuron_type_breakdown,
            "oldest_memory": oldest_memory,
            "newest_memory": newest_memory,
        }

    # ========== Co-Activation Operations ==========

    async def record_co_activation(
        self,
        neuron_a: str,
        neuron_b: str,
        binding_strength: float,
        source_anchor: str | None = None,
    ) -> str:
        brain_id = self._get_brain_id()
        event_id = str(uuid4())
        a, b = (neuron_a, neuron_b) if neuron_a < neuron_b else (neuron_b, neuron_a)
        self._co_activations[brain_id].append(
            {
                "id": event_id,
                "neuron_a": a,
                "neuron_b": b,
                "binding_strength": binding_strength,
                "source_anchor": source_anchor,
                "created_at": datetime.utcnow(),
            }
        )
        return event_id

    async def get_co_activation_counts(
        self,
        since: datetime | None = None,
        min_count: int = 1,
    ) -> list[tuple[str, str, int, float]]:
        brain_id = self._get_brain_id()
        pair_counts: dict[tuple[str, str], list[float]] = defaultdict(list)

        for event in self._co_activations[brain_id]:
            if since is not None and event["created_at"] < since:
                continue
            pair = (event["neuron_a"], event["neuron_b"])
            pair_counts[pair].append(event["binding_strength"])

        results: list[tuple[str, str, int, float]] = []
        for (a, b), strengths in pair_counts.items():
            count = len(strengths)
            if count >= min_count:
                avg_strength = sum(strengths) / count
                results.append((a, b, count, avg_strength))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    async def prune_co_activations(self, older_than: datetime) -> int:
        brain_id = self._get_brain_id()
        original_count = len(self._co_activations[brain_id])
        self._co_activations[brain_id] = [
            e for e in self._co_activations[brain_id] if e["created_at"] >= older_than
        ]
        return original_count - len(self._co_activations[brain_id])

    # ========== Action Event Operations ==========

    async def record_action(
        self,
        action_type: str,
        action_context: str = "",
        tags: tuple[str, ...] | list[str] = (),
        session_id: str | None = None,
        fiber_id: str | None = None,
    ) -> str:
        brain_id = self._get_brain_id()
        event_id = str(uuid4())
        self._action_events[brain_id].append(
            {
                "id": event_id,
                "brain_id": brain_id,
                "session_id": session_id,
                "action_type": action_type,
                "action_context": action_context,
                "tags": tuple(tags),
                "fiber_id": fiber_id,
                "created_at": datetime.utcnow(),
            }
        )
        return event_id

    async def get_action_sequences(
        self,
        session_id: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list:
        from neural_memory.core.action_event import ActionEvent

        brain_id = self._get_brain_id()
        results: list[ActionEvent] = []

        for event in self._action_events[brain_id]:
            if session_id is not None and event["session_id"] != session_id:
                continue
            if since is not None and event["created_at"] < since:
                continue
            results.append(
                ActionEvent(
                    id=event["id"],
                    brain_id=event["brain_id"],
                    session_id=event["session_id"],
                    action_type=event["action_type"],
                    action_context=event["action_context"],
                    tags=event["tags"],
                    fiber_id=event["fiber_id"],
                    created_at=event["created_at"],
                )
            )
            if len(results) >= limit:
                break

        results.sort(key=lambda e: e.created_at)
        return results

    async def prune_action_events(self, older_than: datetime) -> int:
        brain_id = self._get_brain_id()
        original_count = len(self._action_events[brain_id])
        self._action_events[brain_id] = [
            e for e in self._action_events[brain_id] if e["created_at"] >= older_than
        ]
        return original_count - len(self._action_events[brain_id])

    # ========== Version Operations ==========

    async def save_version(
        self,
        brain_id: str,
        version: BrainVersion,
        snapshot_json: str,
    ) -> None:
        # Check unique name constraint
        for existing_version, _ in self._versions[brain_id].values():
            if existing_version.version_name == version.version_name:
                raise ValueError(
                    f"Version name '{version.version_name}' already exists for brain {brain_id}"
                )
        self._versions[brain_id][version.id] = (version, snapshot_json)

    async def get_version(
        self,
        brain_id: str,
        version_id: str,
    ) -> tuple[BrainVersion, str] | None:
        return self._versions[brain_id].get(version_id)

    async def list_versions(
        self,
        brain_id: str,
        limit: int = 20,
    ) -> list[BrainVersion]:
        versions = [v for v, _ in self._versions[brain_id].values()]
        versions.sort(key=lambda v: v.version_number, reverse=True)
        return versions[:limit]

    async def get_next_version_number(self, brain_id: str) -> int:
        if not self._versions[brain_id]:
            return 1
        max_num = max(v.version_number for v, _ in self._versions[brain_id].values())
        return max_num + 1

    async def delete_version(self, brain_id: str, version_id: str) -> bool:
        if version_id in self._versions[brain_id]:
            del self._versions[brain_id][version_id]
            return True
        return False

    # ========== Cleanup ==========

    async def clear(self, brain_id: str) -> None:
        nodes_to_remove = [
            n for n in self._graph.nodes() if self._graph.nodes[n].get("brain_id") == brain_id
        ]
        self._graph.remove_nodes_from(nodes_to_remove)

        self._neurons[brain_id].clear()
        self._synapses[brain_id].clear()
        self._fibers[brain_id].clear()
        self._states[brain_id].clear()
        self._typed_memories[brain_id].clear()
        self._projects[brain_id].clear()
        self._co_activations[brain_id].clear()
        self._action_events[brain_id].clear()
        self._brains.pop(brain_id, None)
        # Note: versions are NOT cleared — they survive rollbacks (matches SQLite behavior)
