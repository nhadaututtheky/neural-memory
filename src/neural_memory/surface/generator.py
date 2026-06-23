"""Surface generator: extracts top knowledge from brain.db into a KnowledgeSurface.

Algorithm:
1. Select top neurons by composite score (activation * 0.4 + recency * 0.3 + connections * 0.2 + priority * 0.1)
2. Extract GRAPH edges from synapses between top neurons
3. Build CLUSTERS from entity co-occurrence in fibers
4. Extract SIGNALS from high-priority recent fibers
5. Compute DEPTH MAP based on surface coverage per node
6. Trim to token budget
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from neural_memory.surface.models import (
    Cluster,
    DepthHint,
    DepthLevel,
    GraphEntry,
    KnowledgeSurface,
    Signal,
    SignalLevel,
    SurfaceEdge,
    SurfaceFrontmatter,
    SurfaceMeta,
    SurfaceNode,
)
from neural_memory.surface.token_budget import trim_to_budget
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from datetime import datetime

    from neural_memory.core.neuron import Neuron, NeuronState

logger = logging.getLogger(__name__)

# ── Synapse type → human-readable edge label ───────

_SYNAPSE_EDGE_MAP: dict[str, str] = {
    "caused_by": "caused_by",
    "leads_to": "led_to",
    "enables": "enables",
    "prevents": "prevents",
    "related_to": "related_to",
    "co_occurs": "co_occurs",
    "is_a": "is_a",
    "has_property": "has_property",
    "involves": "involves",
    "contradicts": "contradicts",
    "resolved_by": "resolved_by",
    "similar_to": "similar_to",
    "contains": "contains",
    "before": "before",
    "after": "after",
    "happened_at": "happened_at",
    "evidence_for": "evidence_for",
    "evidence_against": "evidence_against",
    "supersedes": "supersedes",
}

# ── Neuron type → surface node ID prefix ───────────

_TYPE_PREFIX: dict[str, str] = {
    "concept": "c",
    "entity": "e",
    "action": "a",
    "state": "s",
    "intent": "i",
    "hypothesis": "h",
    "prediction": "pr",
    "time": "t",
    "spatial": "sp",
    "sensory": "sn",
    "schema": "sc",
}

# Fiber metadata keys that indicate memory type for ID prefix
_MEMORY_TYPE_PREFIX: dict[str, str] = {
    "decision": "d",
    "error": "er",
    "fact": "f",
    "preference": "p",
    "insight": "in",
    "workflow": "w",
    "instruction": "ins",
    "todo": "td",
}

# Synapse types to skip when building GRAPH edges (too noisy)
_SKIP_SYNAPSE_TYPES = {"happened_at", "before", "after", "during", "contains"}


class SurfaceGenerator:
    """Generates a KnowledgeSurface from brain.db storage."""

    def __init__(
        self,
        storage: Any,
        brain_name: str = "default",
        token_budget: int = 1200,
        max_graph_nodes: int = 30,
        max_signals: int = 5,
    ) -> None:
        self._storage = storage
        self._brain_name = brain_name
        self._token_budget = token_budget
        self._max_graph_nodes = max_graph_nodes
        self._max_signals = max_signals
        # neuron UUID -> assigned surface node id (e.g. 'd1'), built during
        # _extract_graph and reused by _build_clusters so clusters reference
        # short surface ids rather than raw UUIDs (#40/#41).
        self._node_id_map: dict[str, str] = {}

    async def generate(self) -> KnowledgeSurface:
        """Generate a complete KnowledgeSurface from brain.db.

        Returns:
            A KnowledgeSurface trimmed to the token budget.
        """
        now = utcnow()
        self._node_id_map = {}  # reset per-generation neuron->surface-id map

        # Step 1: Select top neurons by composite score
        scored_neurons = await self._select_top_neurons(now)

        neuron_ids = [n.id for n, _score, _state in scored_neurons]

        # Step 2: Extract GRAPH from synapses between top neurons
        graph_entries = await self._extract_graph(scored_neurons) if scored_neurons else []

        # Step 3: Build CLUSTERS from fiber co-occurrence
        clusters = await self._build_clusters(neuron_ids) if neuron_ids else []

        # Step 4: Extract SIGNALS from high-priority recent fibers (always run)
        signals = await self._extract_signals(now)

        # Step 5: Compute DEPTH MAP
        depth_map = (
            await self._compute_depth_map(graph_entries, neuron_ids) if graph_entries else []
        )

        # Step 6: Build META
        meta = await self._build_meta(now, graph_entries)

        # Assemble surface
        stats = await self._get_stats()
        surface = KnowledgeSurface(
            frontmatter=SurfaceFrontmatter(
                brain=self._brain_name,
                updated=now.isoformat(timespec="seconds"),
                neurons=stats.get("neuron_count", 0),
                synapses=stats.get("synapse_count", 0),
                token_budget=self._token_budget,
            ),
            graph=tuple(graph_entries),
            clusters=tuple(clusters),
            signals=tuple(signals),
            depth_map=tuple(depth_map),
            meta=meta,
        )

        # Step 7: Trim to budget
        return trim_to_budget(surface, self._token_budget)

    async def _select_top_neurons(
        self,
        now: datetime,
    ) -> list[tuple[Neuron, float, NeuronState | None]]:
        """Select top neurons by composite score.

        Score = activation * 0.4 + recency * 0.3 + connections * 0.2 + priority * 0.1
        """
        # Fetch meaningful neuron types (skip TIME, SPATIAL, SENSORY — too granular)
        from neural_memory.core.neuron import NeuronType

        meaningful_types = [
            NeuronType.ENTITY,
            NeuronType.CONCEPT,
            NeuronType.ACTION,
            NeuronType.STATE,
            NeuronType.INTENT,
            NeuronType.HYPOTHESIS,
        ]

        all_neurons: list[Neuron] = []
        seen_ids: set[str] = set()
        for ntype in meaningful_types:
            neurons = await self._storage.find_neurons(
                type=ntype, limit=min(self._max_graph_nodes * 3, 200)
            )
            for n in neurons:
                if n.id not in seen_ids:
                    seen_ids.add(n.id)
                    all_neurons.append(n)

        if not all_neurons:
            return []

        # Fetch activation states
        neuron_ids = [n.id for n in all_neurons]
        states = await self._storage.get_neuron_states_batch(neuron_ids)

        # Fetch connection counts (outgoing synapses)
        synapse_map = await self._storage.get_synapses_for_neurons(neuron_ids, direction="out")

        # Score each neuron
        scored: list[tuple[Neuron, float, NeuronState | None]] = []
        for neuron in all_neurons:
            state = states.get(neuron.id)
            activation = state.activation_level if state else 0.0

            # Recency: days since creation, mapped to 0-1 (1 = today, 0 = 90+ days)
            age_days = (now - neuron.created_at).total_seconds() / 86400
            recency = max(0.0, 1.0 - age_days / 90.0)

            # Connection count normalized
            conn_count = len(synapse_map.get(neuron.id, []))
            connections = min(conn_count / 10.0, 1.0)

            # Priority from metadata (fiber-level priority if available)
            priority_raw = neuron.metadata.get("priority", 5)
            priority = min(int(priority_raw), 10) / 10.0 if priority_raw else 0.5

            score = activation * 0.4 + recency * 0.3 + connections * 0.2 + priority * 0.1
            scored.append((neuron, score, state))

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self._max_graph_nodes]

    async def _extract_graph(
        self,
        scored_neurons: list[tuple[Neuron, float, NeuronState | None]],
    ) -> list[GraphEntry]:
        """Extract GRAPH entries from scored neurons + their synapses."""
        neuron_map = {n.id: (n, score) for n, score, _state in scored_neurons}
        neuron_ids = list(neuron_map.keys())

        # Fetch outgoing synapses for all top neurons
        synapse_map = await self._storage.get_synapses_for_neurons(neuron_ids, direction="out")

        # Also fetch target neurons we might reference
        all_target_ids: set[str] = set()
        for synapses in synapse_map.values():
            for syn in synapses:
                all_target_ids.add(syn.target_id)
        # Only fetch targets not already in our neuron map
        missing_ids = list(all_target_ids - set(neuron_ids))
        target_neurons: dict[str, Neuron] = {}
        if missing_ids:
            target_neurons = await self._storage.get_neurons_batch(missing_ids[:100])

        # --- Pass 1: assign every surface node ID up front so that BOTH
        # forward and backward edge references resolve. Iterating in score
        # order means a higher-scored source can point at a lower-scored
        # target whose entry doesn't exist yet; building the full
        # neuron_id -> node_id map first fixes that (#40).
        id_counter: dict[str, int] = defaultdict(int)
        nodes: list[tuple[SurfaceNode, float]] = []
        for neuron, score, _state in scored_neurons:
            node_id = self._make_node_id(neuron, id_counter)
            node = SurfaceNode(
                id=node_id,
                content=self._truncate(neuron.content, 80),
                node_type=neuron.type.value,
                priority=self._score_to_priority(score),
                neuron_id=neuron.id,
            )
            nodes.append((node, score))
            self._node_id_map[neuron.id] = node_id

        # --- Pass 2: build edges now that the full id map exists.
        entries: list[GraphEntry] = []
        for node, _score in nodes:
            source_id = node.neuron_id or ""

            edges: list[SurfaceEdge] = []
            for syn in synapse_map.get(source_id, []):
                if syn.type.value in _SKIP_SYNAPSE_TYPES:
                    continue
                edge_type = _SYNAPSE_EDGE_MAP.get(syn.type.value, syn.type.value)

                # Try to find target in our scored set first
                target_id_ref = None
                target_text = ""

                if syn.target_id in neuron_map:
                    target_n = neuron_map[syn.target_id][0]
                    # Resolve via the full map — works for forward refs too.
                    target_id_ref = self._node_id_map.get(syn.target_id)
                    target_text = self._truncate(target_n.content, 60)
                elif syn.target_id in target_neurons:
                    target_n = target_neurons[syn.target_id]
                    target_text = self._truncate(target_n.content, 60)
                else:
                    target_text = syn.metadata.get("description", "")

                if target_text:
                    edges.append(
                        SurfaceEdge(
                            edge_type=edge_type,
                            target_id=target_id_ref,
                            target_text=target_text,
                        )
                    )

            entries.append(GraphEntry(node=node, edges=tuple(edges[:5])))

        return entries

    async def _build_clusters(
        self,
        neuron_ids: list[str],
    ) -> list[Cluster]:
        """Build CLUSTERS from entity co-occurrence in fibers."""
        # Find fibers that contain our top neurons
        fiber_neurons: dict[str, set[str]] = defaultdict(set)

        for nid in neuron_ids[:50]:  # Cap to avoid too many queries
            try:
                fibers = await self._storage.find_fibers(contains_neuron=nid, limit=10)
                for fiber in fibers:
                    fiber_neurons[fiber.id].update(fiber.neuron_ids & set(neuron_ids))
            except Exception:
                continue

        # Group neurons that co-occur in 2+ fibers
        co_occurrence: dict[str, set[str]] = defaultdict(set)
        for nids in fiber_neurons.values():
            for nid in nids:
                co_occurrence[nid].update(nids)

        # Build clusters via connected components (simple union-find)
        visited: set[str] = set()
        clusters: list[set[str]] = []

        for nid in co_occurrence:
            if nid in visited:
                continue
            # BFS from this neuron
            component: set[str] = set()
            queue = [nid]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in co_occurrence.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(component) >= 2:
                clusters.append(component)

        # Fetch neuron content for naming
        all_cluster_ids = {nid for c in clusters for nid in c}
        neuron_batch = await self._storage.get_neurons_batch(list(all_cluster_ids))

        result: list[Cluster] = []
        for i, component in enumerate(clusters[:8]):  # Max 8 clusters
            # Name by most common neuron content keyword
            contents = [neuron_batch[nid].content for nid in component if nid in neuron_batch]
            cluster_name = self._infer_cluster_name(contents, i)

            # Map neuron UUIDs to their assigned surface node IDs (built in
            # _extract_graph). Drop neurons that never made it into the graph
            # so cluster refs always point at real short surface ids — keeps
            # token_budget._trim_lowest_priority_graph able to scrub them and
            # keeps the serialized CLUSTERS section consistent (#41).
            surface_ids = sorted(
                self._node_id_map[nid] for nid in component if nid in self._node_id_map
            )
            node_ids = tuple(surface_ids[:5])  # Cap refs per cluster
            if not node_ids:
                continue

            result.append(
                Cluster(
                    name=cluster_name,
                    node_ids=node_ids,
                    description=self._summarize_cluster(contents),
                )
            )

        return result

    async def _extract_signals(
        self,
        now: datetime,
    ) -> list[Signal]:
        """Extract SIGNALS from high-priority recent fibers."""
        signals: list[Signal] = []

        # Get recent fibers sorted by salience
        try:
            recent_fibers = await self._storage.get_fibers(
                limit=50, order_by="created_at", descending=True
            )
        except Exception:
            return signals

        for fiber in recent_fibers:
            if len(signals) >= self._max_signals:
                break

            age_days = (now - fiber.created_at).total_seconds() / 86400
            salience = fiber.salience
            memory_type = fiber.metadata.get("memory_type", "")

            # Urgent: high salience + recent
            if salience >= 0.7 and age_days < 7:
                signals.append(
                    Signal(
                        level=SignalLevel.URGENT,
                        text=self._truncate(self._fiber_summary(fiber), 100),
                    )
                )
            # Watching: moderate salience + somewhat recent
            elif salience >= 0.4 and age_days < 14:
                signals.append(
                    Signal(
                        level=SignalLevel.WATCHING,
                        text=self._truncate(self._fiber_summary(fiber), 100),
                    )
                )
            # Uncertain: todos or low-confidence items
            elif memory_type == "todo":
                signals.append(
                    Signal(
                        level=SignalLevel.UNCERTAIN,
                        text=self._truncate(self._fiber_summary(fiber), 100),
                    )
                )

        return signals

    async def _compute_depth_map(
        self,
        graph_entries: list[GraphEntry],
        neuron_ids: list[str],
    ) -> list[DepthHint]:
        """Compute DEPTH MAP based on surface coverage."""
        hints: list[DepthHint] = []

        for entry in graph_entries:
            edge_count = len(entry.edges)
            nid = entry.node.neuron_id or ""

            # Count total synapses in brain.db for this neuron
            total_synapses = 0
            if nid:
                try:
                    synapses = await self._storage.get_synapses(source_id=nid)
                    total_synapses = len(synapses)
                except Exception:
                    pass

            # Determine depth level
            if edge_count >= 3 and edge_count >= total_synapses * 0.5:
                level = DepthLevel.SUFFICIENT
                context = f"{edge_count} edges on surface"
            elif total_synapses > edge_count * 3:
                level = DepthLevel.NEEDS_DEEP
                context = (
                    f"{total_synapses} synapses in brain.db, "
                    f'recall "{self._truncate(entry.node.content, 40)}"'
                )
            elif total_synapses > edge_count:
                level = DepthLevel.NEEDS_DETAIL
                context = (
                    f"{total_synapses} total synapses, "
                    f'recall "{self._truncate(entry.node.content, 40)}"'
                )
            else:
                level = DepthLevel.SUFFICIENT
                context = f"{edge_count} edges, fully represented"

            hints.append(
                DepthHint(
                    node_id=entry.node.id,
                    level=level,
                    context=context,
                )
            )

        return hints

    async def _build_meta(
        self,
        now: datetime,
        graph_entries: list[GraphEntry],
    ) -> SurfaceMeta:
        """Build META section with brain stats."""
        stats = await self._get_stats()
        total_neurons = stats.get("neuron_count", 1)
        surface_neurons = len(graph_entries)

        coverage = min(surface_neurons / max(total_neurons, 1), 1.0)

        # Staleness: average age of surface nodes mapped to 0-1
        ages: list[float] = []
        for entry in graph_entries:
            if entry.node.neuron_id:
                try:
                    neurons = await self._storage.get_neurons_batch([entry.node.neuron_id])
                    if entry.node.neuron_id in neurons:
                        n = neurons[entry.node.neuron_id]
                        age = (now - n.created_at).total_seconds() / 86400
                        ages.append(min(age / 90.0, 1.0))
                except Exception:
                    pass
        staleness = sum(ages) / max(len(ages), 1)

        # Top entities: most-connected nodes on surface
        top = sorted(graph_entries, key=lambda e: len(e.edges), reverse=True)
        top_entities = tuple(e.node.content.split()[0] for e in top[:5] if e.node.content)

        return SurfaceMeta(
            coverage=round(coverage, 2),
            staleness=round(staleness, 2),
            last_consolidation=now.isoformat(timespec="seconds"),
            top_entities=top_entities,
        )

    async def _get_stats(self) -> dict[str, int]:
        """Get brain stats, handling missing method gracefully."""
        try:
            result: dict[str, int] = await self._storage.get_stats(self._brain_name)
            return result
        except Exception:
            return {"neuron_count": 0, "synapse_count": 0, "fiber_count": 0}

    def _empty_surface(self, now: datetime) -> KnowledgeSurface:
        """Return a valid but empty surface for empty brains."""
        return KnowledgeSurface(
            frontmatter=SurfaceFrontmatter(
                brain=self._brain_name,
                updated=now.isoformat(timespec="seconds"),
            ),
            meta=SurfaceMeta(
                last_consolidation=now.isoformat(timespec="seconds"),
            ),
        )

    # ── Helpers ────────────────────────────────────

    def _make_node_id(
        self,
        neuron: Neuron,
        counter: dict[str, int],
    ) -> str:
        """Generate a short surface node ID like 'd1', 'f2', 'c3'."""
        ntype = neuron.type.value
        prefix = _TYPE_PREFIX.get(ntype, "n")

        # Check fiber metadata for memory type (decision, error, etc.)
        mem_type = neuron.metadata.get("memory_type", "")
        if mem_type in _MEMORY_TYPE_PREFIX:
            prefix = _MEMORY_TYPE_PREFIX[mem_type]

        counter[prefix] = counter.get(prefix, 0) + 1
        return f"{prefix}{counter[prefix]}"

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Truncate text to max_len, preserving word boundaries."""
        if len(text) <= max_len:
            return text
        truncated = text[: max_len - 1]
        # Try to break at last space
        last_space = truncated.rfind(" ")
        if last_space > max_len // 2:
            return truncated[:last_space] + "…"
        return truncated + "…"

    @staticmethod
    def _score_to_priority(score: float) -> int:
        """Map 0.0-1.0 composite score to 1-10 priority."""
        return max(1, min(10, round(score * 10)))

    @staticmethod
    def _infer_cluster_name(contents: list[str], index: int) -> str:
        """Infer a cluster name from neuron contents."""
        if not contents:
            return f"topic{index + 1}"
        # Use first significant word from the most common content
        words: dict[str, int] = defaultdict(int)
        stop_words = {"the", "a", "an", "is", "was", "for", "to", "in", "of", "and", "or"}
        for content in contents:
            for word in content.lower().split()[:5]:
                clean = word.strip(".,;:!?\"'()[]")
                if clean and len(clean) > 2 and clean not in stop_words:
                    words[clean] += 1
        if words:
            return max(words, key=words.get)  # type: ignore[arg-type]
        return f"topic{index + 1}"

    @staticmethod
    def _summarize_cluster(contents: list[str]) -> str:
        """Create a short description from cluster neuron contents."""
        if not contents:
            return ""
        # Take first 3 unique content snippets, join
        unique = list(dict.fromkeys(c[:40] for c in contents))[:3]
        return ", ".join(unique)

    @staticmethod
    def _fiber_summary(fiber: Any) -> str:
        """Extract a summary from a fiber object."""
        if hasattr(fiber, "summary") and fiber.summary:
            summary: str = fiber.summary
            return summary
        # Fallback: use tags
        tags: set[str] = getattr(fiber, "tags", set())
        if tags:
            return f"[{', '.join(sorted(tags)[:5])}]"
        return f"Memory from {getattr(fiber, 'created_at', 'unknown')}"
