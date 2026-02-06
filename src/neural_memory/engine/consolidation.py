"""Memory consolidation engine — prune, merge, and summarize memories.

Provides automated memory maintenance:
- Prune: Remove dead synapses and orphan neurons
- Merge: Combine overlapping fibers
- Summarize: Create concept neurons for topic clusters
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import uuid4

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage


class ConsolidationStrategy(StrEnum):
    """Available consolidation strategies."""

    PRUNE = "prune"
    MERGE = "merge"
    SUMMARIZE = "summarize"
    ALL = "all"


@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration for consolidation operations."""

    prune_weight_threshold: float = 0.05
    prune_min_inactive_days: float = 7.0
    prune_isolated_neurons: bool = True
    merge_overlap_threshold: float = 0.5
    merge_max_fiber_size: int = 50
    summarize_min_cluster_size: int = 3
    summarize_tag_overlap_threshold: float = 0.4


@dataclass(frozen=True)
class MergeDetail:
    """Details of a single fiber merge operation."""

    original_fiber_ids: tuple[str, ...]
    merged_fiber_id: str
    neuron_count: int
    reason: str


@dataclass
class ConsolidationReport:
    """Report of consolidation operation results."""

    started_at: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    synapses_pruned: int = 0
    neurons_pruned: int = 0
    fibers_merged: int = 0
    fibers_removed: int = 0
    fibers_created: int = 0
    summaries_created: int = 0
    merge_details: list[MergeDetail] = field(default_factory=list)
    dry_run: bool = False

    def summary(self) -> str:
        """Generate human-readable summary."""
        mode = " (dry run)" if self.dry_run else ""
        lines = [
            f"Consolidation Report{mode} ({self.started_at.strftime('%Y-%m-%d %H:%M')})",
            f"  Synapses pruned: {self.synapses_pruned}",
            f"  Neurons pruned: {self.neurons_pruned}",
            f"  Fibers merged: {self.fibers_merged} -> {self.fibers_created} new",
            f"  Fibers removed: {self.fibers_removed}",
            f"  Summaries created: {self.summaries_created}",
            f"  Duration: {self.duration_ms:.1f}ms",
        ]
        if self.merge_details:
            lines.append("  Merge details:")
            for detail in self.merge_details:
                lines.append(
                    f"    {len(detail.original_fiber_ids)} fibers -> {detail.merged_fiber_id[:8]}... "
                    f"({detail.neuron_count} neurons, {detail.reason})"
                )
        return "\n".join(lines)


class ConsolidationEngine:
    """Engine for memory consolidation operations.

    Supports three strategies:
    - prune: Remove weak synapses and orphan neurons
    - merge: Combine overlapping fibers
    - summarize: Create concept neurons for topic clusters
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: ConsolidationConfig | None = None,
    ) -> None:
        self._storage = storage
        self._config = config or ConsolidationConfig()

    async def run(
        self,
        strategies: list[ConsolidationStrategy] | None = None,
        dry_run: bool = False,
        reference_time: datetime | None = None,
    ) -> ConsolidationReport:
        """Run consolidation with specified strategies.

        Args:
            strategies: List of strategies to run (default: all)
            dry_run: If True, calculate but don't apply changes
            reference_time: Reference time for age calculations

        Returns:
            ConsolidationReport with operation statistics
        """
        if strategies is None:
            strategies = [ConsolidationStrategy.ALL]

        reference_time = reference_time or datetime.now()
        report = ConsolidationReport(started_at=reference_time, dry_run=dry_run)
        start = time.perf_counter()

        run_all = ConsolidationStrategy.ALL in strategies

        if run_all or ConsolidationStrategy.PRUNE in strategies:
            await self._prune(report, reference_time, dry_run)

        if run_all or ConsolidationStrategy.MERGE in strategies:
            await self._merge(report, dry_run)

        if run_all or ConsolidationStrategy.SUMMARIZE in strategies:
            await self._summarize(report, dry_run)

        report.duration_ms = (time.perf_counter() - start) * 1000
        return report

    async def _prune(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Prune weak synapses and orphan neurons."""
        # Ensure brain context is set (validates state)
        self._storage._get_brain_id()

        # Get all synapses
        all_synapses = await self._storage.get_synapses()
        pruned_synapse_ids: set[str] = set()

        for synapse in all_synapses:
            # Apply time-based decay before checking weight threshold
            decayed = synapse.time_decay(reference_time=reference_time)
            should_prune = decayed.weight < self._config.prune_weight_threshold

            # Check inactivity
            if synapse.last_activated is not None:
                days_inactive = (reference_time - synapse.last_activated).total_seconds() / 86400
                should_prune = (
                    should_prune and days_inactive >= self._config.prune_min_inactive_days
                )
            elif synapse.created_at is not None:
                days_since_creation = (reference_time - synapse.created_at).total_seconds() / 86400
                should_prune = (
                    should_prune and days_since_creation >= self._config.prune_min_inactive_days
                )

            if should_prune:
                # Protect bridge synapses (only connection between source and target)
                if synapse.weight >= 0.02:
                    neighbors = await self._storage.get_neighbors(synapse.source_id)
                    neighbor_ids = {n.id for n in neighbors}
                    if synapse.target_id in neighbor_ids and len(neighbor_ids) <= 1:
                        continue  # Bridge synapse — don't prune

                pruned_synapse_ids.add(synapse.id)
                report.synapses_pruned += 1
                if not dry_run:
                    await self._storage.delete_synapse(synapse.id)

        if not pruned_synapse_ids:
            return

        # Update fiber synapse_ids to remove pruned refs
        fibers = await self._storage.get_fibers(limit=10000)
        for fiber in fibers:
            removed = fiber.synapse_ids & pruned_synapse_ids
            if removed and not dry_run:
                updated_fiber = Fiber(
                    id=fiber.id,
                    neuron_ids=fiber.neuron_ids,
                    synapse_ids=fiber.synapse_ids - pruned_synapse_ids,
                    anchor_neuron_id=fiber.anchor_neuron_id,
                    pathway=fiber.pathway,
                    conductivity=fiber.conductivity,
                    last_conducted=fiber.last_conducted,
                    time_start=fiber.time_start,
                    time_end=fiber.time_end,
                    coherence=fiber.coherence,
                    salience=fiber.salience,
                    frequency=fiber.frequency,
                    summary=fiber.summary,
                    tags=fiber.tags,
                    metadata=fiber.metadata,
                    created_at=fiber.created_at,
                )
                await self._storage.update_fiber(updated_fiber)

        # Find orphan neurons (no synapses, not a fiber anchor)
        if not self._config.prune_isolated_neurons:
            return

        remaining_synapses = await self._storage.get_synapses()
        connected_neuron_ids: set[str] = set()
        for syn in remaining_synapses:
            connected_neuron_ids.add(syn.source_id)
            connected_neuron_ids.add(syn.target_id)

        anchor_neuron_ids: set[str] = set()
        fibers_after = await self._storage.get_fibers(limit=10000)
        for fiber in fibers_after:
            anchor_neuron_ids.add(fiber.anchor_neuron_id)

        all_neurons = await self._storage.find_neurons(limit=100000)
        for neuron in all_neurons:
            if neuron.id not in connected_neuron_ids and neuron.id not in anchor_neuron_ids:
                report.neurons_pruned += 1
                if not dry_run:
                    await self._storage.delete_neuron(neuron.id)

    async def _merge(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Merge overlapping fibers using Jaccard similarity."""
        fibers = await self._storage.get_fibers(limit=10000)
        if len(fibers) < 2:
            return

        # Build adjacency using Jaccard similarity
        fiber_list = list(fibers)
        n = len(fiber_list)

        # Union-Find
        parent: dict[int, int] = {i: i for i in range(n)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Pairwise Jaccard on neuron_ids
        for i in range(n):
            if len(fiber_list[i].neuron_ids) > self._config.merge_max_fiber_size:
                continue
            for j in range(i + 1, n):
                if len(fiber_list[j].neuron_ids) > self._config.merge_max_fiber_size:
                    continue

                set_a = fiber_list[i].neuron_ids
                set_b = fiber_list[j].neuron_ids
                intersection = len(set_a & set_b)
                union_size = len(set_a | set_b)

                if union_size > 0:
                    jaccard = intersection / union_size
                    # Lower threshold for temporally-close fibers (same session)
                    time_diff = abs(
                        (fiber_list[i].created_at - fiber_list[j].created_at).total_seconds()
                    )
                    effective_threshold = (
                        self._config.merge_overlap_threshold * 0.6
                        if time_diff < 3600
                        else self._config.merge_overlap_threshold
                    )
                    if jaccard >= effective_threshold:
                        union(i, j)

        # Group fibers by root
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # Merge groups with more than 1 member
        for _root, members in groups.items():
            if len(members) < 2:
                continue

            member_fibers = [fiber_list[i] for i in members]

            # Create merged fiber
            merged_neuron_ids: set[str] = set()
            merged_synapse_ids: set[str] = set()
            merged_tags: set[str] = set()
            max_salience = 0.0
            best_anchor = member_fibers[0].anchor_neuron_id
            best_frequency = 0

            for fiber in member_fibers:
                merged_neuron_ids |= fiber.neuron_ids
                merged_synapse_ids |= fiber.synapse_ids
                merged_tags |= fiber.tags
                if fiber.salience > max_salience:
                    max_salience = fiber.salience
                if fiber.frequency > best_frequency:
                    best_frequency = fiber.frequency
                    best_anchor = fiber.anchor_neuron_id

            merged_fiber_id = str(uuid4())
            merged_fiber = Fiber(
                id=merged_fiber_id,
                neuron_ids=merged_neuron_ids,
                synapse_ids=merged_synapse_ids,
                anchor_neuron_id=best_anchor,
                pathway=[best_anchor],
                salience=max_salience,
                frequency=best_frequency,
                tags=merged_tags,
                summary=f"Merged from {len(member_fibers)} fibers",
                metadata={"merged_from": [f.id for f in member_fibers]},
                created_at=min(f.created_at for f in member_fibers),
            )

            report.fibers_merged += len(member_fibers)
            report.fibers_created += 1
            report.merge_details.append(
                MergeDetail(
                    original_fiber_ids=tuple(f.id for f in member_fibers),
                    merged_fiber_id=merged_fiber_id,
                    neuron_count=len(merged_neuron_ids),
                    reason="neuron_overlap",
                )
            )

            if not dry_run:
                # Delete originals
                for fiber in member_fibers:
                    await self._storage.delete_fiber(fiber.id)
                    report.fibers_removed += 1
                # Add merged
                await self._storage.add_fiber(merged_fiber)

    async def _summarize(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Create concept neurons for tag-based clusters."""
        fibers = await self._storage.get_fibers(limit=10000)
        if len(fibers) < self._config.summarize_min_cluster_size:
            return

        # Group fibers by tags using overlap threshold
        fiber_list = [f for f in fibers if f.tags]
        if len(fiber_list) < self._config.summarize_min_cluster_size:
            return

        n = len(fiber_list)

        # Union-Find for tag clustering
        parent: dict[int, int] = {i: i for i in range(n)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                tags_a = fiber_list[i].tags
                tags_b = fiber_list[j].tags
                intersection = len(tags_a & tags_b)
                union_size = len(tags_a | tags_b)
                if (
                    union_size > 0
                    and intersection / union_size >= self._config.summarize_tag_overlap_threshold
                ):
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        for _root, members in groups.items():
            if len(members) < self._config.summarize_min_cluster_size:
                continue

            cluster_fibers = [fiber_list[i] for i in members]

            # Build summary content from fiber summaries
            summaries = [f.summary for f in cluster_fibers if f.summary]
            all_tags: set[str] = set()
            for f in cluster_fibers:
                all_tags |= f.tags

            summary_content = (
                "; ".join(summaries[:10])
                if summaries
                else f"Cluster of {len(cluster_fibers)} memories"
            )
            tag_label = ", ".join(sorted(all_tags)[:5])
            concept_content = f"[{tag_label}] {summary_content[:200]}"

            if dry_run:
                report.summaries_created += 1
                continue

            # Create a CONCEPT neuron
            concept_neuron = Neuron.create(
                type=NeuronType.CONCEPT,
                content=concept_content,
                metadata={
                    "_consolidation": "summary",
                    "cluster_size": len(cluster_fibers),
                    "tags": sorted(all_tags),
                },
            )
            await self._storage.add_neuron(concept_neuron)

            # Link concept to anchor neurons via RELATED_TO synapses
            anchor_ids: set[str] = set()
            for fiber in cluster_fibers:
                anchor_ids.add(fiber.anchor_neuron_id)

            synapse_ids: set[str] = set()
            for anchor_id in list(anchor_ids)[:10]:
                synapse = Synapse.create(
                    source_id=concept_neuron.id,
                    target_id=anchor_id,
                    type=SynapseType.RELATED_TO,
                    weight=0.6,
                )
                await self._storage.add_synapse(synapse)
                synapse_ids.add(synapse.id)

            # Create summary fiber
            summary_fiber = Fiber.create(
                neuron_ids={concept_neuron.id} | anchor_ids,
                synapse_ids=synapse_ids,
                anchor_neuron_id=concept_neuron.id,
                summary=concept_content,
                tags=all_tags,
                metadata={
                    "_consolidation": "summary_fiber",
                    "source_fibers": [f.id for f in cluster_fibers],
                },
            )
            await self._storage.add_fiber(summary_fiber)
            report.summaries_created += 1
