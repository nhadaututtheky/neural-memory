"""Memory consolidation engine — prune, merge, and summarize memories.

Provides automated memory maintenance:
- Prune: Remove dead synapses and orphan neurons
- Merge: Combine overlapping fibers
- Summarize: Create concept neurons for topic clusters
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.abstraction import induce_abstraction
from neural_memory.engine.clustering import UnionFind
from neural_memory.utils.simhash import is_near_duplicate as _simhash_near_dup
from neural_memory.utils.timeutils import ensure_naive_utc, utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import TierConfig

logger = logging.getLogger(__name__)


# Causal synapses encode explanatory structure — never prune manually-created ones,
# even if weight decays below threshold. Inferred causal links (metadata._inferred=True)
# remain prunable since they may be noisy.
_CAUSAL_SYNAPSE_TYPES = frozenset(
    {
        SynapseType.CAUSED_BY,
        SynapseType.LEADS_TO,
        SynapseType.ENABLES,
        SynapseType.PREVENTS,
    }
)


class ConsolidationStrategy(StrEnum):
    """Available consolidation strategies."""

    DECAY = "decay"  # Ebbinghaus decay pass (activation + synapse weight)
    PRUNE = "prune"
    MERGE = "merge"
    SUMMARIZE = "summarize"
    MATURE = "mature"
    INFER = "infer"
    ENRICH = "enrich"
    DREAM = "dream"
    LEARN_HABITS = "learn_habits"
    DEDUP = "dedup"
    SEMANTIC_LINK = "semantic_link"
    COMPRESS = "compress"
    LIFECYCLE = "lifecycle"
    PROCESS_TOOL_EVENTS = "process_tool_events"
    ESSENCE_BACKFILL = "essence_backfill"
    DETECT_DRIFT = "detect_drift"
    SMART_MERGE = "smart_merge"  # Pro: HNSW-accelerated merge via plugin
    REPLAY = "replay"  # Hippocampal replay: LTP/LTD on recent fibers
    SCHEMA = "schema"  # Schema assimilation: bottom-up knowledge organization
    INTERFERENCE = "interference"  # Interference forgetting: memory competition
    ALL = "all"


@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration for consolidation operations."""

    prune_weight_threshold: float = 0.05
    prune_min_inactive_days: float = 7.0
    prune_isolated_neurons: bool = True
    prune_semantic_factor: float = 0.5
    bridge_weight_floor: float = 0.01
    merge_overlap_threshold: float = 0.5
    merge_max_fiber_size: int = 50
    merge_temporal_halflife_seconds: float = 7200.0
    # When a MERGE groups 5+ fibers, induce an abstract CONCEPT neuron
    # summarizing the cluster (CLS pattern — exemplars remain intact).
    enable_dynamic_abstraction: bool = True
    abstraction_cluster_min_size: int = 5
    summarize_min_cluster_size: int = 3
    summarize_tag_overlap_threshold: float = 0.4
    infer_co_activation_threshold: int = 3
    infer_window_days: int = 7
    infer_max_per_run: int = 50
    surface_regen_prune_threshold: int = 10
    maturation_fast_track_rehearsals: int = 10
    maturation_fast_track_time_days: float = 1.0
    strategy_timeout_seconds: float = 120.0
    total_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        """Validate config invariants to prevent degenerate behavior."""
        if self.prune_semantic_factor < 0 or self.prune_semantic_factor > 1:
            object.__setattr__(
                self, "prune_semantic_factor", max(0.0, min(1.0, self.prune_semantic_factor))
            )
        if self.bridge_weight_floor < 0:
            object.__setattr__(self, "bridge_weight_floor", 0.0)
        if self.maturation_fast_track_rehearsals < 1:
            object.__setattr__(self, "maturation_fast_track_rehearsals", 1)
        if self.maturation_fast_track_time_days <= 0:
            object.__setattr__(self, "maturation_fast_track_time_days", 0.1)


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

    started_at: datetime = field(default_factory=utcnow)
    duration_ms: float = 0.0
    synapses_pruned: int = 0
    neurons_pruned: int = 0
    fibers_merged: int = 0
    fibers_removed: int = 0
    fibers_created: int = 0
    summaries_created: int = 0
    concepts_created: int = 0
    stages_advanced: int = 0
    patterns_extracted: int = 0
    synapses_inferred: int = 0
    co_activations_pruned: int = 0
    synapses_enriched: int = 0
    dream_synapses_created: int = 0
    habits_learned: int = 0
    action_events_pruned: int = 0
    duplicates_found: int = 0
    semantic_synapses_created: int = 0
    memories_promoted: int = 0
    fibers_compressed: int = 0
    tokens_saved: int = 0
    neurons_reactivated: int = 0
    essences_generated: int = 0
    merge_details: list[MergeDetail] = field(default_factory=list)
    dry_run: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

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
            f"  Synapses inferred: {self.synapses_inferred}",
            f"  Co-activations pruned: {self.co_activations_pruned}",
            f"  Synapses enriched: {self.synapses_enriched}",
            f"  Dream synapses created: {self.dream_synapses_created}",
            f"  Habits learned: {self.habits_learned}",
            f"  Action events pruned: {self.action_events_pruned}",
            f"  Duplicates found: {self.duplicates_found}",
            f"  Semantic synapses: {self.semantic_synapses_created}",
            f"  Memories promoted: {self.memories_promoted}",
            f"  Concepts created: {self.concepts_created}",
            f"  Fibers compressed: {self.fibers_compressed}",
            f"  Tokens saved: {self.tokens_saved}",
            f"  Duration: {self.duration_ms:.1f}ms",
        ]
        if self.merge_details:
            lines.append("  Merge details:")
            for detail in self.merge_details:
                lines.append(
                    f"    {len(detail.original_fiber_ids)} fibers -> {detail.merged_fiber_id[:8]}... "
                    f"({detail.neuron_count} neurons, {detail.reason})"
                )

        # Add eligibility hints when nothing happened
        hints = self._eligibility_hints()
        if hints:
            lines.append("")
            lines.append("  Why nothing changed:")
            for hint in hints:
                lines.append(f"    - {hint}")

        return "\n".join(lines)

    def _eligibility_hints(self) -> list[str]:
        """Explain why consolidation produced no changes."""
        hints: list[str] = []
        total_changes = (
            self.synapses_pruned
            + self.neurons_pruned
            + self.fibers_merged
            + self.fibers_removed
            + self.summaries_created
            + self.synapses_inferred
            + self.synapses_enriched
            + self.dream_synapses_created
            + self.habits_learned
            + self.duplicates_found
            + self.semantic_synapses_created
            + self.fibers_compressed
            + self.stages_advanced
        )
        if total_changes > 0:
            return hints

        hints.append("Prune: synapses must be inactive for 7+ days with weight below 0.05")
        hints.append("Merge: fibers need >50% neuron overlap (Jaccard) and <=50 neurons each")
        hints.append("Summarize: need 3+ fibers sharing >40% tag overlap to form a cluster")
        hints.append("Mature: memories advance stages over time through repeated recall")
        hints.append("Habits: need 3+ occurrences of the same action sequence within 30 days")
        hints.append(
            "Tip: store more memories and recall them over several days, then consolidate again"
        )
        return hints


class ConsolidationEngine:
    """Engine for memory consolidation operations.

    Supports strategies: prune, merge, summarize, mature, infer, enrich,
    dream, learn_habits, dedup.

    Strategies are grouped into dependency tiers and run in parallel
    within each tier sequentially (to avoid stale data).
    """

    # Dependency tiers — strategies within a tier are independent and
    # can safely run concurrently. Tiers execute sequentially because
    # later tiers depend on results from earlier ones.
    STRATEGY_TIERS: tuple[frozenset[ConsolidationStrategy], ...] = (
        # Tier 0: DECAY runs first so decayed activation can drop items
        # below prune thresholds in the next tier.
        frozenset({ConsolidationStrategy.DECAY}),
        frozenset(
            {
                ConsolidationStrategy.PRUNE,
                ConsolidationStrategy.LEARN_HABITS,
                ConsolidationStrategy.DEDUP,
                ConsolidationStrategy.PROCESS_TOOL_EVENTS,
            }
        ),
        frozenset(
            {
                ConsolidationStrategy.MERGE,
                ConsolidationStrategy.SMART_MERGE,
                ConsolidationStrategy.INTERFERENCE,
                ConsolidationStrategy.MATURE,
                ConsolidationStrategy.COMPRESS,
                ConsolidationStrategy.LIFECYCLE,
            }
        ),
        frozenset(
            {
                ConsolidationStrategy.SUMMARIZE,
                ConsolidationStrategy.INFER,
                ConsolidationStrategy.SCHEMA,
                ConsolidationStrategy.ESSENCE_BACKFILL,
            }
        ),
        frozenset(
            {
                ConsolidationStrategy.ENRICH,
                ConsolidationStrategy.DREAM,
                ConsolidationStrategy.REPLAY,
            }
        ),
        frozenset(
            {
                ConsolidationStrategy.SEMANTIC_LINK,
                ConsolidationStrategy.DETECT_DRIFT,
            }
        ),
    )

    def __init__(
        self,
        storage: NeuralStorage,
        config: ConsolidationConfig | None = None,
        dream_decay_multiplier: float = 3.0,
        tier_config: TierConfig | None = None,
    ) -> None:
        self._storage = storage
        self._config = config or ConsolidationConfig()
        self._dream_decay_multiplier = dream_decay_multiplier
        self._tier_config = tier_config

    async def _run_strategy(
        self,
        strategy: ConsolidationStrategy,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Dispatch a single strategy to its implementation method."""
        dispatch: dict[ConsolidationStrategy, Callable[[], Awaitable[None]]] = {
            ConsolidationStrategy.DECAY: lambda: self._decay(report, reference_time, dry_run),
            ConsolidationStrategy.PRUNE: lambda: self._prune(report, reference_time, dry_run),
            ConsolidationStrategy.MERGE: lambda: self._merge(report, dry_run),
            ConsolidationStrategy.SUMMARIZE: lambda: self._summarize(report, dry_run),
            ConsolidationStrategy.MATURE: lambda: self._mature(report, reference_time, dry_run),
            ConsolidationStrategy.INFER: lambda: self._infer(report, reference_time, dry_run),
            ConsolidationStrategy.ENRICH: lambda: self._enrich(report, dry_run),
            ConsolidationStrategy.DREAM: lambda: self._dream(report, dry_run),
            ConsolidationStrategy.LEARN_HABITS: lambda: self._learn_habits(
                report, reference_time, dry_run
            ),
            ConsolidationStrategy.DEDUP: lambda: self._dedup(report, dry_run),
            ConsolidationStrategy.SEMANTIC_LINK: lambda: self._semantic_link(report, dry_run),
            ConsolidationStrategy.COMPRESS: lambda: self._compress(report, reference_time, dry_run),
            ConsolidationStrategy.LIFECYCLE: lambda: self._lifecycle(
                report, reference_time, dry_run
            ),
            ConsolidationStrategy.PROCESS_TOOL_EVENTS: lambda: self._process_tool_events(
                report, dry_run
            ),
            ConsolidationStrategy.DETECT_DRIFT: lambda: self._detect_drift(report, dry_run),
            ConsolidationStrategy.ESSENCE_BACKFILL: lambda: self._essence_backfill(report, dry_run),
            ConsolidationStrategy.SMART_MERGE: lambda: self._smart_merge_pro(report, dry_run),
            ConsolidationStrategy.REPLAY: lambda: self._replay(report, dry_run),
            ConsolidationStrategy.SCHEMA: lambda: self._schema(report, dry_run),
            ConsolidationStrategy.INTERFERENCE: lambda: self._interference(report, dry_run),
        }
        handler = dispatch.get(strategy)
        if handler is not None:
            await handler()

    async def run(
        self,
        strategies: list[ConsolidationStrategy] | None = None,
        dry_run: bool = False,
        reference_time: datetime | None = None,
    ) -> ConsolidationReport:
        """Run consolidation with specified strategies.

        Strategies are grouped into dependency tiers and run in parallel
        within each tier. Tiers execute sequentially so that later
        strategies can depend on results from earlier ones.

        Each strategy has a per-strategy timeout (default 120s) and the
        entire consolidation has a total timeout (default 600s) to prevent
        runaway execution.

        Args:
            strategies: List of strategies to run (default: all)
            dry_run: If True, calculate but don't apply changes
            reference_time: Reference time for age calculations

        Returns:
            ConsolidationReport with operation statistics
        """
        if strategies is None:
            strategies = [ConsolidationStrategy.ALL]

        reference_time = ensure_naive_utc(reference_time) if reference_time else utcnow()
        report = ConsolidationReport(started_at=reference_time, dry_run=dry_run)
        start = time.perf_counter()

        # Normalize string strategies to enum values (callers may pass raw strings)
        normalized: list[ConsolidationStrategy] = [
            s if isinstance(s, ConsolidationStrategy) else ConsolidationStrategy(s)
            for s in strategies
        ]

        run_all = ConsolidationStrategy.ALL in normalized
        requested: set[ConsolidationStrategy] = (
            {s for s in ConsolidationStrategy if s != ConsolidationStrategy.ALL}
            if run_all
            else set(normalized)
        )

        strategy_timeout = self._config.strategy_timeout_seconds
        total_timeout = self._config.total_timeout_seconds
        timed_out_strategies: list[str] = []

        for tier in self.STRATEGY_TIERS:
            tier_strategies = tier & requested
            if not tier_strategies:
                continue
            # Run strategies sequentially within each tier to avoid
            # stale data snapshots and shared mutable report races
            for strategy in tier_strategies:
                elapsed = time.perf_counter() - start
                if elapsed > total_timeout:
                    remaining = [s.value for s in tier_strategies if s >= strategy]
                    timed_out_strategies.extend(remaining)
                    logger.warning(
                        "Consolidation total timeout (%.0fs) reached after %.1fs, "
                        "skipping remaining strategies: %s",
                        total_timeout,
                        elapsed,
                        remaining,
                    )
                    break

                logger.info("Consolidation: starting %s", strategy.value)
                strategy_start = time.perf_counter()
                try:
                    await asyncio.wait_for(
                        self._run_strategy(strategy, report, reference_time, dry_run),
                        timeout=strategy_timeout,
                    )
                except TimeoutError:
                    strategy_elapsed = time.perf_counter() - strategy_start
                    logger.warning(
                        "Consolidation: %s timed out after %.1fs (limit: %.0fs)",
                        strategy.value,
                        strategy_elapsed,
                        strategy_timeout,
                    )
                    timed_out_strategies.append(strategy.value)
                finally:
                    strategy_elapsed = time.perf_counter() - strategy_start
                    logger.info(
                        "Consolidation: %s finished in %.1fs",
                        strategy.value,
                        strategy_elapsed,
                    )
            else:
                continue
            break  # break outer loop if inner broke due to total timeout

        if timed_out_strategies:
            report.extra["timed_out_strategies"] = timed_out_strategies

        # Auto-tier promotion/demotion (Pro feature, runs after standard strategies)
        await self._run_auto_tier(report, dry_run)

        # T4.5: Regenerate surface after consolidation if structural changes occurred
        if not dry_run and (
            report.fibers_merged > 0
            or report.fibers_removed > 0
            or report.fibers_compressed > 0
            or report.synapses_pruned >= self._config.surface_regen_prune_threshold
            or report.extra.get("stale_flagged", 0) > 0
            or report.extra.get("cold_demoted", 0) > 0
            or report.extra.get("lifecycle_states_updated", 0) > 0
        ):
            await self._regenerate_surface_after_consolidation()

        report.duration_ms = (time.perf_counter() - start) * 1000
        return report

    async def _run_auto_tier(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run auto-tier promotion/demotion if enabled (Pro feature).

        Runs after all standard consolidation strategies complete.
        Results are attached to report.extra["auto_tier"].
        """
        if self._tier_config is None or not self._tier_config.auto_enabled:
            return

        # Pro gate: auto-tier requires Pro license
        try:
            from neural_memory.plugins import has_pro

            if not has_pro():
                return
        except ImportError:
            return

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return

        try:
            from neural_memory.engine.tier_engine import TierEngine

            engine = TierEngine(self._storage, self._tier_config)
            tier_report = await engine.apply(brain_id, dry_run=dry_run)
            report.extra["auto_tier"] = tier_report.to_dict()
        except Exception as e:
            logger.error("Auto-tier failed during consolidation: %s", e, exc_info=True)
            report.extra["auto_tier"] = {"error": "auto-tier failed"}

    async def _regenerate_surface_after_consolidation(self) -> None:
        """T4.5: Regenerate Knowledge Surface to reflect post-consolidation state."""
        try:
            from neural_memory.surface.lifecycle import regenerate_surface

            brain_id = self._storage.current_brain_id
            if not brain_id:
                return
            brain = await self._storage.get_brain(brain_id)
            brain_name = brain.name if brain else "default"
            await regenerate_surface(storage=self._storage, brain_name=brain_name)
            logger.info("Surface regenerated after consolidation")
        except Exception:
            logger.warning("Surface regeneration after consolidation failed", exc_info=True)

    async def _decay(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Apply Ebbinghaus decay to neurons + synapses before pruning.

        Previously decay only ran on the scheduled 12h cycle, so between
        those cycles old memories kept their full activation and crowded
        out fresh ones during recall. Running decay as Tier 0 of
        consolidation keeps activation honest with the consolidation
        cadence (typically a few hours). Dispatch is idempotent:
        DecayManager's `min_age_days` constraint and per-access
        timestamps prevent double-decay.
        """
        if not self._storage.current_brain_id:
            return
        from neural_memory.engine.lifecycle import DecayManager

        manager = DecayManager()
        try:
            decay_report = await manager.apply_decay(
                self._storage, reference_time=reference_time, dry_run=dry_run
            )
        except Exception:
            logger.warning("Consolidation: decay pass failed (non-fatal)", exc_info=True)
            return

        report.extra["decay"] = {
            "neurons_processed": decay_report.neurons_processed,
            "neurons_decayed": decay_report.neurons_decayed,
            "synapses_processed": decay_report.synapses_processed,
            "synapses_decayed": decay_report.synapses_decayed,
            "duration_ms": decay_report.duration_ms,
        }

    async def _prune(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Prune weak synapses and orphan neurons."""
        logger = logging.getLogger(__name__)

        # Ensure brain context is set
        if not self._storage.current_brain_id:
            return

        # Get all synapses
        all_synapses = await self._storage.get_synapses()
        pruned_synapse_ids: set[str] = set()

        # Preload pinned neuron IDs to protect from pruning
        pinned_neuron_ids: set[str] = set()
        if hasattr(self._storage, "get_pinned_neuron_ids"):
            pinned_neuron_ids = await self._storage.get_pinned_neuron_ids()

        # Build fiber salience cache for high-salience protection
        fibers_for_salience = await self._storage.get_fibers(limit=10000)
        fiber_salience_cache: dict[str, list[Fiber]] = {}
        semantic_neuron_ids: set[str] = set()
        for fib in fibers_for_salience:
            if fib.salience > 0.8:
                for nid in fib.neuron_ids:
                    fiber_salience_cache.setdefault(nid, []).append(fib)
            # Track neurons in semantic-stage fibers (mature merged fibers)
            fib_meta = fib.metadata or {}
            if fib_meta.get("_stage") == "semantic":
                semantic_neuron_ids.update(fib.neuron_ids)

        # Pre-fetch neighbor counts for bridge detection (avoid N+1 queries)
        # Collect all unique source neuron IDs from synapses eligible for pruning
        bridge_floor = self._config.bridge_weight_floor
        candidate_source_ids = list({s.source_id for s in all_synapses if s.weight >= bridge_floor})
        neighbor_synapses_map: dict[str, list[Synapse]] = {}
        if candidate_source_ids:
            neighbor_synapses_map = await self._storage.get_synapses_for_neurons(
                candidate_source_ids, direction="out"
            )

        for syn_idx, synapse in enumerate(all_synapses):
            if syn_idx % 500 == 0 and syn_idx > 0:
                await asyncio.sleep(0)  # Yield to event loop

            # Skip synapses connected to pinned (KB) neurons
            if synapse.source_id in pinned_neuron_ids or synapse.target_id in pinned_neuron_ids:
                continue

            # Protect causal synapses — they encode "why" and must survive decay.
            # Inferred ones can still be pruned (may be noisy), but manually-created
            # causal links are never removed even if weak.
            if synapse.type in _CAUSAL_SYNAPSE_TYPES and not synapse.metadata.get(
                "_inferred", False
            ):
                continue

            # Apply time-based decay before checking weight threshold
            decayed = synapse.time_decay(reference_time=reference_time)

            # Inferred synapses with low reinforcement decay 2x faster
            is_inferred = synapse.metadata.get("_inferred", False)
            if is_inferred and synapse.reinforced_count < 2:
                decayed = decayed.decay(factor=0.5)

            # Dream synapses decay Nx faster (default 10x)
            is_dream = synapse.metadata.get("_dream", False)
            if is_dream and synapse.reinforced_count < 2:
                dream_factor = 1.0 / self._dream_decay_multiplier
                decayed = decayed.decay(factor=dream_factor)

            # Semantic discovery synapses decay 2x faster unless reinforced
            is_semantic = synapse.metadata.get("_semantic_discovery", False)
            if is_semantic and synapse.reinforced_count < 2:
                decayed = decayed.decay(factor=0.5)

            # Weak associative synapses (CO_OCCURS, ALIAS) decay 3x faster
            # unless reinforced — prevents noise from dominating the graph
            # Skip if already decayed above (inferred/dream/semantic) to avoid stacking
            # Skip dedup ALIAS synapses — they are structural, not associative
            is_dedup = synapse.metadata.get("_dedup", False)
            if (
                not is_inferred
                and not is_dream
                and not is_semantic
                and not is_dedup
                and synapse.type in (SynapseType.CO_OCCURS, SynapseType.ALIAS)
                and synapse.reinforced_count < 3
            ):
                decayed = decayed.decay(factor=0.33)

            # Semantic-stage neurons use reduced threshold (harder to prune)
            effective_prune_threshold = self._config.prune_weight_threshold
            if synapse.source_id in semantic_neuron_ids or synapse.target_id in semantic_neuron_ids:
                effective_prune_threshold *= self._config.prune_semantic_factor

            should_prune = decayed.weight < effective_prune_threshold

            # Check inactivity
            if synapse.last_activated is not None:
                days_inactive = (reference_time - synapse.last_activated).total_seconds() / 86400
                should_prune = (
                    should_prune and days_inactive >= self._config.prune_min_inactive_days
                )
            elif synapse.created_at is not None:
                days_since_creation = (reference_time - synapse.created_at).total_seconds() / 86400
                # Never-activated synapses use a shorter grace period
                grace_period = max(1.0, self._config.prune_min_inactive_days / 7)
                should_prune = should_prune and days_since_creation >= grace_period

            if should_prune:
                # High-salience fibers resist pruning
                source_fibers = fiber_salience_cache.get(synapse.source_id, [])
                for fib in source_fibers:
                    if fib.salience > 0.8:
                        should_prune = False
                        break

            if should_prune:
                # Protect bridge synapses (only connection between source and target)
                if synapse.weight >= bridge_floor:
                    out_synapses = neighbor_synapses_map.get(synapse.source_id, [])
                    neighbor_ids = {s.target_id for s in out_synapses}
                    if synapse.target_id in neighbor_ids and len(neighbor_ids) <= 1:
                        continue  # Bridge synapse — don't prune

                pruned_synapse_ids.add(synapse.id)
                report.synapses_pruned += 1

        # Batch delete all pruned synapses at once
        if pruned_synapse_ids and not dry_run:
            if hasattr(self._storage, "delete_synapses_batch"):
                await self._storage.delete_synapses_batch(pruned_synapse_ids)
            else:
                for sid in pruned_synapse_ids:
                    await self._storage.delete_synapse(sid)

        # Update fiber synapse_ids to remove pruned refs (only if synapses were pruned)
        fibers = fibers_for_salience
        if pruned_synapse_ids:
            # Build inverted index: synapse_id -> fiber indices (only for pruned IDs)
            synapse_to_fiber_idx: dict[str, list[int]] = {}
            for idx, fiber in enumerate(fibers):
                for sid in fiber.synapse_ids & pruned_synapse_ids:
                    synapse_to_fiber_idx.setdefault(sid, []).append(idx)

            # Only update fibers that reference pruned synapses
            affected_indices: set[int] = set()
            for indices in synapse_to_fiber_idx.values():
                affected_indices.update(indices)

            for idx in affected_indices:
                if not dry_run:
                    fiber = fibers[idx]
                    updated_fiber = dc_replace(
                        fiber,
                        synapse_ids=fiber.synapse_ids - pruned_synapse_ids,
                    )
                    await self._storage.update_fiber(updated_fiber)

        # Find orphan neurons (no synapses AND not in any fiber)
        if not self._config.prune_isolated_neurons:
            return

        # Derive remaining synapses from cached list instead of re-fetching
        connected_neuron_ids: set[str] = set()
        for syn in all_synapses:
            if syn.id not in pruned_synapse_ids:
                connected_neuron_ids.add(syn.source_id)
                connected_neuron_ids.add(syn.target_id)

        # Protect ALL neurons in fibers, not just anchors
        fiber_neuron_ids: set[str] = set()
        for fiber in fibers:
            fiber_neuron_ids.update(fiber.neuron_ids)

        # Dead neuron pruning: never-accessed + old enough + not pinned
        dead_neuron_days = getattr(self._config, "prune_dead_neuron_days", 14.0)

        # Paginate through all neurons to avoid OOM (batch 5k)
        batch_size = 5000
        offset = 0
        orphan_ids: list[str] = []
        dead_ids: list[str] = []
        while True:
            batch = await self._storage.find_neurons(
                limit=batch_size, offset=offset, ephemeral=False
            )
            if not batch:
                break

            # Check for dead neurons (never accessed, old enough)
            batch_ids = [n.id for n in batch]
            states = await self._storage.get_neuron_states_batch(batch_ids)

            for neuron in batch:
                is_orphan = (
                    neuron.id not in connected_neuron_ids and neuron.id not in fiber_neuron_ids
                )
                if is_orphan:
                    report.neurons_pruned += 1
                    orphan_ids.append(neuron.id)
                    continue

                # Dead neuron: has connections but never accessed, old enough, not pinned
                if neuron.id in pinned_neuron_ids:
                    continue
                state = states.get(neuron.id)
                freq = state.access_frequency if state else 0
                if freq > 0:
                    continue
                age_days = (reference_time - neuron.created_at).total_seconds() / 86400
                if age_days >= dead_neuron_days:
                    report.neurons_pruned += 1
                    dead_ids.append(neuron.id)

            offset += len(batch)
            if len(batch) < batch_size:
                break

        all_prune_ids = orphan_ids + dead_ids
        if dead_ids:
            logger.info(
                "Dead neuron prune: %d orphans + %d dead (never accessed, >%gd old)",
                len(orphan_ids),
                len(dead_ids),
                dead_neuron_days,
            )

        if not dry_run and all_prune_ids:
            # Use batch delete if available, else fall back to individual deletes
            if hasattr(self._storage, "delete_neurons_batch"):
                await self._storage.delete_neurons_batch(all_prune_ids)
            else:
                for nid in all_prune_ids:
                    await self._storage.delete_neuron(nid)

        # Prune old unpromoted entity refs (lazy entity promotion cleanup)
        if not dry_run and hasattr(self._storage, "prune_old_entity_refs"):
            prune_days = getattr(self._config, "lazy_entity_prune_days", 90)
            try:
                pruned_refs = await self._storage.prune_old_entity_refs(prune_days)
                if pruned_refs > 0:
                    logger.info("Pruned %d old unpromoted entity refs", pruned_refs)
            except Exception:
                logger.debug("Entity ref pruning skipped (table may not exist)")

    async def _merge(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Merge overlapping fibers using inverted index for O(n*m) performance.

        Instead of O(n²) pairwise comparison, builds a neuron→fiber inverted
        index to find only fibers that actually share neurons.
        """
        fibers = await self._storage.get_fibers(limit=10000)
        if len(fibers) < 2:
            return

        fiber_list = list(fibers)
        n = len(fiber_list)

        # Build inverted index: neuron_id → set of fiber indices
        neuron_to_fibers: dict[str, set[int]] = {}
        for idx, fiber in enumerate(fiber_list):
            if len(fiber.neuron_ids) > self._config.merge_max_fiber_size:
                continue
            for nid in fiber.neuron_ids:
                neuron_to_fibers.setdefault(nid, set()).add(idx)

        # Find candidate pairs (fibers sharing at least one neuron)
        candidate_pairs: set[tuple[int, int]] = set()
        max_candidate_pairs = 50_000
        for indices in neuron_to_fibers.values():
            # Skip overly-shared neurons (e.g. entity appearing in 500+ fibers)
            if len(indices) > 100:
                continue
            indices_list = sorted(indices)
            for i_pos in range(len(indices_list)):
                for j_pos in range(i_pos + 1, len(indices_list)):
                    candidate_pairs.add((indices_list[i_pos], indices_list[j_pos]))
            if len(candidate_pairs) >= max_candidate_pairs:
                break

        # Union-Find clustering
        uf = UnionFind(n)

        # Only compute Jaccard for actual candidate pairs
        pairs_checked = 0
        for i, j in candidate_pairs:
            pairs_checked += 1
            if pairs_checked % 1000 == 0:
                await asyncio.sleep(0)  # yield so timeout can fire
            # Domain guard: never merge structured/verbatim fibers with non-structured
            fi_verbatim = fiber_list[i].metadata.get("_verbatim", False)
            fj_verbatim = fiber_list[j].metadata.get("_verbatim", False)
            if fi_verbatim != fj_verbatim:
                continue

            set_a = fiber_list[i].neuron_ids
            set_b = fiber_list[j].neuron_ids
            intersection = len(set_a & set_b)
            union_size = len(set_a | set_b)

            if union_size > 0:
                jaccard = intersection / union_size
                # Graduated temporal proximity: closer fibers need less overlap
                if fiber_list[i].created_at and fiber_list[j].created_at:
                    time_diff = abs(
                        (fiber_list[i].created_at - fiber_list[j].created_at).total_seconds()
                    )
                else:
                    time_diff = float("inf")
                halflife = self._config.merge_temporal_halflife_seconds
                # Smooth curve: 60% threshold at t=0, rising to 100% as t→∞
                temporal_factor = 1.0 - 0.4 * math.exp(-time_diff / halflife)
                effective_threshold = self._config.merge_overlap_threshold * temporal_factor
                if jaccard >= effective_threshold:
                    uf.union(i, j)

        # --- T4.1: SimHash semantic merge pass ---
        # Fetch anchor neuron content_hash for content-based similarity
        simhash_roots: set[int] = set()  # C1 fix: track which roots got SimHash unions
        anchor_neurons: dict[str, Any] = {}
        anchor_ids_unique = list(
            {
                fiber_list[i].anchor_neuron_id
                for i in range(n)
                if len(fiber_list[i].neuron_ids) <= self._config.merge_max_fiber_size
            }
        )
        if anchor_ids_unique:
            anchor_neurons = await self._storage.get_neurons_batch(anchor_ids_unique)

            # Build fiber index → content_hash map
            fiber_content_hash: dict[int, int] = {}
            for idx in range(n):
                if len(fiber_list[idx].neuron_ids) > self._config.merge_max_fiber_size:
                    continue
                anchor = anchor_neurons.get(fiber_list[idx].anchor_neuron_id)
                if anchor and anchor.content_hash and anchor.content_hash != 0:
                    fiber_content_hash[idx] = anchor.content_hash

            if not fiber_content_hash:
                logger.debug("SimHash pass: no anchor content hashes available")

            # Pairwise SimHash comparison (capped to avoid O(n²) blowup)
            hash_indices = sorted(fiber_content_hash.keys())
            max_simhash_checks = 50_000
            simhash_checked = 0
            for i_pos in range(len(hash_indices)):
                if simhash_checked >= max_simhash_checks:
                    break
                idx_a = hash_indices[i_pos]
                for j_pos in range(i_pos + 1, len(hash_indices)):
                    if simhash_checked >= max_simhash_checks:
                        break
                    idx_b = hash_indices[j_pos]
                    simhash_checked += 1
                    if simhash_checked % 1000 == 0:
                        await asyncio.sleep(0)
                    # Skip if already in same group (Jaccard already merged)
                    if uf.find(idx_a) == uf.find(idx_b):
                        continue
                    # Domain guard: never merge verbatim with non-verbatim
                    fi_verbatim = fiber_list[idx_a].metadata.get("_verbatim", False)
                    fj_verbatim = fiber_list[idx_b].metadata.get("_verbatim", False)
                    if fi_verbatim != fj_verbatim:
                        continue
                    if _simhash_near_dup(fiber_content_hash[idx_a], fiber_content_hash[idx_b]):
                        uf.union(idx_a, idx_b)
                        simhash_roots.add(uf.find(idx_a))

        # Group fibers by root
        groups = uf.groups()

        # Merge groups with more than 1 member
        for members in groups.values():
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
            # Merge auto_tags and agent_tags separately
            merged_auto_tags: set[str] = set()
            merged_agent_tags: set[str] = set()
            for fiber in member_fibers:
                merged_auto_tags |= fiber.auto_tags
                merged_agent_tags |= fiber.agent_tags

            # T4.4: Summary fiber for large groups (5+ members)
            is_summary = len(member_fibers) >= 5
            merge_metadata: dict[str, Any] = {
                "merged_from": [f.id for f in member_fibers],
            }
            if is_summary:
                merge_metadata["_stage"] = "semantic"
                merge_metadata["_summary_fiber"] = True
                # Preserve original summaries for context recovery
                original_summaries = [f.summary for f in member_fibers if f.summary][:10]
                if original_summaries:
                    merge_metadata["_original_summaries"] = original_summaries

            # T3: Dynamic abstraction — induce a CONCEPT neuron for large clusters.
            # Kept as a parallel structure: cluster neurons are NOT deleted (CLS).
            abstract_neuron: Neuron | None = None
            abstract_links: list[Synapse] = []
            if (
                is_summary
                and self._config.enable_dynamic_abstraction
                and len(member_fibers) >= self._config.abstraction_cluster_min_size
            ):
                cluster_neurons = [
                    anchor_neurons[f.anchor_neuron_id]
                    for f in member_fibers
                    if f.anchor_neuron_id in anchor_neurons
                ]
                if len(cluster_neurons) >= 2:
                    try:
                        abstract_neuron = induce_abstraction(cluster_neurons)
                        merge_metadata["_abstract_neuron_id"] = abstract_neuron.id
                        # IS_A link from each exemplar → abstract so traversal can reach it.
                        for exemplar in cluster_neurons:
                            abstract_links.append(
                                Synapse.create(
                                    source_id=exemplar.id,
                                    target_id=abstract_neuron.id,
                                    type=SynapseType.IS_A,
                                    weight=0.6,
                                    metadata={"_abstraction_induced": True},
                                )
                            )
                    except Exception as exc:
                        logger.warning("Skipping abstraction induction: %s", exc)
                        abstract_neuron = None
                        abstract_links = []

            # Build summary from anchor neuron content (not just "Merged from N")
            anchor_ids_for_summary = list(dict.fromkeys(f.anchor_neuron_id for f in member_fibers))[
                :3
            ]
            if anchor_ids_for_summary and anchor_neurons:
                snippets = []
                for aid in anchor_ids_for_summary:
                    a_neuron = anchor_neurons.get(aid)
                    if a_neuron and a_neuron.content:
                        snippets.append(a_neuron.content[:80].strip())
                merge_summary = (
                    " | ".join(snippets) if snippets else f"Merged from {len(member_fibers)} fibers"
                )
            else:
                merge_summary = f"Merged from {len(member_fibers)} fibers"

            merged_fiber = Fiber(
                id=merged_fiber_id,
                neuron_ids=merged_neuron_ids,
                synapse_ids=merged_synapse_ids,
                anchor_neuron_id=best_anchor,
                pathway=[best_anchor],
                salience=max_salience,
                frequency=best_frequency,
                auto_tags=merged_auto_tags,
                agent_tags=merged_agent_tags,
                summary=merge_summary,
                metadata=merge_metadata,
                created_at=min(f.created_at for f in member_fibers),
            )

            # C1 fix: attribute merge reason based on actual SimHash participation
            group_root = uf.find(members[0])
            has_simhash = group_root in simhash_roots
            merge_reason = "simhash_content" if has_simhash else "neuron_overlap"
            report.fibers_merged += len(member_fibers)
            report.fibers_created += 1
            report.merge_details.append(
                MergeDetail(
                    original_fiber_ids=tuple(f.id for f in member_fibers),
                    merged_fiber_id=merged_fiber_id,
                    neuron_count=len(merged_neuron_ids),
                    reason=merge_reason,
                )
            )

            if not dry_run:
                # H2 fix: add merged fiber FIRST, then modify originals
                try:
                    await self._storage.add_fiber(merged_fiber)
                except ValueError as exc:
                    # FK failure — referenced neurons may have been pruned or
                    # are mid-transaction in another process.  Skip this merge.
                    logger.warning("Skipping fiber merge %s: %s", merged_fiber.id, exc)
                    continue

                # T3: Persist abstract neuron + IS_A links (best-effort, non-fatal).
                if abstract_neuron is not None:
                    try:
                        await self._storage.add_neuron(abstract_neuron)
                        for link in abstract_links:
                            try:
                                await self._storage.add_synapse(link)
                            except Exception as link_exc:
                                logger.debug("Abstraction IS_A link skipped: %s", link_exc)
                        report.concepts_created += 1
                    except Exception as neuron_exc:
                        logger.warning("Abstraction neuron persistence skipped: %s", neuron_exc)

                if is_summary:
                    # T4.4: Demote originals to COLD instead of deleting
                    for fiber in member_fibers:
                        demoted_meta = {**fiber.metadata, "_demoted_by_merge": True}
                        demoted = dc_replace(fiber, metadata=demoted_meta)
                        await self._storage.update_fiber(demoted)
                else:
                    # Standard merge: delete originals
                    for fiber in member_fibers:
                        await self._storage.delete_fiber(fiber.id)
                        report.fibers_removed += 1

    async def _summarize(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Create concept neurons for tag-based clusters using inverted index."""
        fibers = await self._storage.get_fibers(limit=10000)
        if len(fibers) < self._config.summarize_min_cluster_size:
            return

        fiber_list = [f for f in fibers if f.tags]

        # Cap fiber count for O(N²) pair comparison — keep highest-salience
        max_fibers_for_clustering = 1000
        if len(fiber_list) > max_fibers_for_clustering:
            fiber_list = sorted(fiber_list, key=lambda f: f.salience, reverse=True)[
                :max_fibers_for_clustering
            ]
        if len(fiber_list) < self._config.summarize_min_cluster_size:
            return

        n = len(fiber_list)

        # Build inverted index: tag → set of fiber indices
        tag_to_fibers: dict[str, set[int]] = {}
        for idx, fiber in enumerate(fiber_list):
            for tag in fiber.tags:
                tag_to_fibers.setdefault(tag, set()).add(idx)

        # Find candidate pairs (fibers sharing at least one tag)
        # Skip overly common tags (>100 fibers) to avoid O(N²) explosion
        max_pairs = 50_000
        candidate_pairs: set[tuple[int, int]] = set()
        for indices in tag_to_fibers.values():
            if len(indices) > 100:
                continue  # Tag too common — skip to avoid combinatorial blowup
            indices_list = sorted(indices)
            for i_pos in range(len(indices_list)):
                if len(candidate_pairs) >= max_pairs:
                    break
                for j_pos in range(i_pos + 1, len(indices_list)):
                    if len(candidate_pairs) >= max_pairs:
                        break
                    candidate_pairs.add((indices_list[i_pos], indices_list[j_pos]))
            if len(candidate_pairs) >= max_pairs:
                break

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

        for pair_idx, (i, j) in enumerate(candidate_pairs):
            if pair_idx % 1000 == 0 and pair_idx > 0:
                await asyncio.sleep(0)  # Yield to event loop
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

        for members in groups.values():
            if len(members) < self._config.summarize_min_cluster_size:
                continue

            cluster_fibers = [fiber_list[i] for i in members]

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

            anchor_ids: set[str] = set()
            for fiber in cluster_fibers:
                anchor_ids.add(fiber.anchor_neuron_id)

            # Filter out anchor neurons that were pruned in earlier tiers
            valid_anchor_ids: set[str] = set()
            for aid in anchor_ids:
                anchor_neuron = await self._storage.get_neuron(aid)
                if anchor_neuron is not None:
                    valid_anchor_ids.add(aid)
            anchor_ids = valid_anchor_ids

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
            try:
                await self._storage.add_fiber(summary_fiber)
            except ValueError as exc:
                logger.warning("Skipping summary fiber %s: %s", summary_fiber.id, exc)
                continue
            report.summaries_created += 1

    async def _mature(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Advance memory maturation stages, auto-promote types, extract patterns.

        0. Auto-promote frequently-recalled context memories to fact
        1. Advance all maturation records through stage transitions
        2. Extract patterns from episodic memories ready for semantic promotion
        """
        import logging

        from neural_memory.core.memory_types import MemoryType
        from neural_memory.engine.memory_stages import (
            compute_stage_transition,
        )
        from neural_memory.engine.pattern_extraction import extract_patterns

        _logger = logging.getLogger(__name__)

        # Phase 0: Auto-promote context→fact for frequently-recalled memories
        # Must run before prune to prevent promotion candidates from expiring.
        # Graduated: frequency >= 5 triggers promotion to fact (no expiry).
        if not dry_run:
            try:
                candidates = await self._storage.get_promotion_candidates(
                    min_frequency=5,
                    source_type="context",
                )
                for candidate in candidates:
                    fiber_id = candidate["fiber_id"]
                    meta = candidate.get("metadata", {})
                    # Skip already-promoted memories
                    if meta.get("auto_promoted"):
                        continue
                    promoted = await self._storage.promote_memory_type(
                        fiber_id=fiber_id,
                        new_type=MemoryType.FACT,
                        new_expires_at=None,  # Facts don't expire
                    )
                    if promoted:
                        report.memories_promoted += 1
                if report.memories_promoted > 0:
                    _logger.info(
                        "Auto-promoted %d context memories to fact",
                        report.memories_promoted,
                    )
            except Exception:
                _logger.warning("Auto-promote failed (non-critical)", exc_info=True)

        # Clean up orphaned maturation records (fibers deleted without CASCADE)
        cleaned = await self._storage.cleanup_orphaned_maturations()
        if cleaned > 0:
            _logger.info("Cleaned up %d orphaned maturation records", cleaned)

        # Get all maturation records
        all_maturations = await self._storage.find_maturations()

        # Phase 1: Advance stages (with fast-track for high-recall memories)
        ft_rehearsals = self._config.maturation_fast_track_rehearsals
        ft_time = self._config.maturation_fast_track_time_days
        for record in all_maturations:
            advanced = compute_stage_transition(
                record,
                now=reference_time,
                fast_track_rehearsals=ft_rehearsals,
                fast_track_time_days=ft_time,
            )
            if advanced.stage != record.stage:
                report.stages_advanced += 1
                if not dry_run:
                    try:
                        await self._storage.save_maturation(advanced)
                    except Exception as exc:
                        if "FOREIGN KEY" in str(exc):
                            _logger.warning(
                                "Skipping orphaned maturation for fiber %s",
                                record.fiber_id,
                            )
                            continue
                        raise

        # Phase 2: Extract patterns from mature episodic fibers
        if dry_run:
            return

        # Re-fetch after stage updates
        maturations = await self._storage.find_maturations()
        maturation_map = {m.fiber_id: m for m in maturations}

        fibers = await self._storage.get_fibers(limit=10000)
        patterns, extraction_report = extract_patterns(
            fibers=fibers,
            maturations=maturation_map,
            min_cluster_size=self._config.summarize_min_cluster_size,
            tag_overlap_threshold=self._config.summarize_tag_overlap_threshold,
        )

        report.patterns_extracted = extraction_report.patterns_extracted

        for pattern in patterns:
            await self._storage.add_neuron(pattern.concept_neuron)
            for synapse in pattern.synapses:
                await self._storage.add_synapse(synapse)

        # Phase 3: Generate essence for fibers that have content but no essence
        await self._essence_backfill(report, dry_run)

    async def _essence_backfill(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Generate essence for fibers missing it, or upgrade extractive → LLM.

        Uses configured essence_generator strategy from BrainConfig:
        - "extractive" (default): sentence-level scoring, fast and free
        - "llm": LLM abstractive with cost guard (priority < 3 skipped)

        Paginates in batches of 500 to avoid the storage cap.
        """
        from neural_memory.engine.fidelity import get_essence_generator

        # Resolve generator strategy from brain config
        strategy = "extractive"
        try:
            brain_id = self._storage._get_brain_id()
            brain = await self._storage.get_brain(brain_id)
            if brain and brain.config:
                strategy = getattr(brain.config, "essence_generator", "extractive")
        except Exception:
            logger.debug("Could not read brain config for essence strategy", exc_info=True)

        generator = get_essence_generator(strategy)

        max_backfill = 2000  # Safety cap to avoid runaway

        # Fetch fibers in one batch (get_fibers has no offset param; limit=1000 is storage cap)
        fibers = await self._storage.get_fibers(limit=1000)
        candidates = [f for f in fibers if not f.essence]

        backfilled = 0
        for idx, fiber in enumerate(candidates):
            if backfilled >= max_backfill:
                break
            if idx % 50 == 0 and idx > 0:
                await asyncio.sleep(0)  # Yield to event loop

            anchor = await self._storage.get_neuron(fiber.anchor_neuron_id)
            if not anchor or not anchor.content:
                continue

            # Get priority from typed memory for cost guard
            priority = 5
            try:
                typed_mem = await self._storage.get_typed_memory(fiber.id)
                if (
                    typed_mem
                    and hasattr(typed_mem, "priority")
                    and isinstance(typed_mem.priority, (int, float))
                ):
                    priority = int(typed_mem.priority)
            except Exception:
                pass

            essence = await generator.generate(anchor.content, priority=priority)
            if not essence:
                continue

            if dry_run:
                backfilled += 1
                continue

            updated = fiber.with_essence(essence)
            await self._storage.update_fiber(updated)
            backfilled += 1

        if backfilled > 0:
            logger.info("Essence backfill: %d fibers updated", backfilled)
        report.essences_generated += backfilled

    async def _infer(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Run associative inference from co-activation data.

        1. Query co-activation counts within the time window
        2. Identify new + reinforcement candidates
        3. Create CO_OCCURS synapses for new candidates
        4. Reinforce existing synapses for reinforce candidates
        5. Generate + apply associative tags
        6. Prune old co-activation events
        """
        import logging

        from neural_memory.engine.associative_inference import (
            InferenceConfig,
            create_inferred_synapse,
            generate_associative_tags,
            identify_candidates,
        )
        from neural_memory.utils.tag_normalizer import TagNormalizer

        logger = logging.getLogger(__name__)

        config = InferenceConfig(
            co_activation_threshold=self._config.infer_co_activation_threshold,
            co_activation_window_days=self._config.infer_window_days,
            max_inferences_per_run=self._config.infer_max_per_run,
        )

        # 1. Query co-activation counts within time window
        from datetime import timedelta

        window_start = reference_time - timedelta(days=config.co_activation_window_days)
        counts = await self._storage.get_co_activation_counts(
            since=window_start,
            min_count=config.co_activation_threshold,
        )

        if not counts:
            return

        # 2. Build existing synapse pairs set + lookup for reinforcement
        # Need all types: existing_pairs prevents duplicate creation, synapse_by_pair enables reinforcement
        all_synapses = await self._storage.get_synapses()
        existing_pairs: set[tuple[str, str]] = set()
        synapse_by_pair: dict[tuple[str, str], Synapse] = {}
        for syn in all_synapses:
            existing_pairs.add((syn.source_id, syn.target_id))
            existing_pairs.add((syn.target_id, syn.source_id))
            synapse_by_pair[(syn.source_id, syn.target_id)] = syn

        new_candidates, reinforce_candidates = identify_candidates(counts, existing_pairs, config)

        if dry_run:
            report.synapses_inferred = len(new_candidates) + len(reinforce_candidates)
            return

        # 3. Create CO_OCCURS synapses for new candidates
        for candidate in new_candidates:
            synapse = create_inferred_synapse(candidate)
            try:
                await self._storage.add_synapse(synapse)
                report.synapses_inferred += 1
            except ValueError:
                logger.debug("Inferred synapse already exists, skipping")

        # 4. Reinforce existing synapses for reinforce candidates
        #    Use cached synapse_by_pair lookup instead of N+1 queries
        for candidate in reinforce_candidates:
            a, b = candidate.neuron_a, candidate.neuron_b
            existing_synapse = synapse_by_pair.get((a, b)) or synapse_by_pair.get((b, a))

            if existing_synapse:
                reinforced = existing_synapse.reinforce(delta=0.05)
                try:
                    await self._storage.update_synapse(reinforced)
                    report.synapses_inferred += 1
                except ValueError:
                    logger.debug("Synapse reinforcement failed")

        # 5. Generate and apply associative tags
        all_candidates = new_candidates + reinforce_candidates
        if all_candidates:
            neuron_ids = set()
            for c in all_candidates:
                neuron_ids.add(c.neuron_a)
                neuron_ids.add(c.neuron_b)

            neurons = await self._storage.get_neurons_batch(list(neuron_ids))
            content_map = {nid: n.content for nid, n in neurons.items()}

            fibers = await self._storage.get_fibers(limit=10000)
            existing_tags: set[str] = set()
            for f in fibers:
                existing_tags |= f.tags

            assoc_tags = generate_associative_tags(all_candidates, content_map, existing_tags)

            normalizer = TagNormalizer()

            # Build inverted index: neuron_id -> fiber indices
            neuron_to_fiber_idx: dict[str, set[int]] = {}
            for idx, fiber in enumerate(fibers):
                for nid in fiber.neuron_ids:
                    neuron_to_fiber_idx.setdefault(nid, set()).add(idx)

            # Accumulate all new tags per fiber, then write once
            fiber_new_tags: dict[int, set[str]] = {}
            for atag in assoc_tags:
                normalized_tag = normalizer.normalize(atag.tag)
                # Find affected fibers via inverted index
                affected: set[int] = set()
                for nid in atag.source_neuron_ids:
                    if nid in neuron_to_fiber_idx:
                        affected |= neuron_to_fiber_idx[nid]
                for idx in affected:
                    fiber_new_tags.setdefault(idx, set()).add(normalized_tag)

            # Write accumulated tags in a single pass
            for idx, new_tags in fiber_new_tags.items():
                fiber = fibers[idx]
                updated_auto_tags = fiber.auto_tags | new_tags
                if updated_auto_tags != fiber.auto_tags:
                    updated_fiber = dc_replace(fiber, auto_tags=updated_auto_tags)
                    try:
                        await self._storage.update_fiber(updated_fiber)
                    except Exception:
                        logger.debug("Associative tag update failed", exc_info=True)

            # Log drift detection
            drift_reports = normalizer.detect_drift(existing_tags)
            for dr in drift_reports:
                logger.info("Tag drift detected: %s → %s", dr.variants, dr.canonical)

        # 6. Prune old co-activation events
        pruned = await self._storage.prune_co_activations(older_than=window_start)
        report.co_activations_pruned = pruned

    async def _enrich(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run enrichment: transitive closure + cross-cluster linking."""
        import logging

        from neural_memory.engine.enrichment import enrich

        logger = logging.getLogger(__name__)

        result = await enrich(self._storage)

        all_synapses = result.transitive_synapses + result.cross_cluster_synapses
        if dry_run:
            report.synapses_enriched = len(all_synapses)
            return

        for synapse in all_synapses:
            try:
                await self._storage.add_synapse(synapse)
                report.synapses_enriched += 1
            except ValueError:
                logger.debug("Enriched synapse already exists, skipping")

        # Reactivate dormant neurons (access_frequency=0) to prevent permanent dormancy
        await self._reactivate_dormant(report, dry_run)

    async def _reactivate_dormant(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Bump dormant neurons with minimal activation to simulate memory replay."""
        from dataclasses import replace as dc_replace

        try:
            # Fetch all states — TODO: add dedicated dormant query to storage interface
            all_states = await self._storage.get_all_neuron_states()
        except Exception:
            logging.getLogger(__name__).debug(
                "Failed to get neuron states for dream cycle", exc_info=True
            )
            return

        dormant = [s for s in all_states if s.access_frequency == 0]
        if not dormant:
            return

        # Sample up to 20 dormant neurons (randomize to avoid always picking the same)
        import random

        sample = random.sample(dormant, min(20, len(dormant)))
        if dry_run:
            report.neurons_reactivated = len(sample)
            return

        now = utcnow()
        for state in sample:
            reactivated = dc_replace(
                state,
                activation_level=min(state.activation_level + 0.05, 1.0),
                access_frequency=1,
                last_activated=now,
            )
            await self._storage.update_neuron_state(reactivated)
            report.neurons_reactivated += 1

    async def _dream(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run dream exploration for hidden connections.

        Also scans the dream output for hub neurons (appearing in 3+ new
        synapses) and creates SEMANTIC_LINK-style RELATED_TO synapses between
        the top hubs — these represent concepts that co-emerged during
        exploration without an existing explicit link.
        """
        import logging
        from collections import Counter as _Counter

        from neural_memory.engine.dream import dream

        logger = logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        result = await dream(self._storage, brain.config)

        if dry_run:
            report.dream_synapses_created = len(result.synapses_created)
            return

        for synapse in result.synapses_created:
            try:
                await self._storage.add_synapse(synapse)
                report.dream_synapses_created += 1
            except ValueError:
                logger.debug("Dream synapse already exists, skipping")

        # T4: Hub-entity pattern extraction. A neuron appearing in 3+ new
        # dream synapses is a recurring topic — link the top two hubs
        # semantically so future recall can bridge them.
        if len(result.synapses_created) >= 3:
            participation: _Counter[str] = _Counter()
            for synapse in result.synapses_created:
                participation[synapse.source_id] += 1
                participation[synapse.target_id] += 1
            hubs = [nid for nid, count in participation.most_common(3) if count >= 3]
            if len(hubs) >= 2:
                hub_link = Synapse.create(
                    source_id=hubs[0],
                    target_id=hubs[1],
                    type=SynapseType.RELATED_TO,
                    weight=0.4,
                    metadata={"_dream": True, "_semantic_discovery": True, "_hub": True},
                )
                try:
                    await self._storage.add_synapse(hub_link)
                    report.patterns_extracted += 1
                except (ValueError, Exception) as exc:
                    logger.debug("Dream hub link skipped: %s", exc)

    async def _replay(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run hippocampal replay — LTP/LTD on recent fibers.

        After delegating to the replay engine, counts how many strengthened
        synapses touch abstraction-level concept neurons. These are the
        CLS semantic targets — tracking them lets us know when episodic
        replay is reinforcing existing abstractions (Hebbian on semantics).
        """
        from neural_memory.engine.hippocampal_replay import hippocampal_replay

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        result = await hippocampal_replay(
            self._storage,
            brain.config,
            dry_run=dry_run,
        )
        report.extra["replay_episodes"] = result.episodes_replayed
        report.extra["replay_ltp"] = result.synapses_strengthened
        report.extra["replay_ltd"] = result.synapses_weakened

        # T4: Semantic reinforcement signal — if replay strengthened links,
        # count how many concept-level neurons exist in the brain so users
        # know the abstraction surface. Cheap read, no writes.
        if not dry_run and result.synapses_strengthened > 0:
            try:
                concept_neurons = await self._storage.find_neurons(
                    type=NeuronType.CONCEPT,
                    limit=500,
                )
                induced = sum(1 for n in concept_neurons if n.metadata.get("_abstraction_induced"))
                report.extra["replay_semantic_neurons"] = len(concept_neurons)
                report.extra["replay_induced_concepts"] = induced
            except Exception as exc:
                logger.debug("Replay semantic census skipped: %s", exc)

    async def _schema(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run schema assimilation — create/update schemas from tag clusters."""
        from neural_memory.engine.schema_assimilation import batch_schema_assimilation

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        schemas_created = await batch_schema_assimilation(
            self._storage,
            brain.config,
            dry_run=dry_run,
        )
        report.extra["schemas_created"] = schemas_created

    async def _interference(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run interference scan — detect fan effects across tag clusters."""
        from neural_memory.engine.interference import batch_interference_scan

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        result = await batch_interference_scan(
            self._storage,
            brain.config,
            dry_run=dry_run,
        )
        report.extra["interference_fan_effects"] = result.fan_effects_flagged

    async def _learn_habits(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Learn habits from action event sequences."""
        import logging

        from neural_memory.engine.sequence_mining import learn_habits

        logger = logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        if dry_run:
            return

        try:
            learned, habit_report = await learn_habits(self._storage, brain.config, reference_time)
            report.habits_learned = habit_report.habits_learned
            report.action_events_pruned = habit_report.action_events_pruned
        except Exception:
            logger.debug("Habit learning failed (non-critical)", exc_info=True)

        # Also learn query topic patterns (same substrate, different metadata)
        try:
            from neural_memory.engine.query_pattern_mining import learn_query_patterns

            qp_report = await learn_query_patterns(self._storage, brain.config, reference_time)
            report.habits_learned += qp_report.patterns_learned
        except Exception:
            logger.debug("Query pattern learning failed (non-critical)", exc_info=True)

    async def _dedup(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Deduplicate anchor neurons using SimHash comparison.

        Scans all anchor neurons and finds near-duplicates by Hamming distance.
        Creates ALIAS synapses and redirects fibers to canonical anchors.
        """
        import logging

        from neural_memory.core.synapse import SynapseType
        from neural_memory.utils.simhash import is_near_duplicate

        logger = logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return

        # Paginate through all neurons to collect anchors (avoid OOM)
        batch_size = 5000
        offset = 0
        anchors: list[Neuron] = []
        while True:
            batch = await self._storage.find_neurons(
                limit=batch_size, offset=offset, ephemeral=False
            )
            if not batch:
                break
            anchors.extend(n for n in batch if n.metadata.get("is_anchor", False))
            offset += len(batch)
            if len(batch) < batch_size:
                break

        if len(anchors) < 2:
            return

        # Cap anchors to prevent O(N^2) blowup (N=2000 → 2M comparisons)
        max_anchors = 2000
        if len(anchors) > max_anchors:
            anchors = anchors[:max_anchors]

        # Group duplicates by SimHash proximity
        seen: set[str] = set()
        for i, anchor_a in enumerate(anchors):
            if anchor_a.id in seen:
                continue
            # Yield to event loop every 50 outer iterations so timeout can fire
            if i % 50 == 0:
                await asyncio.sleep(0)
            if anchor_a.content_hash is None or anchor_a.content_hash == 0:
                continue

            for anchor_b in anchors[i + 1 :]:
                if anchor_b.id in seen:
                    continue
                if anchor_b.content_hash is None or anchor_b.content_hash == 0:
                    continue

                if is_near_duplicate(anchor_a.content_hash, anchor_b.content_hash):
                    report.duplicates_found += 1
                    seen.add(anchor_b.id)

                    if dry_run:
                        continue

                    # Create ALIAS synapse from newer to older (canonical)
                    alias_synapse = Synapse.create(
                        source_id=anchor_b.id,
                        target_id=anchor_a.id,
                        type=SynapseType.ALIAS,
                        weight=0.9,
                        metadata={"_dedup": True},
                    )
                    try:
                        await self._storage.add_synapse(alias_synapse)
                    except ValueError:
                        logger.debug("ALIAS synapse already exists")

    async def _semantic_link(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Discover and create SIMILAR_TO synapses via embedding similarity.

        Optional — silently skips if embeddings are not available.
        Created synapses decay 2x faster during pruning unless reinforced.
        """
        import logging

        from neural_memory.engine.semantic_discovery import discover_semantic_synapses

        logger = logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        result = await discover_semantic_synapses(self._storage, brain.config)

        if dry_run:
            report.semantic_synapses_created = result.synapses_created
            return

        for synapse in result.synapses:
            try:
                await self._storage.add_synapse(synapse)
                report.semantic_synapses_created += 1
            except ValueError:
                logger.debug("Semantic synapse already exists, skipping")

    async def _compress(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Run tiered memory compression on all eligible fibers.

        Creates a CompressionEngine with default config and runs it for the
        current brain context.  Results are merged into *report*.
        """
        import logging as _logging

        from neural_memory.engine.compression import CompressionEngine

        _logger = _logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            _logger.debug("COMPRESS skipped: no brain context")
            return

        engine = CompressionEngine(self._storage)
        compression_report = await engine.run(
            reference_time=reference_time,
            dry_run=dry_run,
        )

        report.fibers_compressed += compression_report.fibers_compressed
        report.tokens_saved += compression_report.tokens_saved

    async def _lifecycle(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Calculate heat scores and update lifecycle_state for all neurons.

        Fetches all neurons in the current brain, computes heat scores from
        access frequency and recency, then updates the lifecycle_state column.
        Each state maps to a compression tier range:
          ACTIVE → < 7d or hot (heat > threshold)
          WARM   → 7-30d or accessed in last 14d
          COOL   → 30-90d
          COMPRESSED → 90-180d
          ARCHIVED → 180d+

        Args:
            report: ConsolidationReport to update.
            reference_time: UTC reference time for age/recency calculations.
            dry_run: If True, calculate but do not apply changes.
        """
        import logging as _logging

        from neural_memory.engine.compression import (
            CompressionConfig,
            calculate_heat_score,
            determine_lifecycle_state,
        )

        _logger = _logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            _logger.debug("LIFECYCLE skipped: no brain context")
            return

        # Fetch neurons in batches via find_neurons (full scan)
        try:
            neurons = await self._storage.find_neurons(limit=10000, ephemeral=False)
        except Exception:
            _logger.error("LIFECYCLE failed to fetch neurons", exc_info=True)
            return

        config = CompressionConfig()
        states_updated = 0

        for neuron in neurons:
            # Retrieve last_accessed_at and access_frequency from neuron metadata
            # (access_frequency is stored in neuron_states, not neurons directly)
            last_accessed_raw: str | None = neuron.metadata.get("last_accessed_at")
            last_accessed_at: datetime | None = None
            if last_accessed_raw:
                try:
                    last_accessed_at = datetime.fromisoformat(last_accessed_raw)
                except ValueError:
                    pass

            access_count: int = int(neuron.metadata.get("access_frequency", 0))
            priority: int = int(neuron.metadata.get("priority", 5))

            heat = calculate_heat_score(
                last_accessed_at=last_accessed_at,
                access_count=access_count,
                priority=priority,
                reference_time=reference_time,
                config=config,
            )

            age_days = (reference_time - neuron.created_at).total_seconds() / 86400.0
            new_state = determine_lifecycle_state(age_days, heat, config)

            current_state: str = neuron.metadata.get("lifecycle_state", "active")
            if current_state == str(new_state):
                continue

            if not dry_run:
                try:
                    await self._storage.update_neuron_lifecycle(neuron.id, str(new_state))
                    states_updated += 1
                except Exception:
                    _logger.error(
                        "Failed to update lifecycle_state for neuron %s", neuron.id, exc_info=True
                    )
            else:
                states_updated += 1

        if states_updated:
            report.extra["lifecycle_states_updated"] = (
                report.extra.get("lifecycle_states_updated", 0) + states_updated
            )
        _logger.info("LIFECYCLE: updated %d neuron lifecycle states", states_updated)

        # --- T4.2 + T4.3: Fiber-level stale detection and access-based demotion ---
        await self._lifecycle_fiber_pass(report, reference_time, dry_run)

    _VERSION_PATTERN: re.Pattern[str] = re.compile(r"(?<![A-Za-z])[vV](\d+)\.(\d+)(?:\.\d+)?\b")

    async def _lifecycle_fiber_pass(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Fiber-level lifecycle: stale detection (T4.2) + access demotion (T4.3).

        Scans fibers for:
        - Version references ≥2 major versions behind → ``_stale: true``
        - Never-recalled after 30 days → ``_cold_demoted: true``
        - Never-recalled after 90 days + not pinned → ``_prune_candidate: true``
        """
        try:
            fibers = await self._storage.get_fibers(limit=10000)
        except Exception:
            logger.error("LIFECYCLE fiber pass: failed to fetch fibers", exc_info=True)
            return
        if not fibers:
            return

        # --- T4.2: Stale version detection ---
        # Batch-fetch anchor neurons to check content for version patterns
        anchor_ids = list({f.anchor_neuron_id for f in fibers})
        try:
            anchor_neurons = await self._storage.get_neurons_batch(anchor_ids)
        except Exception:
            logger.error("LIFECYCLE fiber pass: failed to fetch anchors", exc_info=True)
            anchor_neurons = {}

        # Find highest major version across all anchor content
        max_major = 0
        for anchor in anchor_neurons.values():
            for m in self._VERSION_PATTERN.finditer(anchor.content):
                major = int(m.group(1))
                if major > max_major:
                    max_major = major

        stale_count = 0
        cold_demoted = 0
        prune_candidates = 0

        for fiber in fibers:
            fiber_meta = fiber.metadata or {}
            age_days = (reference_time - fiber.created_at).total_seconds() / 86400.0

            # T4.2: Flag fibers with version references ≥2 major versions behind
            if max_major >= 2 and not fiber_meta.get("_stale"):
                fiber_anchor = anchor_neurons.get(fiber.anchor_neuron_id)
                if fiber_anchor:
                    versions = self._VERSION_PATTERN.findall(fiber_anchor.content)
                    if versions:
                        fiber_max_major = max(int(v[0]) for v in versions)
                        if max_major - fiber_max_major >= 2:
                            if not dry_run:
                                await self._storage.update_fiber_metadata(
                                    fiber.id, {"_stale": True}
                                )
                            stale_count += 1

            # T4.3: Access-based demotion (M1 fix: batch metadata updates)
            if fiber.pinned:
                continue

            demotion_updates: dict[str, Any] = {}
            if fiber.frequency == 0 and age_days > 30:
                if not fiber_meta.get("_cold_demoted"):
                    demotion_updates["_cold_demoted"] = True
                    cold_demoted += 1

            if fiber.frequency == 0 and age_days > 90:
                if not fiber_meta.get("_prune_candidate"):
                    demotion_updates["_prune_candidate"] = True
                    prune_candidates += 1

            if demotion_updates and not dry_run:
                await self._storage.update_fiber_metadata(fiber.id, demotion_updates)

        if stale_count:
            report.extra["stale_flagged"] = report.extra.get("stale_flagged", 0) + stale_count
        if cold_demoted:
            report.extra["cold_demoted"] = report.extra.get("cold_demoted", 0) + cold_demoted
        if prune_candidates:
            report.extra["prune_candidates"] = (
                report.extra.get("prune_candidates", 0) + prune_candidates
            )
        logger.info(
            "LIFECYCLE fibers: %d stale, %d cold-demoted, %d prune candidates",
            stale_count,
            cold_demoted,
            prune_candidates,
        )

    async def _process_tool_events(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Process buffered tool events into neurons and synapses.

        Reads the JSONL buffer, ingests into tool_events table, then runs
        pattern detection. Only executes if tool_memory.enabled in config.
        """
        import logging as _logging

        from neural_memory.unified_config import UnifiedConfig

        _logger = _logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            _logger.debug("PROCESS_TOOL_EVENTS skipped: no brain context")
            return

        try:
            config = UnifiedConfig.load()
        except Exception:
            _logger.debug("PROCESS_TOOL_EVENTS skipped: config load failed", exc_info=True)
            return

        if not config.tool_memory.enabled:
            return

        if dry_run:
            _logger.debug("PROCESS_TOOL_EVENTS skipped: dry_run mode")
            return

        from neural_memory.engine.tool_memory import ingest_buffer, process_events

        # Ingest JSONL buffer
        buffer_path = config.data_dir / "tool_events.jsonl"
        ingest_result = await ingest_buffer(
            self._storage,  # type: ignore[arg-type]
            brain_id,
            buffer_path,
            config.tool_memory.max_buffer_lines,
        )
        if ingest_result.events_ingested > 0:
            _logger.debug(
                "PROCESS_TOOL_EVENTS: ingested %d events from buffer",
                ingest_result.events_ingested,
            )

        # Process events into neurons/synapses
        result = await process_events(self._storage, brain_id, config.tool_memory)  # type: ignore[arg-type]
        if result.events_processed > 0:
            _logger.debug(
                "PROCESS_TOOL_EVENTS: processed %d events, created %d neurons, %d synapses",
                result.events_processed,
                result.neurons_created,
                result.synapses_created,
            )

    async def _detect_drift(self, report: ConsolidationReport, dry_run: bool) -> None:
        """Run semantic drift detection to find tag synonyms/aliases."""
        _logger = logging.getLogger(__name__)
        if dry_run:
            _logger.debug("DETECT_DRIFT skipped: dry_run mode")
            return

        try:
            from neural_memory.engine.drift_detection import run_drift_detection

            result = await run_drift_detection(self._storage)
            summary: dict[str, Any] = result.get("summary", {})  # type: ignore[assignment]
            total = summary.get("total_clusters", 0)
            if total > 0:
                _logger.debug(
                    "DETECT_DRIFT: found %d clusters (%d merge, %d alias, %d review)",
                    total,
                    summary.get("merge_suggestions", 0),
                    summary.get("alias_suggestions", 0),
                    summary.get("review_suggestions", 0),
                )
                report.extra["drift_clusters"] = total
        except Exception:
            _logger.debug("DETECT_DRIFT failed (non-critical)", exc_info=True)

    async def _smart_merge_pro(self, report: ConsolidationReport, dry_run: bool) -> None:
        """Pro: HNSW-accelerated smart merge (direct import)."""
        _logger = logging.getLogger(__name__)

        merge_fn = None
        try:
            from neural_memory.pro.consolidation.smart_merge import smart_merge

            merge_fn = smart_merge
        except ImportError:
            from neural_memory.plugins import get_consolidation_strategy

            merge_fn = get_consolidation_strategy("smart_merge")
        if merge_fn is None:
            _logger.debug("SMART_MERGE skipped: Pro not available")
            return

        db = getattr(self._storage, "_db", None)
        if db is None:
            _logger.debug("SMART_MERGE skipped: storage is not InfinityDB")
            return

        try:
            result = await merge_fn(db, dry_run=dry_run)
            merged_count = result.get("merged_count", 0)
            if merged_count > 0:
                report.neurons_pruned += merged_count
                report.extra["smart_merge_count"] = merged_count
                _logger.info("SMART_MERGE: %d neurons merged via Pro HNSW clustering", merged_count)
        except Exception:
            _logger.debug("SMART_MERGE failed (non-critical)", exc_info=True)
