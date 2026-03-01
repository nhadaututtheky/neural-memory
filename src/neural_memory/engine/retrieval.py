"""Reflex retrieval pipeline - the main memory retrieval mechanism."""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.activation import ActivationResult, SpreadingActivation
from neural_memory.engine.causal_traversal import (
    trace_causal_chain,
    trace_event_sequence,
)
from neural_memory.engine.lifecycle import ReinforcementManager
from neural_memory.engine.reconstruction import (
    SynthesisMethod,
    format_causal_chain,
    format_event_sequence,
    format_temporal_range,
    reconstruct_answer,
)
from neural_memory.engine.reflex_activation import CoActivation, ReflexActivation
from neural_memory.engine.retrieval_context import format_context
from neural_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from neural_memory.engine.stabilization import StabilizationConfig, stabilize
from neural_memory.engine.write_queue import DeferredWriteQueue
from neural_memory.extraction.parser import QueryIntent, QueryParser, Stimulus
from neural_memory.extraction.router import QueryRouter
from neural_memory.utils.timeutils import utcnow

__all__ = ["DepthLevel", "ReflexPipeline", "RetrievalResult"]

logger = logging.getLogger(__name__)

_UNSET = object()  # Sentinel for lazy-init cache

# Morphological expansion constants for query term expansion.
_EXPANSION_SUFFIXES: tuple[str, ...] = (
    "tion",
    "ment",
    "ing",
    "ed",
    "er",
    "ity",
    "ness",
    "ize",
    "ise",
    "ate",
)
_EXPANSION_PREFIXES: tuple[str, ...] = ("un", "re", "pre", "de", "dis")

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.engine.depth_prior import AdaptiveDepthSelector, DepthDecision
    from neural_memory.engine.embedding.provider import EmbeddingProvider
    from neural_memory.storage.base import NeuralStorage


def _fiber_valid_at(fiber: Fiber, dt: datetime) -> bool:
    """Check if a fiber is temporally valid at the given datetime.

    A fiber is valid if its time window contains dt. Missing bounds
    are treated as unbounded (open interval).
    """
    if fiber.time_start is not None and fiber.time_start > dt:
        return False
    if fiber.time_end is not None and fiber.time_end < dt:
        return False
    return True


class ReflexPipeline:
    """
    Main retrieval engine - the "consciousness" of the memory system.

    The reflex pipeline:
    1. Decomposes queries into activation signals (Stimulus)
    2. Finds anchor neurons matching signals
    3. Spreads activation through the graph
    4. Finds intersection points
    5. Extracts relevant subgraph
    6. Reconstitutes answer/context

    This mimics human memory retrieval - associative recall through
    spreading activation rather than database search.
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: BrainConfig,
        parser: QueryParser | None = None,
        use_reflex: bool = True,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """
        Initialize the retrieval pipeline.

        Args:
            storage: Storage backend
            config: Brain configuration
            parser: Custom query parser (creates default if None)
            use_reflex: If True, use ReflexActivation; else use SpreadingActivation
            embedding_provider: Optional embedding provider for semantic fallback
        """
        self._storage = storage
        self._config = config
        self._parser = parser or QueryParser()
        self._use_reflex = use_reflex
        self._embedding_provider = embedding_provider
        self._activator = SpreadingActivation(storage, config)
        self._reflex_activator = ReflexActivation(storage, config)
        self._reinforcer = ReinforcementManager(
            reinforcement_delta=config.reinforcement_delta,
        )
        self._write_queue = DeferredWriteQueue()
        self._query_router = QueryRouter()
        self._cached_encryptor: Any = _UNSET

        # Adaptive depth selection (Bayesian priors)
        self._adaptive_selector: AdaptiveDepthSelector | None = None
        if config.adaptive_depth_enabled:
            from neural_memory.engine.depth_prior import AdaptiveDepthSelector

            self._adaptive_selector = AdaptiveDepthSelector(
                storage,
                epsilon=config.adaptive_depth_epsilon,
            )

    def _get_encryptor(self) -> Any:
        """Get cached MemoryEncryptor instance, or None if encryption disabled."""
        if self._cached_encryptor is not _UNSET:
            return self._cached_encryptor
        try:
            from neural_memory.unified_config import get_config as _get_cfg

            _cfg = _get_cfg()
            if _cfg.encryption.enabled:
                from pathlib import Path

                from neural_memory.safety.encryption import MemoryEncryptor

                _keys_dir_str = _cfg.encryption.keys_dir
                _keys_dir = Path(_keys_dir_str) if _keys_dir_str else (_cfg.data_dir / "keys")
                self._cached_encryptor = MemoryEncryptor(keys_dir=_keys_dir)
            else:
                self._cached_encryptor = None
        except Exception:
            self._cached_encryptor = None
        return self._cached_encryptor

    async def query(
        self,
        query: str,
        depth: DepthLevel | None = None,
        max_tokens: int | None = None,
        reference_time: datetime | None = None,
        valid_at: datetime | None = None,
    ) -> RetrievalResult:
        """
        Execute the retrieval pipeline.

        Args:
            query: The query text
            depth: Retrieval depth (auto-detect if None)
            max_tokens: Maximum tokens in context
            reference_time: Reference time for temporal parsing

        Returns:
            RetrievalResult with answer and context
        """
        start_time = time.perf_counter()

        # Clear stale writes from any previous failed query
        self._write_queue.clear()

        if max_tokens is None:
            max_tokens = self._config.max_context_tokens
        max_tokens = min(max_tokens, 200_000)

        if reference_time is None:
            reference_time = utcnow()

        # 1. Parse query into stimulus
        stimulus = self._parser.parse(query, reference_time)

        # 2. Auto-detect depth if not specified
        _depth_decision: DepthDecision | None = None
        if depth is None:
            rule_depth = self._detect_depth(stimulus)
            if self._adaptive_selector is not None:
                try:
                    _depth_decision = await self._adaptive_selector.select_depth(
                        stimulus,
                        rule_depth,
                    )
                    depth = _depth_decision.depth
                except NotImplementedError:
                    # Storage doesn't support depth priors (e.g. InMemoryStorage)
                    depth = rule_depth
                except Exception:
                    logger.debug("Adaptive depth selection failed, using rule-based", exc_info=True)
                    depth = rule_depth
            else:
                depth = rule_depth

        # 2.5 Temporal reasoning fast-path (v0.19.0)
        temporal_result = await self._try_temporal_reasoning(
            stimulus, depth, reference_time, start_time
        )
        if temporal_result is not None:
            return temporal_result

        # 3. Find anchor neurons (time-first)
        anchor_sets = await self._find_anchors_time_first(stimulus)

        # Choose activation method
        if self._use_reflex:
            # Reflex activation: trail-based through fiber pathways
            activations, intersections, co_activations = await self._reflex_query(
                anchor_sets,
                reference_time,
            )
        else:
            # Classic spreading activation
            activations, intersections = await self._activator.activate_from_multiple(
                anchor_sets,
                max_hops=self._depth_to_hops(depth),
            )
            co_activations = []

        # 4.5 Lateral inhibition: top-K winners suppress competitors
        activations = self._apply_lateral_inhibition(activations)

        # 4.6 Stabilization: iterative dampening until convergence
        activations, _stab_report = stabilize(activations, StabilizationConfig())

        # 4.7 Deprioritize disputed neurons (conflict resolution)
        activations, disputed_ids = await self._deprioritize_disputed(activations)

        # 4.8 Sufficiency check: early exit if signal is too weak
        from neural_memory.engine.sufficiency import check_sufficiency

        _sufficiency = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=intersections if not self._use_reflex else [],
            stab_converged=_stab_report.converged,
            stab_neurons_removed=_stab_report.neurons_removed,
        )

        if not _sufficiency.sufficient:
            _early_latency = (time.perf_counter() - start_time) * 1000
            _early_result = RetrievalResult(
                answer=None,
                confidence=_sufficiency.confidence,
                depth_used=depth,
                neurons_activated=len(activations),
                fibers_matched=[],
                subgraph=Subgraph(
                    neuron_ids=list(activations.keys()),
                    synapse_ids=[],
                    anchor_ids=[a for anchors in anchor_sets for a in anchors],
                ),
                context="",
                latency_ms=_early_latency,
                co_activations=co_activations,
                synthesis_method="insufficient_signal",
                metadata={
                    "query_intent": stimulus.intent.value,
                    "anchors_found": sum(len(a) for a in anchor_sets),
                    "sufficiency_gate": _sufficiency.gate,
                    "sufficiency_reason": _sufficiency.reason,
                    "sufficiency_confidence": _sufficiency.confidence,
                },
            )
            # Flush any pending writes even on early exit
            if self._write_queue.pending_count > 0:
                try:
                    await self._write_queue.flush(self._storage)
                except Exception:
                    logger.debug("Deferred write flush failed (non-critical)", exc_info=True)
            return _early_result

        # 5. Find matching fibers
        fibers_matched = await self._find_matching_fibers(activations, valid_at=valid_at)

        # 6. Extract subgraph
        neuron_ids, synapse_ids = await self._activator.get_activated_subgraph(
            activations,
            min_activation=self._config.activation_threshold,
            max_neurons=50,
        )

        subgraph = Subgraph(
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_ids=[a for anchors in anchor_sets for a in anchors],
        )

        # 7. Reconstruct answer from activated subgraph
        co_activated_ids = [neuron_id for co in co_activations for neuron_id in co.neuron_ids]
        all_intersections = co_activated_ids + [
            n for n in intersections if n not in co_activated_ids
        ]

        reconstruction = await reconstruct_answer(
            self._storage,
            activations,
            all_intersections,
            fibers_matched,
        )

        # Create encryptor for decryption if encryption is enabled (cached)
        _encryptor = self._get_encryptor()
        _brain_id = self._storage._current_brain_id or "" if _encryptor else ""

        context, tokens_used = await format_context(
            self._storage,
            activations,
            fibers_matched,
            max_tokens,
            encryptor=_encryptor,
            brain_id=_brain_id,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 8. Reinforce accessed memories (deferred to after response)
        if activations and reconstruction.confidence > 0.3:
            try:
                top_neuron_ids = [
                    nid
                    for nid, _ in heapq.nlargest(
                        10,
                        activations.items(),
                        key=lambda x: x[1].activation_level,
                    )
                ]
                top_synapse_ids = subgraph.synapse_ids[:20] if subgraph.synapse_ids else None
                await self._reinforcer.reinforce(self._storage, top_neuron_ids, top_synapse_ids)
            except Exception:
                logger.debug("Reinforcement failed (non-critical)", exc_info=True)

        result = RetrievalResult(
            answer=reconstruction.answer,
            confidence=reconstruction.confidence,
            depth_used=depth,
            neurons_activated=len(activations),
            fibers_matched=[f.id for f in fibers_matched],
            subgraph=subgraph,
            context=context,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            co_activations=co_activations,
            score_breakdown=reconstruction.score_breakdown,
            contributing_neurons=reconstruction.contributing_neuron_ids,
            synthesis_method=reconstruction.method.value,
            metadata={
                "query_intent": stimulus.intent.value,
                "anchors_found": sum(len(a) for a in anchor_sets),
                "intersections": len(all_intersections),
                "co_activations": len(co_activations),
                "use_reflex": self._use_reflex,
                "stabilization_iterations": _stab_report.iterations,
                "stabilization_converged": _stab_report.converged,
                "disputed_ids": disputed_ids,
                "sufficiency_gate": _sufficiency.gate,
                "sufficiency_confidence": _sufficiency.confidence,
            },
        )

        # Record calibration feedback (non-critical)
        try:
            await self._storage.save_calibration_record(
                gate=_sufficiency.gate,
                predicted_sufficient=True,
                actual_confidence=reconstruction.confidence,
                actual_fibers=len(fibers_matched),
                query_intent=stimulus.intent.value,
            )
        except (AttributeError, Exception):
            # AttributeError: storage doesn't have calibration mixin (e.g. InMemoryStorage)
            logger.debug("Calibration record save failed (non-critical)", exc_info=True)

        # Record adaptive depth outcome (non-critical)
        if _depth_decision is not None:
            result.metadata["depth_selection"] = {
                "method": _depth_decision.method,
                "reason": _depth_decision.reason,
                "exploration": _depth_decision.exploration,
            }
            if self._adaptive_selector is not None:
                try:
                    await self._adaptive_selector.record_outcome(
                        stimulus=stimulus,
                        depth_used=depth,
                        confidence=reconstruction.confidence,
                        fibers_matched=len(fibers_matched),
                    )
                except (NotImplementedError, Exception):
                    logger.debug("Depth prior update failed (non-critical)", exc_info=True)

        # Optionally attach workflow suggestions (non-critical)
        try:
            from neural_memory.engine.workflow_suggest import suggest_next_action

            suggestions = await suggest_next_action(
                self._storage,
                stimulus.intent.value,
                self._config,
            )
            if suggestions:
                result.metadata["workflow_suggestions"] = [
                    {
                        "action": s.action_type,
                        "confidence": round(s.confidence, 4),
                        "source_habit": s.source_habit,
                    }
                    for s in suggestions[:3]
                ]
        except Exception:
            logger.debug("Workflow suggestion failed (non-critical)", exc_info=True)

        # Flush deferred writes (fiber conductivity, Hebbian strengthening)
        if self._write_queue.pending_count > 0:
            try:
                await self._write_queue.flush(self._storage)
            except Exception:
                logger.debug("Deferred write flush failed (non-critical)", exc_info=True)

        return result

    def _detect_depth(self, stimulus: Stimulus) -> DepthLevel:
        """Auto-detect required depth from query intent."""
        # Deep questions need full exploration
        if stimulus.intent in (QueryIntent.ASK_WHY, QueryIntent.ASK_FEELING):
            return DepthLevel.DEEP

        # Pattern questions need cross-time analysis
        if stimulus.intent == QueryIntent.ASK_PATTERN:
            return DepthLevel.HABIT

        # Contextual questions need some exploration
        if stimulus.intent in (QueryIntent.ASK_HOW, QueryIntent.COMPARE):
            return DepthLevel.CONTEXT

        # Check for context keywords
        context_words = {"before", "after", "then", "trước", "sau", "rồi"}
        query_words = set(stimulus.raw_query.lower().split())
        if query_words & context_words:
            return DepthLevel.CONTEXT

        # Complexity-based depth: multiple entities/time hints = intersection query
        signal_count = len(stimulus.entities) + len(stimulus.time_hints)
        if signal_count >= 3 or len(stimulus.keywords) >= 5:
            return DepthLevel.CONTEXT
        if signal_count >= 2:
            return DepthLevel.CONTEXT

        # Simple queries use instant retrieval
        return DepthLevel.INSTANT

    async def _try_temporal_reasoning(
        self,
        stimulus: Stimulus,
        depth: DepthLevel,
        reference_time: datetime,
        start_time: float,
    ) -> RetrievalResult | None:
        """Attempt specialized traversal for causal/temporal queries.

        This is a fast-path shortcut that bypasses the full activation
        pipeline when the query is clearly causal or temporal AND the
        specialized traversal finds results. Returns None to fall through
        to the standard pipeline otherwise.
        """
        route = self._query_router.route(stimulus)
        metadata = route.metadata or {}
        traversal = metadata.get("traversal", "")

        if not traversal:
            return None

        # Find seed neuron from entities or keywords
        seed_id = await self._find_seed_neuron(stimulus)
        if seed_id is None and traversal != "temporal_range":
            return None

        if traversal == "causal":
            assert seed_id is not None  # guarded by None check above
            direction = metadata.get("direction", "causes")
            chain = await trace_causal_chain(
                self._storage,
                seed_id,
                direction,
                max_depth=5,
            )
            if not chain.steps:
                return None

            answer = format_causal_chain(chain)
            return self._build_temporal_result(
                answer=answer,
                confidence=min(1.0, chain.total_weight),
                depth=depth,
                neuron_ids=[s.neuron_id for s in chain.steps],
                method=SynthesisMethod.CAUSAL_CHAIN,
                start_time=start_time,
                intent=stimulus.intent.value,
            )

        if traversal == "temporal_range" and stimulus.time_hints:
            hint = stimulus.time_hints[0]
            from neural_memory.engine.causal_traversal import query_temporal_range

            fibers = await query_temporal_range(
                self._storage, hint.absolute_start, hint.absolute_end
            )
            if not fibers:
                return None

            answer = format_temporal_range(fibers)
            return self._build_temporal_result(
                answer=answer,
                confidence=min(1.0, 0.3 + 0.1 * len(fibers)),
                depth=depth,
                neuron_ids=[],
                method=SynthesisMethod.TEMPORAL_SEQUENCE,
                start_time=start_time,
                intent=stimulus.intent.value,
                fiber_ids=[f.id for f in fibers],
            )

        if traversal == "event_sequence" and seed_id is not None:
            direction = metadata.get("direction", "forward")
            sequence = await trace_event_sequence(
                self._storage,
                seed_id,
                direction,
                max_steps=10,
            )
            if not sequence.events:
                return None

            answer = format_event_sequence(sequence)
            return self._build_temporal_result(
                answer=answer,
                confidence=min(1.0, 0.3 + 0.1 * len(sequence.events)),
                depth=depth,
                neuron_ids=[e.neuron_id for e in sequence.events],
                method=SynthesisMethod.TEMPORAL_SEQUENCE,
                start_time=start_time,
                intent=stimulus.intent.value,
            )

        return None

    async def _find_seed_neuron(self, stimulus: Stimulus) -> str | None:
        """Find the best seed neuron for temporal reasoning.

        Searches entities first (highest specificity), then keywords.
        Returns the first matching neuron ID, or None.
        """
        # Try entities first
        for entity in stimulus.entities:
            neurons = await self._storage.find_neurons(content_contains=entity.text, limit=1)
            if neurons:
                return neurons[0].id

        # Fall back to keywords
        for keyword in stimulus.keywords:
            neurons = await self._storage.find_neurons(content_contains=keyword, limit=1)
            if neurons:
                return neurons[0].id

        return None

    def _build_temporal_result(
        self,
        *,
        answer: str,
        confidence: float,
        depth: DepthLevel,
        neuron_ids: list[str],
        method: SynthesisMethod,
        start_time: float,
        intent: str,
        fiber_ids: list[str] | None = None,
    ) -> RetrievalResult:
        """Build a RetrievalResult for temporal reasoning responses."""
        latency_ms = (time.perf_counter() - start_time) * 1000
        return RetrievalResult(
            answer=answer,
            confidence=confidence,
            depth_used=depth,
            neurons_activated=len(neuron_ids),
            fibers_matched=fiber_ids or [],
            subgraph=Subgraph(neuron_ids=neuron_ids, synapse_ids=[], anchor_ids=[]),
            context=answer,
            latency_ms=latency_ms,
            synthesis_method=method.value,
            metadata={
                "query_intent": intent,
                "temporal_reasoning": True,
            },
        )

    def _depth_to_hops(self, depth: DepthLevel) -> int:
        """Convert depth level to maximum hops."""
        mapping = {
            DepthLevel.INSTANT: 1,
            DepthLevel.CONTEXT: 3,
            DepthLevel.HABIT: 4,
            DepthLevel.DEEP: self._config.max_spread_hops,
        }
        return mapping.get(depth, 2)

    async def _reflex_query(
        self,
        anchor_sets: list[list[str]],
        reference_time: datetime,
    ) -> tuple[dict[str, ActivationResult], list[str], list[CoActivation]]:
        """
        Execute hybrid reflex + classic activation.

        Strategy:
        1. Run reflex trail activation through fiber pathways (fast, focused)
        2. Run limited classic BFS to discover neurons outside fibers (coverage)
        3. Merge results: reflex activations are primary, classic fills gaps
        """
        # Get all fibers containing any anchor neurons (batch query)
        all_anchors = [a for anchors in anchor_sets for a in anchors]
        fibers = await self._storage.find_fibers_batch(all_anchors, limit_per_neuron=10)

        # If no fibers found, fall back entirely to classic activation
        if not fibers:
            activations, intersections = await self._activator.activate_from_multiple(
                anchor_sets,
                max_hops=self._config.max_spread_hops,
            )
            return activations, intersections, []

        # --- Phase 1: Reflex activation (primary) ---
        reflex_activations, co_activations = await self._reflex_activator.activate_with_co_binding(
            anchor_sets=anchor_sets,
            fibers=fibers,
            reference_time=reference_time,
        )

        # --- Phase 2: Limited classic BFS (discovery) ---
        discovery_hops = max(1, self._config.max_spread_hops // 2)
        classic_activations, classic_intersections = await self._activator.activate_from_multiple(
            anchor_sets,
            max_hops=discovery_hops,
        )

        # --- Phase 3: Merge results ---
        discovery_dampen = 0.6
        activations = dict(reflex_activations)

        for neuron_id, classic_result in classic_activations.items():
            existing = activations.get(neuron_id)
            dampened_level = classic_result.activation_level * discovery_dampen

            if existing is None or dampened_level > existing.activation_level:
                activations[neuron_id] = ActivationResult(
                    neuron_id=neuron_id,
                    activation_level=dampened_level,
                    hop_distance=classic_result.hop_distance,
                    path=classic_result.path,
                    source_anchor=classic_result.source_anchor,
                )

        # Merge intersections
        co_intersection_ids = [neuron_id for co in co_activations for neuron_id in co.neuron_ids]
        intersections = co_intersection_ids + [
            n for n in classic_intersections if n not in co_intersection_ids
        ]

        # Defer fiber conductivity updates (non-blocking)
        for fiber in fibers:
            conducted_fiber = fiber.conduct(conducted_at=reference_time)
            self._write_queue.defer_fiber_update(conducted_fiber)

        # Defer Hebbian strengthening (non-blocking)
        if co_activations:
            await self._defer_co_activated(co_activations, activations=activations)

        return activations, intersections, co_activations

    async def _defer_co_activated(
        self,
        co_activations: list[CoActivation],
        activations: dict[str, ActivationResult] | None = None,
    ) -> None:
        """Defer Hebbian strengthening writes to the write queue.

        Uses batch synapse lookups to reduce per-pair queries.
        """
        threshold = self._config.hebbian_threshold
        delta = self._config.hebbian_delta
        initial_weight = self._config.hebbian_initial_weight

        # Collect all neuron pairs that need synapse lookup
        pairs_to_check: list[tuple[str, str, float | None, float | None]] = []

        for co in co_activations:
            if co.binding_strength < threshold:
                continue

            neuron_ids = sorted(co.neuron_ids)
            if len(neuron_ids) < 2:
                continue

            for i in range(len(neuron_ids)):
                for j in range(i + 1, len(neuron_ids)):
                    a, b = neuron_ids[i], neuron_ids[j]
                    pre_act = (
                        activations[a].activation_level
                        if activations and a in activations
                        else None
                    )
                    post_act = (
                        activations[b].activation_level
                        if activations and b in activations
                        else None
                    )
                    pairs_to_check.append((a, b, pre_act, post_act))

            # Persist co-activation event
            source_anchor = co.source_anchors[0] if co.source_anchors else None
            for i in range(len(neuron_ids)):
                for j in range(i + 1, len(neuron_ids)):
                    self._write_queue.defer_co_activation(
                        neuron_ids[i], neuron_ids[j], co.binding_strength, source_anchor
                    )

        if not pairs_to_check:
            return

        # Batch fetch: get all synapses for involved neurons in one query
        all_neuron_ids = list({nid for pair in pairs_to_check for nid in pair[:2]})
        outgoing = await self._storage.get_synapses_for_neurons(all_neuron_ids, direction="out")

        # Build lookup: (source, target) -> Synapse
        existing_map: dict[tuple[str, str], Synapse] = {}
        for synapses in outgoing.values():
            for syn in synapses:
                existing_map[(syn.source_id, syn.target_id)] = syn

        # Process pairs using cached lookups
        for a, b, pre_act, post_act in pairs_to_check:
            forward = existing_map.get((a, b))
            reverse = existing_map.get((b, a))

            if forward:
                reinforced = forward.reinforce(
                    delta,
                    pre_activation=pre_act,
                    post_activation=post_act,
                )
                self._write_queue.defer_synapse_update(reinforced)
            elif reverse:
                reinforced = reverse.reinforce(
                    delta,
                    pre_activation=post_act,
                    post_activation=pre_act,
                )
                self._write_queue.defer_synapse_update(reinforced)
            else:
                synapse = Synapse.create(
                    source_id=a,
                    target_id=b,
                    type=SynapseType.RELATED_TO,
                    weight=initial_weight,
                )
                self._write_queue.defer_synapse_create(synapse)

    def _apply_lateral_inhibition(
        self,
        activations: dict[str, ActivationResult],
    ) -> dict[str, ActivationResult]:
        """Apply cluster-aware lateral inhibition.

        Instead of global top-K, group neurons by source_anchor and
        allow top winners per cluster, preserving diversity across
        different query aspects.
        """
        k = self._config.lateral_inhibition_k
        factor = self._config.lateral_inhibition_factor
        threshold = self._config.activation_threshold

        if len(activations) <= k:
            return activations

        # Group by source_anchor (cluster)
        clusters: dict[str | None, list[tuple[str, ActivationResult]]] = {}
        for neuron_id, activation in activations.items():
            anchor = activation.source_anchor
            clusters.setdefault(anchor, []).append((neuron_id, activation))

        # Sort each cluster by activation level
        for cluster_key in clusters:
            clusters[cluster_key].sort(key=lambda x: x[1].activation_level, reverse=True)

        # Distribute K across clusters proportionally, minimum 1 per cluster
        num_clusters = len(clusters)
        if num_clusters == 0:
            return activations

        per_cluster = max(1, -(-k // num_clusters))  # ceiling division
        winner_ids: set[str] = set()

        for items in clusters.values():
            for nid, _act in items[:per_cluster]:
                winner_ids.add(nid)

        # If we still have budget, fill from global top using heapq for O(n log k)
        if len(winner_ids) < k:
            remaining_budget = k - len(winner_ids)
            # Fetch slightly more than needed to account for already-selected winners
            top_candidates = heapq.nlargest(
                remaining_budget + len(winner_ids),
                activations.items(),
                key=lambda x: x[1].activation_level,
            )
            for nid, _act in top_candidates:
                if nid not in winner_ids:
                    winner_ids.add(nid)
                if len(winner_ids) >= k:
                    break

        result: dict[str, ActivationResult] = {}
        for neuron_id, activation in activations.items():
            if neuron_id in winner_ids:
                result[neuron_id] = activation
            else:
                suppressed_level = activation.activation_level * factor
                if suppressed_level >= threshold:
                    result[neuron_id] = ActivationResult(
                        neuron_id=neuron_id,
                        activation_level=suppressed_level,
                        hop_distance=activation.hop_distance,
                        path=activation.path,
                        source_anchor=activation.source_anchor,
                    )

        return result

    async def _deprioritize_disputed(
        self,
        activations: dict[str, ActivationResult],
    ) -> tuple[dict[str, ActivationResult], list[str]]:
        """Reduce activation of disputed neurons by 50%.

        Neurons marked with _disputed metadata get their activation
        halved, making them less likely to appear in results. Superseded
        neurons are suppressed even further (75% reduction).

        Args:
            activations: Current activation results

        Returns:
            Tuple of (new dict with disputed neurons deprioritized, list of disputed neuron IDs)
        """
        if not activations:
            return activations, []

        disputed_factor = 0.5
        superseded_factor = 0.25

        # Batch-fetch neurons to check for disputed metadata
        neuron_ids = list(activations.keys())
        neurons = await self._storage.get_neurons_batch(neuron_ids)

        disputed_ids: list[str] = []
        result: dict[str, ActivationResult] = {}
        for neuron_id, activation in activations.items():
            neuron = neurons.get(neuron_id)
            if neuron is not None and neuron.metadata.get("_disputed"):
                disputed_ids.append(neuron_id)
                factor = (
                    superseded_factor if neuron.metadata.get("_superseded") else disputed_factor
                )
                new_level = activation.activation_level * factor
                if new_level >= self._config.activation_threshold:
                    result[neuron_id] = ActivationResult(
                        neuron_id=neuron_id,
                        activation_level=new_level,
                        hop_distance=activation.hop_distance,
                        path=activation.path,
                        source_anchor=activation.source_anchor,
                    )
            else:
                result[neuron_id] = activation

        return result, disputed_ids

    def _expand_query_terms(self, keywords: list[str]) -> list[str]:
        """Expand query keywords with basic stemming and synonyms.

        Adds common morphological variants so that 'auth' matches
        'authentication', 'authorize', etc.
        """
        expanded: list[str] = list(keywords)
        seen = {k.lower() for k in keywords}

        for kw in keywords:
            kw_lower = kw.lower()
            # If keyword looks like a stem (short), try common expansions
            if 3 <= len(kw_lower) <= 6:
                for suffix in _EXPANSION_SUFFIXES:
                    candidate = kw_lower + suffix
                    if candidate not in seen:
                        expanded.append(candidate)
                        seen.add(candidate)
                        break  # Only add first plausible expansion

            # If keyword is long, try extracting stem
            for suffix in _EXPANSION_SUFFIXES:
                if kw_lower.endswith(suffix) and len(kw_lower) - len(suffix) >= 3:
                    stem = kw_lower[: -len(suffix)]
                    if stem not in seen:
                        expanded.append(stem)
                        seen.add(stem)
                    break

        return expanded

    async def _find_embedding_anchors(self, query: str, top_k: int = 10) -> list[str]:
        """Find anchor neurons via embedding similarity.

        Embeds the query, then finds neurons whose stored embeddings
        (in metadata['_embedding']) are above the similarity threshold.
        """
        if self._embedding_provider is None:
            return []

        try:
            query_vec = await self._embedding_provider.embed(query)
        except Exception:
            logger.debug("Embedding query failed (non-critical)", exc_info=True)
            return []

        # Get all anchor neurons (with embeddings) - limit search scope
        candidates = await self._storage.find_neurons(limit=500)

        # Collect candidates with embeddings, compute similarity in parallel
        embed_pairs: list[tuple[str, list[float]]] = []
        for neuron in candidates:
            stored_embedding = neuron.metadata.get("_embedding")
            if stored_embedding and isinstance(stored_embedding, list):
                embed_pairs.append((neuron.id, stored_embedding))

        if not embed_pairs:
            return []

        async def _compute_sim(nid: str, stored: list[float]) -> tuple[str, float]:
            try:
                sim = await self._embedding_provider.similarity(query_vec, stored)  # type: ignore[union-attr]
                return (nid, sim)
            except Exception:
                return (nid, 0.0)

        results = await asyncio.gather(*[_compute_sim(nid, emb) for nid, emb in embed_pairs])
        scored = [(nid, sim) for nid, sim in results if sim >= 0.7]

        # Sort by similarity descending, return top-K IDs
        scored.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scored[:top_k]]

    async def _find_anchors_time_first(self, stimulus: Stimulus) -> list[list[str]]:
        """
        Find anchor neurons with time as primary signal.

        Priority order:
        1. Time neurons (weight 1.0) - temporal context
        2. Entity neurons (weight 0.8) - who/what
        3. Action neurons (weight 0.6) - verbs
        4. Concept neurons (weight 0.4) - abstract
        """
        anchor_sets: list[list[str]] = []

        # 1. TIME ANCHORS FIRST (primary) — batch via asyncio.gather
        time_anchors: list[str] = []
        if stimulus.time_hints:
            time_tasks = [
                self._storage.find_neurons(
                    type=NeuronType.TIME,
                    time_range=(hint.absolute_start, hint.absolute_end),
                    limit=5,
                )
                for hint in stimulus.time_hints
            ]
            time_results = await asyncio.gather(*time_tasks)
            for neurons in time_results:
                time_anchors.extend(n.id for n in neurons)

        if time_anchors:
            anchor_sets.append(time_anchors)

        # 2 & 3. Entity + keyword anchors (parallel)
        entity_tasks = [
            self._storage.find_neurons(content_contains=entity.text, limit=3)
            for entity in stimulus.entities
        ]

        # Expand keywords for better recall
        expanded_keywords = self._expand_query_terms(list(stimulus.keywords[:5]))
        keyword_tasks = [
            self._storage.find_neurons(content_contains=keyword, limit=2)
            for keyword in expanded_keywords[:8]  # cap at 8 to limit queries
        ]

        all_tasks = entity_tasks + keyword_tasks
        if all_tasks:
            all_results = await asyncio.gather(*all_tasks)

            entity_anchors: list[str] = []
            for neurons in all_results[: len(entity_tasks)]:
                entity_anchors.extend(n.id for n in neurons)

            keyword_anchors: list[str] = []
            for neurons in all_results[len(entity_tasks) :]:
                keyword_anchors.extend(n.id for n in neurons)

            if entity_anchors:
                anchor_sets.append(entity_anchors)
            if keyword_anchors:
                anchor_sets.append(keyword_anchors)

        # 4. EMBEDDING FALLBACK - when no substring anchors found
        if not anchor_sets and self._embedding_provider is not None:
            embedding_anchors = await self._find_embedding_anchors(stimulus.raw_query)
            if embedding_anchors:
                anchor_sets.append(embedding_anchors)

        return anchor_sets

    async def _find_matching_fibers(
        self,
        activations: dict[str, ActivationResult],
        valid_at: datetime | None = None,
    ) -> list[Fiber]:
        """Find fibers that contain activated neurons (batch query)."""
        # Get highly activated neurons
        top_neurons = sorted(
            activations.values(),
            key=lambda a: a.activation_level,
            reverse=True,
        )[:20]

        top_neuron_ids = [a.neuron_id for a in top_neurons]
        fibers = await self._storage.find_fibers_batch(top_neuron_ids, limit_per_neuron=3)

        # Apply point-in-time temporal filter
        if valid_at is not None:
            fibers = [f for f in fibers if _fiber_valid_at(f, valid_at)]

        # Sort by composite score: salience * freshness * conductivity
        # Doc-trained fibers start at lower salience (ceiling 0.5) and EPISODIC stage,
        # so lifecycle naturally handles ranking without retrieval-time hacks.
        fw = self._config.freshness_weight

        def _fiber_score(fiber: Fiber) -> float:
            recency = 0.5
            if fiber.last_conducted:
                hours_ago = (utcnow() - fiber.last_conducted).total_seconds() / 3600
                recency = max(0.1, 1.0 / (1.0 + math.exp((hours_ago - 72) / 36)))

            base_score = fiber.salience * recency * fiber.conductivity

            # Creation-age freshness penalty (opt-in via freshness_weight > 0)
            if fw > 0.0 and fiber.created_at:
                from neural_memory.safety.freshness import evaluate_freshness

                age_result = evaluate_freshness(fiber.created_at)
                # fw=0: 1.0x | fw=0.5: 0.55x-1.0x | fw=1.0: 0.1x-1.0x
                base_score *= (1.0 - fw) + fw * age_result.score

            return base_score

        fibers.sort(key=_fiber_score, reverse=True)

        return fibers[:10]

    async def query_with_stimulus(
        self,
        stimulus: Stimulus,
        depth: DepthLevel | None = None,
        max_tokens: int | None = None,
    ) -> RetrievalResult:
        """
        Execute retrieval with a pre-parsed stimulus.

        Useful when you want to control the parsing or reuse a stimulus.
        """
        return await self.query(
            stimulus.raw_query,
            depth=depth,
            max_tokens=max_tokens,
        )
