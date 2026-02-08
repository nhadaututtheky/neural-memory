"""Reflex retrieval pipeline - the main memory retrieval mechanism."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime
from typing import TYPE_CHECKING

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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
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
    ) -> None:
        """
        Initialize the retrieval pipeline.

        Args:
            storage: Storage backend
            config: Brain configuration
            parser: Custom query parser (creates default if None)
            use_reflex: If True, use ReflexActivation; else use SpreadingActivation
        """
        self._storage = storage
        self._config = config
        self._parser = parser or QueryParser()
        self._use_reflex = use_reflex
        self._activator = SpreadingActivation(storage, config)
        self._reflex_activator = ReflexActivation(storage, config)
        self._reinforcer = ReinforcementManager(
            reinforcement_delta=config.reinforcement_delta,
        )
        self._write_queue = DeferredWriteQueue()

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

        if max_tokens is None:
            max_tokens = self._config.max_context_tokens

        if reference_time is None:
            reference_time = datetime.now()

        # 1. Parse query into stimulus
        stimulus = self._parser.parse(query, reference_time)

        # 2. Auto-detect depth if not specified
        if depth is None:
            depth = self._detect_depth(stimulus)

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
        activations = await self._deprioritize_disputed(activations)

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

        context, tokens_used = await format_context(
            self._storage,
            activations,
            fibers_matched,
            max_tokens,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 8. Reinforce accessed memories (deferred to after response)
        if activations and reconstruction.confidence > 0.3:
            try:
                top_neuron_ids = [
                    nid
                    for nid, _ in sorted(
                        activations.items(),
                        key=lambda x: x[1].activation_level,
                        reverse=True,
                    )[:10]
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
            },
        )

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
        route = QueryRouter().route(stimulus)
        metadata = route.metadata or {}
        traversal = metadata.get("traversal", "")

        if not traversal:
            return None

        # Find seed neuron from entities or keywords
        seed_id = await self._find_seed_neuron(stimulus)
        if seed_id is None and traversal != "temporal_range":
            return None

        if traversal == "causal":
            direction = metadata.get("direction", "causes")
            chain = await trace_causal_chain(
                self._storage,
                seed_id,
                direction,
                max_depth=5,  # type: ignore[arg-type]
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
                max_steps=10,  # type: ignore[arg-type]
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

        Reads existing synapses to determine update vs create, but
        defers the actual writes to flush time.

        When activation levels are available, passes them to the formal
        Hebbian learning rule for principled weight updates.
        """
        threshold = self._config.hebbian_threshold
        delta = self._config.hebbian_delta
        initial_weight = self._config.hebbian_initial_weight

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
                    await self._defer_reinforce_or_create(
                        a,
                        b,
                        delta,
                        initial_weight,
                        pre_act,
                        post_act,
                    )

                    # Persist co-activation event for associative inference
                    source_anchor = co.source_anchors[0] if co.source_anchors else None
                    self._write_queue.defer_co_activation(a, b, co.binding_strength, source_anchor)

    async def _defer_reinforce_or_create(
        self,
        neuron_a: str,
        neuron_b: str,
        delta: float,
        initial_weight: float,
        pre_activation: float | None = None,
        post_activation: float | None = None,
    ) -> None:
        """Check synapse existence (read) and defer the write."""
        # Check A->B
        forward = await self._storage.get_synapses(source_id=neuron_a, target_id=neuron_b)
        if forward:
            reinforced = forward[0].reinforce(
                delta,
                pre_activation=pre_activation,
                post_activation=post_activation,
            )
            self._write_queue.defer_synapse_update(reinforced)
            return

        # Check B->A
        reverse = await self._storage.get_synapses(source_id=neuron_b, target_id=neuron_a)
        if reverse:
            reinforced = reverse[0].reinforce(
                delta,
                pre_activation=post_activation,
                post_activation=pre_activation,
            )
            self._write_queue.defer_synapse_update(reinforced)
            return

        # No existing synapse — defer creation
        synapse = Synapse.create(
            source_id=neuron_a,
            target_id=neuron_b,
            type=SynapseType.RELATED_TO,
            weight=initial_weight,
        )
        self._write_queue.defer_synapse_create(synapse)

    def _apply_lateral_inhibition(
        self,
        activations: dict[str, ActivationResult],
    ) -> dict[str, ActivationResult]:
        """Apply lateral inhibition: top-K neurons survive, rest are suppressed.

        This is the competition phase — highly activated neurons inhibit
        weaker competitors, sharpening the activation landscape.

        Args:
            activations: Current activation results

        Returns:
            New dict with suppressed activations applied
        """
        k = self._config.lateral_inhibition_k
        factor = self._config.lateral_inhibition_factor
        threshold = self._config.activation_threshold

        if len(activations) <= k:
            return activations

        # Sort by activation level descending
        sorted_items = sorted(
            activations.items(),
            key=lambda x: x[1].activation_level,
            reverse=True,
        )

        # Top-K survive unchanged
        winner_ids = {item[0] for item in sorted_items[:k]}

        result: dict[str, ActivationResult] = {}
        for neuron_id, activation in sorted_items:
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
    ) -> dict[str, ActivationResult]:
        """Reduce activation of disputed neurons by 50%.

        Neurons marked with _disputed metadata get their activation
        halved, making them less likely to appear in results. Superseded
        neurons are suppressed even further (75% reduction).

        Args:
            activations: Current activation results

        Returns:
            New dict with disputed neurons deprioritized
        """
        if not activations:
            return activations

        disputed_factor = 0.5
        superseded_factor = 0.25

        # Batch-fetch neurons to check for disputed metadata
        neuron_ids = list(activations.keys())
        neurons = await self._storage.get_neurons_batch(neuron_ids)

        result: dict[str, ActivationResult] = {}
        for neuron_id, activation in activations.items():
            neuron = neurons.get(neuron_id)
            if neuron is not None and neuron.metadata.get("_disputed"):
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

        return result

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

        # 1. TIME ANCHORS FIRST (primary)
        time_anchors: list[str] = []
        for hint in stimulus.time_hints:
            neurons = await self._storage.find_neurons(
                type=NeuronType.TIME,
                time_range=(hint.absolute_start, hint.absolute_end),
                limit=5,
            )
            time_anchors.extend(n.id for n in neurons)

        if time_anchors:
            anchor_sets.append(time_anchors)

        # 2 & 3. Entity + keyword anchors (parallel)
        entity_tasks = [
            self._storage.find_neurons(content_contains=entity.text, limit=3)
            for entity in stimulus.entities
        ]
        keyword_tasks = [
            self._storage.find_neurons(content_contains=keyword, limit=2)
            for keyword in stimulus.keywords[:5]
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
        def _fiber_score(fiber: Fiber) -> float:
            freshness = 0.5
            if fiber.last_conducted:
                hours_ago = (datetime.now() - fiber.last_conducted).total_seconds() / 3600
                freshness = max(0.1, 1.0 / (1.0 + math.exp((hours_ago - 72) / 36)))
            return fiber.salience * freshness * fiber.conductivity

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
