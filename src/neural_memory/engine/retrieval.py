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
from neural_memory.engine.lifecycle import ReinforcementManager
from neural_memory.engine.reflex_activation import CoActivation, ReflexActivation
from neural_memory.engine.retrieval_context import format_context, reconstitute_answer
from neural_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from neural_memory.engine.write_queue import DeferredWriteQueue
from neural_memory.extraction.parser import QueryIntent, QueryParser, Stimulus

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage


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

        # 5. Find matching fibers
        fibers_matched = await self._find_matching_fibers(activations)

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

        # 7. Reconstitute answer and context
        # Use co-activated neurons as priority intersections
        co_activated_ids = [neuron_id for co in co_activations for neuron_id in co.neuron_ids]
        all_intersections = co_activated_ids + [
            n for n in intersections if n not in co_activated_ids
        ]

        answer, confidence = await reconstitute_answer(
            self._storage,
            activations,
            all_intersections,
            stimulus,
        )

        context, tokens_used = await format_context(
            self._storage,
            activations,
            fibers_matched,
            max_tokens,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 8. Reinforce accessed memories (deferred to after response)
        # "Neurons that fire together wire together" — recalled memories
        # become easier to find next time
        if activations and confidence > 0.3:
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
            answer=answer,
            confidence=confidence,
            depth_used=depth,
            neurons_activated=len(activations),
            fibers_matched=[f.id for f in fibers_matched],
            subgraph=subgraph,
            context=context,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            co_activations=co_activations,
            metadata={
                "query_intent": stimulus.intent.value,
                "anchors_found": sum(len(a) for a in anchor_sets),
                "intersections": len(all_intersections),
                "co_activations": len(co_activations),
                "use_reflex": self._use_reflex,
            },
        )

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
            await self._defer_co_activated(co_activations)

        return activations, intersections, co_activations

    async def _defer_co_activated(
        self,
        co_activations: list[CoActivation],
    ) -> None:
        """Defer Hebbian strengthening writes to the write queue.

        Reads existing synapses to determine update vs create, but
        defers the actual writes to flush time.
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
                    await self._defer_reinforce_or_create(a, b, delta, initial_weight)

    async def _defer_reinforce_or_create(
        self,
        neuron_a: str,
        neuron_b: str,
        delta: float,
        initial_weight: float,
    ) -> None:
        """Check synapse existence (read) and defer the write."""
        # Check A->B
        forward = await self._storage.get_synapses(source_id=neuron_a, target_id=neuron_b)
        if forward:
            reinforced = forward[0].reinforce(delta)
            self._write_queue.defer_synapse_update(reinforced)
            return

        # Check B->A
        reverse = await self._storage.get_synapses(source_id=neuron_b, target_id=neuron_a)
        if reverse:
            reinforced = reverse[0].reinforce(delta)
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
