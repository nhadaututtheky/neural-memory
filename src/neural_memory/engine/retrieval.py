"""Reflex retrieval pipeline - the main memory retrieval mechanism."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.activation import ActivationResult, SpreadingActivation
from neural_memory.engine.reflex_activation import CoActivation, ReflexActivation
from neural_memory.engine.retrieval_context import format_context, reconstitute_answer
from neural_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from neural_memory.extraction.parser import QueryIntent, QueryParser, Stimulus

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

        context = await format_context(
            self._storage,
            activations,
            fibers_matched,
            max_tokens,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            answer=answer,
            confidence=confidence,
            depth_used=depth,
            neurons_activated=len(activations),
            fibers_matched=[f.id for f in fibers_matched],
            subgraph=subgraph,
            context=context,
            latency_ms=latency_ms,
            co_activations=co_activations,
            metadata={
                "query_intent": stimulus.intent.value,
                "anchors_found": sum(len(a) for a in anchor_sets),
                "intersections": len(all_intersections),
                "co_activations": len(co_activations),
                "use_reflex": self._use_reflex,
            },
        )

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
        # Get all fibers containing any anchor neurons
        all_anchors = [a for anchors in anchor_sets for a in anchors]
        fibers: list[Fiber] = []
        seen_fiber_ids: set[str] = set()

        for anchor_id in all_anchors:
            matching_fibers = await self._storage.find_fibers(
                contains_neuron=anchor_id,
                limit=10,
            )
            for fiber in matching_fibers:
                if fiber.id not in seen_fiber_ids:
                    fibers.append(fiber)
                    seen_fiber_ids.add(fiber.id)

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

        # Update fiber conductivity for accessed fibers
        for fiber in fibers:
            conducted_fiber = fiber.conduct(conducted_at=reference_time)
            await self._storage.update_fiber(conducted_fiber)

        # Hebbian strengthening: co-activated neurons wire together
        if co_activations:
            await self._strengthen_co_activated(co_activations)

        return activations, intersections, co_activations

    async def _strengthen_co_activated(
        self,
        co_activations: list[CoActivation],
    ) -> None:
        """
        Strengthen or create synapses between co-activated neurons.

        Implements Hebbian plasticity: neurons that fire together
        get their connections reinforced. Only processes co-activations
        with binding_strength >= config.hebbian_threshold and with
        2+ neuron_ids (a pair to connect).

        Args:
            co_activations: List of co-activation records from retrieval
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

            # Process all pairs in the co-activation set
            for i in range(len(neuron_ids)):
                for j in range(i + 1, len(neuron_ids)):
                    a, b = neuron_ids[i], neuron_ids[j]
                    await self._reinforce_or_create_synapse(a, b, delta, initial_weight)

    async def _reinforce_or_create_synapse(
        self,
        neuron_a: str,
        neuron_b: str,
        delta: float,
        initial_weight: float,
    ) -> None:
        """
        Reinforce an existing synapse between two neurons, or create one.

        Checks both directions (A->B and B->A) since RELATED_TO is
        bidirectional and may be stored in either direction.

        Args:
            neuron_a: First neuron ID
            neuron_b: Second neuron ID
            delta: Weight increase for reinforcement
            initial_weight: Weight for newly created synapses
        """
        # Check A->B
        forward = await self._storage.get_synapses(source_id=neuron_a, target_id=neuron_b)
        if forward:
            reinforced = forward[0].reinforce(delta)
            await self._storage.update_synapse(reinforced)
            return

        # Check B->A
        reverse = await self._storage.get_synapses(source_id=neuron_b, target_id=neuron_a)
        if reverse:
            reinforced = reverse[0].reinforce(delta)
            await self._storage.update_synapse(reinforced)
            return

        # No existing synapse — create a new RELATED_TO connection
        synapse = Synapse.create(
            source_id=neuron_a,
            target_id=neuron_b,
            type=SynapseType.RELATED_TO,
            weight=initial_weight,
        )
        await self._storage.add_synapse(synapse)

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

        # 2. Entity anchors (secondary - constrained by time context)
        entity_anchors: list[str] = []
        for entity in stimulus.entities:
            neurons = await self._storage.find_neurons(
                content_contains=entity.text,
                limit=3,
            )
            entity_anchors.extend(n.id for n in neurons)

        if entity_anchors:
            anchor_sets.append(entity_anchors)

        # 3. Keyword anchors (tertiary)
        keyword_anchors: list[str] = []
        for keyword in stimulus.keywords[:5]:
            neurons = await self._storage.find_neurons(
                content_contains=keyword,
                limit=2,
            )
            keyword_anchors.extend(n.id for n in neurons)

        if keyword_anchors:
            anchor_sets.append(keyword_anchors)

        return anchor_sets

    async def _find_anchors(self, stimulus: Stimulus) -> list[list[str]]:
        """Find anchor neurons for each signal type."""
        anchor_sets: list[list[str]] = []

        # Time anchors
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

        # Entity anchors
        entity_anchors: list[str] = []
        for entity in stimulus.entities:
            neurons = await self._storage.find_neurons(
                content_contains=entity.text,
                limit=3,
            )
            entity_anchors.extend(n.id for n in neurons)

        if entity_anchors:
            anchor_sets.append(entity_anchors)

        # Keyword anchors
        keyword_anchors: list[str] = []
        for keyword in stimulus.keywords[:5]:  # Limit keywords
            neurons = await self._storage.find_neurons(
                content_contains=keyword,
                limit=2,
            )
            keyword_anchors.extend(n.id for n in neurons)

        if keyword_anchors:
            anchor_sets.append(keyword_anchors)

        return anchor_sets

    async def _find_matching_fibers(
        self,
        activations: dict[str, ActivationResult],
    ) -> list[Fiber]:
        """Find fibers that contain activated neurons."""
        fibers: list[Fiber] = []
        seen_fiber_ids: set[str] = set()

        # Get highly activated neurons
        top_neurons = sorted(
            activations.values(),
            key=lambda a: a.activation_level,
            reverse=True,
        )[:20]

        for activation in top_neurons:
            matching = await self._storage.find_fibers(
                contains_neuron=activation.neuron_id,
                limit=3,
            )

            for fiber in matching:
                if fiber.id not in seen_fiber_ids:
                    fibers.append(fiber)
                    seen_fiber_ids.add(fiber.id)

        # Sort by salience
        fibers.sort(key=lambda f: f.salience, reverse=True)

        return fibers[:10]  # Limit to top 10

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
