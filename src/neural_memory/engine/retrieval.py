"""Reflex retrieval pipeline - the main memory retrieval mechanism."""

from __future__ import annotations

import asyncio
import collections
import heapq
import logging
import math
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import NeuronType
from neural_memory.core.synapse import (
    ACTIVE_ROLE_TYPES,
    REINFORCEMENT_TYPES,
    SEQUENTIAL_TYPES,
    SUPERSESSION_SOURCE_IS_NEWER,
    SUPERSESSION_TYPES,
    WEAKENING_TYPES,
    Synapse,
    SynapseType,
)
from neural_memory.engine.activation import ActivationResult, SpreadingActivation
from neural_memory.engine.causal_traversal import (
    trace_causal_chain,
    trace_event_sequence,
)
from neural_memory.engine.lifecycle import ReinforcementManager
from neural_memory.engine.query_expansion import expand_via_graph
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
from neural_memory.engine.score_fusion import (
    RankedAnchor,
    rrf_fuse,
    rrf_to_activation_levels,
)
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
    from neural_memory.engine.ppr_activation import PPRActivation
    from neural_memory.engine.session_state import SessionState
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

        # Auto-create embedding provider if enabled but not passed
        if embedding_provider is None and config.embedding_enabled:
            try:
                from neural_memory.engine.semantic_discovery import _create_provider

                self._embedding_provider: EmbeddingProvider | None = _create_provider(config)
            except Exception:
                logger.debug("Could not auto-create embedding provider", exc_info=True)
                self._embedding_provider = None
        else:
            self._embedding_provider = embedding_provider
        self._activator = SpreadingActivation(storage, config)
        self._reflex_activator = ReflexActivation(storage, config)

        # PPR activator (lazy: only create if strategy requires it)
        self._ppr_activator: PPRActivation | None = None
        if config.activation_strategy in ("ppr", "hybrid", "auto"):
            from neural_memory.engine.ppr_activation import PPRActivation

            self._ppr_activator = PPRActivation(storage, config)

        self._reinforcer = ReinforcementManager(
            reinforcement_delta=config.reinforcement_delta,
        )
        self._write_queue = DeferredWriteQueue()
        self._query_router = QueryRouter()
        self._supersession_map: dict[str, str] = {}
        self._cached_encryptor: Any = _UNSET
        self._encryptor_cached_at: float = 0.0
        self._encryptor_ttl: float = 300.0  # Re-check encryption config every 5 min

        # Predictive priming caches (per-session, keyed by session_id, LRU-bounded)
        self._activation_caches: collections.OrderedDict[str, Any] = collections.OrderedDict()
        self._priming_metrics: collections.OrderedDict[str, Any] = collections.OrderedDict()
        self._max_session_cache = 256

        # Warm-start: cached activation levels from prior sessions (ActivationCache).
        # Populated via set_warm_activations() at startup.
        self._warm_activations: dict[str, float] | None = None

        # Adaptive depth selection (Bayesian priors)
        self._adaptive_selector: AdaptiveDepthSelector | None = None
        if config.adaptive_depth_enabled:
            from neural_memory.engine.depth_prior import AdaptiveDepthSelector

            self._adaptive_selector = AdaptiveDepthSelector(
                storage,
                epsilon=config.adaptive_depth_epsilon,
            )

    def set_warm_activations(self, warm: dict[str, float] | None) -> None:
        """Set warm activation levels for anchor boosting (from ActivationCache).

        Called at startup after loading cache snapshot. Anchors present in
        `warm` start with elevated initial activation (max of default and cached).
        Pass None to disable warm-start.
        """
        self._warm_activations = warm

    def _get_encryptor(self) -> Any:
        """Get cached MemoryEncryptor instance, or None if encryption disabled."""
        import time

        now = time.monotonic()
        if (
            self._cached_encryptor is not _UNSET
            and (now - self._encryptor_cached_at) < self._encryptor_ttl
        ):
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
        self._encryptor_cached_at = now
        return self._cached_encryptor

    async def query(
        self,
        query: str,
        depth: DepthLevel | None = None,
        max_tokens: int | None = None,
        reference_time: datetime | None = None,
        valid_at: datetime | None = None,
        tags: set[str] | None = None,
        session_id: str | None = None,
        exclude_ephemeral: bool = False,
        tag_mode: str = "and",
        as_of: datetime | None = None,
        simhash_threshold: int | None = None,
        exclude_reflexes: bool = False,
    ) -> RetrievalResult:
        """
        Execute the retrieval pipeline.

        Args:
            query: The query text
            depth: Retrieval depth (auto-detect if None)
            max_tokens: Maximum tokens in context
            reference_time: Reference time for temporal parsing
            simhash_threshold: Per-query SimHash pre-filter override (None = use brain config)

        Returns:
            RetrievalResult with answer and context
        """
        start_time = time.perf_counter()
        _phase_timings: dict[str, float] = {}

        # Clear stale writes from any previous failed query
        self._write_queue.clear()

        if max_tokens is None:
            max_tokens = self._config.max_context_tokens
        max_tokens = min(max_tokens, 200_000)

        if reference_time is None:
            reference_time = utcnow()

        # 1. Parse query into stimulus
        stimulus = self._parser.parse(query, reference_time)
        _phase_timings["parse"] = (time.perf_counter() - start_time) * 1000

        # 2. Auto-detect depth if not specified
        _depth_decision: DepthDecision | None = None
        _session_state = None
        if session_id:
            try:
                from neural_memory.engine.session_state import SessionManager

                _session_state = SessionManager.get_instance().get_or_create(session_id)
            except Exception:
                logger.debug("Failed to load session state for %s", session_id, exc_info=True)

        if depth is None:
            rule_depth = self._detect_depth(stimulus)
            if self._adaptive_selector is not None:
                try:
                    _depth_decision = await self._adaptive_selector.select_depth(
                        stimulus,
                        rule_depth,
                        session_state=_session_state,
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

        # 2.8 Fiber summary tier — lightweight first-pass retrieval
        if self._config.fiber_summary_tier_enabled and depth != DepthLevel.INSTANT:
            fiber_result = await self._try_fiber_summary_tier(
                stimulus, depth, max_tokens, start_time
            )
            if fiber_result is not None:
                return fiber_result

        # 2.9 SimHash pre-filter: exclude distant neurons before anchor search
        exclude_ids: set[str] = set()
        effective_simhash = (
            simhash_threshold
            if simhash_threshold is not None
            else self._config.simhash_prefilter_threshold
        )
        if effective_simhash > 0 and query.strip():
            from neural_memory.engine.simhash_filter import compute_exclude_set

            neuron_hashes = await self._storage.get_neuron_hashes()
            exclude_ids = compute_exclude_set(query, neuron_hashes, effective_simhash)
        _phase_timings["simhash_prefilter"] = (time.perf_counter() - start_time) * 1000

        # 3. Find anchor neurons (time-first) with ranked results
        anchor_sets, ranked_lists = await self._find_anchors_ranked(
            stimulus,
            exclude_ephemeral=exclude_ephemeral,
            exclude_ids=exclude_ids,
            created_before=as_of,
        )

        # 3.5 RRF score fusion: compute initial activation levels from multi-retriever ranks
        # Use dynamic per-brain retriever weights when available
        _rrf_weights: dict[str, float] | None = None
        try:
            _rrf_weights = await self._storage.get_retriever_weights()  # type: ignore[attr-defined]
        except Exception:
            pass  # Storage doesn't support retriever calibration — use defaults

        anchor_activations: dict[str, float] | None = None
        if ranked_lists and any(ranked_lists):
            fused_scores = rrf_fuse(
                ranked_lists,
                k=self._config.rrf_k,
                retriever_weights=_rrf_weights,
            )
            if fused_scores:
                anchor_activations = rrf_to_activation_levels(fused_scores)
        _phase_timings["anchors_rrf"] = (time.perf_counter() - start_time) * 1000

        # 3.7 Predictive priming: merge session-aware activation boosts
        _priming_result = None
        _primed_neuron_ids: set[str] = set()
        if session_id and _session_state is not None:
            try:
                from neural_memory.engine.priming import (
                    ActivationCache,
                    PrimingMetrics,
                    compute_priming,
                    merge_priming_into_activations,
                )

                # Get or create per-session cache and metrics (LRU-bounded)
                if session_id not in self._activation_caches:
                    self._activation_caches[session_id] = ActivationCache()
                    if len(self._activation_caches) > self._max_session_cache:
                        self._activation_caches.popitem(last=False)
                else:
                    self._activation_caches.move_to_end(session_id)
                if session_id not in self._priming_metrics:
                    self._priming_metrics[session_id] = PrimingMetrics()
                    if len(self._priming_metrics) > self._max_session_cache:
                        self._priming_metrics.popitem(last=False)
                else:
                    self._priming_metrics.move_to_end(session_id)

                _act_cache = self._activation_caches[session_id]
                _prim_metrics = self._priming_metrics[session_id]

                # Get recent result neuron IDs from cache for co-activation priming
                _recent_nids = list(_act_cache.get_priming_activations().keys())[:50]

                _priming_result = await compute_priming(
                    storage=self._storage,
                    session_state=_session_state,
                    activation_cache=_act_cache,
                    recent_neuron_ids=_recent_nids,
                    metrics=_prim_metrics,
                )

                if _priming_result.total_primed > 0:
                    _primed_neuron_ids = set(_priming_result.activation_boosts.keys())
                    anchor_activations = merge_priming_into_activations(
                        anchor_activations, _priming_result
                    )
                    logger.debug(
                        "Priming: %d neurons from %s",
                        _priming_result.total_primed,
                        _priming_result.source_counts,
                    )
            except Exception:
                logger.debug("Predictive priming failed (non-critical)", exc_info=True)

        # Choose activation method based on strategy (auto-select from graph density)
        strategy = self._config.activation_strategy
        if strategy == "auto":
            strategy = await self._auto_select_strategy()

        if strategy == "cone":
            # Pro cone queries: HNSW nearest-neighbor (direct import)
            cone_done = False
            cone_fn = None
            try:
                from neural_memory.pro.retrieval.cone_queries import cone_recall

                cone_fn = cone_recall
            except ImportError:
                # Fallback: try plugin registry for third-party extensions
                from neural_memory.plugins import get_retrieval_strategy

                cone_fn = get_retrieval_strategy("cone")
            if cone_fn is not None and self._embedding_provider is not None:
                try:
                    db = getattr(self._storage, "_db", None)
                    if db is not None:
                        query_vec = await self._embedding_provider.embed(query)
                        cone_results = await cone_fn(query_vec, db)
                        activations = {}
                        for cr in cone_results:
                            activations[cr.neuron_id] = ActivationResult(
                                neuron_id=cr.neuron_id,
                                activation_level=cr.combined_score,
                                hop_distance=0,
                                path=[cr.neuron_id],
                                source_anchor=cr.neuron_id,
                            )
                        intersections: list[str] = []
                        co_activations: list[CoActivation] = []
                        cone_done = True
                except Exception:
                    logger.debug("Cone query failed, falling back", exc_info=True)
            if not cone_done:
                logger.debug("Cone unavailable — falling back to classic activation")
                strategy = "classic"

        _hybrid_scope: set[str] | None = None
        if strategy == "hnsw_hybrid":
            # HNSW-first hybrid: narrow candidates via vector search, then scoped cognitive
            hybrid_done = False
            if self._embedding_provider is not None and hasattr(self._storage, "knn_search"):
                try:
                    from neural_memory.pro.retrieval.hybrid_recall import hnsw_hybrid_recall

                    query_vec = await self._embedding_provider.embed(query)
                    hybrid_result = await hnsw_hybrid_recall(
                        query_embedding=query_vec,
                        storage=self._storage,
                        activator=self._activator,
                        anchor_sets=anchor_sets,
                        max_hops=self._depth_to_hops(depth),
                        anchor_activations=anchor_activations,
                        bm25_query=query if hasattr(self._storage, "text_search") else None,
                    )
                    if hybrid_result is not None:
                        activations, intersections, _hybrid_scope = hybrid_result
                        co_activations = []
                        hybrid_done = True
                except Exception:
                    logger.debug("Hybrid recall failed, falling back", exc_info=True)
            if not hybrid_done:
                logger.debug("Hybrid recall unavailable — falling back to classic")
                strategy = "classic"

        if strategy == "ppr" and self._ppr_activator is not None:
            # Personalized PageRank activation
            activations, intersections = await self._ppr_activator.activate_from_multiple(
                anchor_sets,
                anchor_activations=anchor_activations,
            )
            co_activations = []
        elif strategy == "hybrid" and self._ppr_activator is not None:
            # Hybrid: PPR primary + reflex for fiber pathways
            ppr_activations, ppr_intersections = await self._ppr_activator.activate_from_multiple(
                anchor_sets,
                anchor_activations=anchor_activations,
            )
            # Also run reflex if fibers exist
            reflex_activations: dict[str, ActivationResult] = {}
            co_activations = []
            if self._use_reflex:
                reflex_activations, _, co_activations = await self._reflex_query(
                    anchor_sets,
                    reference_time,
                    anchor_activations=anchor_activations,
                )
            # Merge: PPR primary, reflex fills gaps (dampened 0.6x)
            activations = dict(ppr_activations)
            for nid, reflex_result in reflex_activations.items():
                existing = activations.get(nid)
                dampened = reflex_result.activation_level * 0.6
                if existing is None or dampened > existing.activation_level:
                    activations[nid] = ActivationResult(
                        neuron_id=nid,
                        activation_level=dampened,
                        hop_distance=reflex_result.hop_distance,
                        path=reflex_result.path,
                        source_anchor=reflex_result.source_anchor,
                    )
            intersections = ppr_intersections
        elif self._use_reflex:
            # Reflex activation: trail-based through fiber pathways
            activations, intersections, co_activations = await self._reflex_query(
                anchor_sets,
                reference_time,
                anchor_activations=anchor_activations,
            )
        else:
            # Classic spreading activation
            activations, intersections = await self._activator.activate_from_multiple(
                anchor_sets,
                max_hops=self._depth_to_hops(depth),
                anchor_activations=anchor_activations,
                warm_activations=self._warm_activations,
            )
            co_activations = []
        _phase_timings["activation"] = (time.perf_counter() - start_time) * 1000

        # 4.5 Lateral inhibition: top-K winners suppress competitors
        activations = self._apply_lateral_inhibition(activations)

        # 4.6 Stabilization: iterative dampening until convergence
        activations, _stab_report = stabilize(
            activations,
            StabilizationConfig(),
            density_scaling=self._config.graph_density_scaling_enabled,
        )

        # 4.7 Deprioritize disputed neurons (conflict resolution)
        activations, disputed_ids = await self._deprioritize_disputed(activations)

        # 4.75 Apply causal semantics: role-aware post-processing
        activations = await self._apply_causal_semantics(activations)
        _phase_timings["post_activation"] = (time.perf_counter() - start_time) * 1000

        # 4.8 Sufficiency check: early exit if signal is too weak
        from neural_memory.engine.sufficiency import GateCalibration, check_sufficiency

        # Fetch EMA calibration stats (non-critical; falls back gracefully)
        _gate_calibration: dict[str, GateCalibration] | None = None
        try:
            _raw_cal = await self._storage.get_gate_ema_stats()  # type: ignore[attr-defined]
            _gate_calibration = {
                gate: GateCalibration(
                    accuracy=stats["accuracy"],
                    avg_confidence=stats["avg_confidence"],
                    sample_count=int(stats["sample_count"]),
                )
                for gate, stats in _raw_cal.items()
            }
        except Exception:
            logger.debug("Gate calibration fetch failed (non-critical)", exc_info=True)

        # 4.5 Goal-directed recall: compute proximity to active goals (early, shared with familiarity)
        _goal_proximity: dict[str, float] = {}
        if getattr(self._config, "goal_proximity_boost", 0.0) > 0:
            try:
                from neural_memory.engine.goal_proximity import (
                    compute_goal_proximity,
                    find_active_goals,
                )

                active_goals = await find_active_goals(self._storage)
                if active_goals:
                    goal_ids = [g.id for g in active_goals]
                    goal_priorities = {g.id: g.goal_priority for g in active_goals}
                    _parent_map = {g.id: g.parent_goal_id for g in active_goals}
                    _goal_proximity = await compute_goal_proximity(
                        self._storage,
                        goal_ids,
                        max_hops=getattr(self._config, "goal_max_hops", 3),
                        goal_priorities=goal_priorities,
                        parent_map=_parent_map,
                    )
            except Exception:
                logger.debug("Goal proximity computation failed", exc_info=True)

        _sufficiency = check_sufficiency(
            activations=activations,
            anchor_sets=anchor_sets,
            intersections=intersections if not self._use_reflex else [],
            stab_converged=_stab_report.converged,
            stab_neurons_removed=_stab_report.neurons_removed,
            query_intent=stimulus.intent.value,
            calibration=_gate_calibration,
            density_scaling=self._config.graph_density_scaling_enabled,
        )

        if not _sufficiency.sufficient:
            # Dual-process fallback: try familiarity recall before giving up
            if getattr(self._config, "familiarity_fallback_enabled", True):
                _fam_result = await self._familiarity_recall(
                    stimulus=stimulus,
                    activations=activations,
                    anchor_sets=anchor_sets,
                    co_activations=co_activations,
                    depth=depth,
                    query=query,
                    tags=tags,
                    tag_mode=tag_mode,
                    valid_at=valid_at,
                    as_of=as_of,
                    max_tokens=max_tokens,
                    start_time=start_time,
                    phase_timings=_phase_timings,
                    sufficiency_gate=_sufficiency.gate,
                    goal_proximity=_goal_proximity,
                    session_state=_session_state,
                )
                if _fam_result is not None:
                    # Record surfaced fibers from familiarity path
                    if _session_state is not None and _fam_result.fibers_matched:
                        try:
                            _session_state.record_surfaced(_fam_result.fibers_matched)
                        except Exception:
                            logger.debug(
                                "Record surfaced fibers failed (non-critical)",
                                exc_info=True,
                            )
                    # Flush pending writes before returning familiarity result
                    if self._write_queue.pending_count > 0:
                        try:
                            await self._write_queue.flush(self._storage)
                        except Exception:
                            logger.debug(
                                "Deferred write flush failed (non-critical)", exc_info=True
                            )
                    return _fam_result

            _early_latency = (time.perf_counter() - start_time) * 1000
            _phase_timings["early_exit"] = _early_latency
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
                    "phase_timings_ms": _phase_timings,
                },
            )
            # Flush any pending writes even on early exit
            if self._write_queue.pending_count > 0:
                try:
                    await self._write_queue.flush(self._storage)
                except Exception:
                    logger.debug("Deferred write flush failed (non-critical)", exc_info=True)
            return _early_result

        # 4.9 Cross-encoder reranking (optional post-SA refinement)
        if self._config.reranker_enabled and len(activations) > 1:
            try:
                from neural_memory.engine.reranker import reranker_available

                if reranker_available():
                    from neural_memory.engine.reranker import rerank_activations

                    # Fetch content for top candidates
                    top_nids = [
                        nid
                        for nid, _ in sorted(
                            activations.items(),
                            key=lambda x: x[1].activation_level,
                            reverse=True,
                        )
                    ][: self._config.reranker_max_candidates]
                    neuron_batch = await self._storage.get_neurons_batch(top_nids)
                    neuron_contents = {
                        nid: n.content for nid, n in neuron_batch.items() if n.content
                    }

                    if neuron_contents:
                        activations = rerank_activations(
                            query,
                            activations,
                            neuron_contents,
                            model_name=self._config.reranker_model,
                            blend_weight=self._config.reranker_blend_weight,
                            min_score=self._config.reranker_min_score,
                            max_candidates=self._config.reranker_max_candidates,
                            limit=50,
                        )
                        logger.debug(
                            "Reranked %d → %d activations",
                            len(neuron_contents),
                            len(activations),
                        )
            except ImportError:
                logger.debug("Reranker not available (sentence-transformers not installed)")
            except Exception:
                logger.debug("Reranking failed (non-critical)", exc_info=True)

        # 5. Find matching fibers
        query_tokens = set(query.lower().split())
        # Pass session topics for affinity scoring (A8 Phase 1)
        _session_topics: set[str] = set()
        if _session_state is not None:
            try:
                top_topics = _session_state.get_topic_weights(limit=5)
                _session_topics = {t for t, w in top_topics.items() if w > 0.3}
            except Exception:
                pass

        # Preference query detection for preference-aware scoring
        _is_preference_query = False
        if getattr(self._config, "preference_detection_enabled", True):
            from neural_memory.engine.preference_detector import is_preference_query

            _is_preference_query = is_preference_query(query)

        # Temporal query detection for event anchor boosting
        _temporal_event_anchors: set[str] = set()
        if getattr(self._config, "temporal_routing_enabled", True):
            from neural_memory.engine.temporal_query import detect_temporal_query

            _temporal_signal = detect_temporal_query(query)
            if _temporal_signal is not None:
                _temporal_event_anchors = set(_temporal_signal.event_anchors)

        # Role-aware scoring: detect if query targets assistant or user content
        _role_target: str | None = None
        if getattr(self._config, "role_aware_scoring_enabled", True):
            from neural_memory.engine.role_query import detect_role_target

            _role_result = detect_role_target(query)
            if _role_result is not None:
                _role_target = _role_result.value  # "assistant" or "user"

        fibers_matched = await self._find_matching_fibers(
            activations,
            valid_at=valid_at,
            tags=tags,
            query_tokens=query_tokens,
            tag_mode=tag_mode,
            session_topics=_session_topics,
            created_before=as_of,
            goal_proximity=_goal_proximity,
            session_state=_session_state,
            is_preference_query=_is_preference_query,
            temporal_event_anchors=_temporal_event_anchors,
            role_target=_role_target,
            ranked_lists=ranked_lists,
            query_intent=stimulus.intent.value,
        )
        _phase_timings["fibers"] = (time.perf_counter() - start_time) * 1000

        # 5.5 Causal auto-inclusion: trace CAUSED_BY/LEADS_TO from matched fibers
        _causal_supplement = ""
        if fibers_matched and getattr(self._config, "causal_auto_include", True):
            try:
                from neural_memory.engine.causal_inclusion import gather_causal_context

                _fiber_neuron_ids = [list(f.neuron_ids) for f in fibers_matched[:10]]
                _causal_max_hops = getattr(self._config, "causal_auto_include_max_hops", 2)
                # Budget: 20% of max_tokens (approximate 4 chars/token), min 200 chars
                _causal_budget = max(200, int(max_tokens * 0.2 * 4))
                # Exclude neurons already in matched fibers (dedup with temporal binding)
                _matched_nids: set[str] = set()
                for _f in fibers_matched:
                    _matched_nids.update(_f.neuron_ids)
                _causal_ctx = await gather_causal_context(
                    self._storage,
                    _fiber_neuron_ids,
                    max_hops=_causal_max_hops,
                    max_tokens_budget=_causal_budget,
                    exclude_neuron_ids=_matched_nids,
                )
                _causal_supplement = _causal_ctx.supplement_text
            except Exception:
                logger.debug("Causal auto-inclusion failed (non-critical)", exc_info=True)
        _phase_timings["causal"] = (time.perf_counter() - start_time) * 1000

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
        _brain_id = self._storage.brain_id or "" if _encryptor else ""

        # ── Reflex injection: prepend always-on neurons before regular context ──
        _reflex_prefix = ""
        _reflex_count = 0
        if not exclude_reflexes:
            try:
                reflex_neurons = await self._storage.find_reflex_neurons(limit=50)
                if reflex_neurons:
                    _reflex_count = len(reflex_neurons)
                    _reflex_lines = [f"- {n.content}" for n in reflex_neurons]
                    _reflex_prefix = "[Reflexes]\n" + "\n".join(_reflex_lines) + "\n\n"
            except Exception:
                logger.debug("Reflex injection failed (non-critical)", exc_info=True)

        context, tokens_used = await format_context(
            self._storage,
            activations,
            fibers_matched,
            max_tokens,
            encryptor=_encryptor,
            brain_id=_brain_id,
        )

        if _reflex_prefix:
            context = _reflex_prefix + context

        _phase_timings["reconstruction"] = (time.perf_counter() - start_time) * 1000

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

        # 9. Track access time for lifecycle heat scoring (batch update, non-critical)
        if activations:
            try:
                await self._storage.batch_update_last_accessed(list(activations.keys()))
            except Exception:
                logger.debug("batch_update_last_accessed failed (non-critical)", exc_info=True)

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
                "activation_levels": {
                    nid: round(ar.activation_level, 4) for nid, ar in activations.items()
                },
                "activation_paths": {nid: ar.path for nid, ar in activations.items() if ar.path},
                "phase_timings_ms": _phase_timings,
                "reflex_count": _reflex_count,
            },
        )

        # Attach causal supplement to result metadata
        if _causal_supplement:
            result.metadata["causal_context"] = _causal_supplement

        # Update priming cache and metrics (non-critical)
        if session_id and _priming_result is not None:
            try:
                from neural_memory.engine.priming import record_priming_outcome

                _act_cache = self._activation_caches.get(session_id)
                _prim_metrics = self._priming_metrics.get(session_id)

                # Update activation cache with this query's results
                if _act_cache is not None:
                    activation_levels = {
                        nid: ar.activation_level for nid, ar in activations.items()
                    }
                    _act_cache.update_from_result(activation_levels)

                # Record priming outcome (hit/miss)
                if _prim_metrics is not None and _primed_neuron_ids:
                    _result_nids = set(activations.keys())
                    record_priming_outcome(_prim_metrics, _primed_neuron_ids, _result_nids)
                    result.metadata["priming"] = {
                        "neurons_primed": _priming_result.total_primed,
                        "sources": _priming_result.source_counts,
                        "hit_rate": round(_prim_metrics.hit_rate, 4),
                        "aggressiveness": round(_prim_metrics.aggressiveness_multiplier, 2),
                    }
                    # Store full PrimingResult for proactive hint selection
                    result.metadata["_priming_result"] = _priming_result
            except Exception:
                logger.debug("Priming cache update failed (non-critical)", exc_info=True)

        # Record calibration feedback (non-critical)
        try:
            await self._storage.save_calibration_record(  # type: ignore[attr-defined]
                gate=_sufficiency.gate,
                predicted_sufficient=True,
                actual_confidence=reconstruction.confidence,
                actual_fibers=len(fibers_matched),
                query_intent=stimulus.intent.value,
            )
        except Exception:
            # AttributeError: storage doesn't have calibration mixin (e.g. InMemoryStorage)
            logger.debug("Calibration record save failed (non-critical)", exc_info=True)

        # Record retriever contribution outcomes for dynamic RRF weights (non-critical)
        if ranked_lists and fibers_matched:
            try:
                # Which neurons ended up in final results?
                result_neuron_ids = {
                    next(iter(f.neuron_ids)) for f in fibers_matched if f.neuron_ids
                }
                for ranked_list in ranked_lists:
                    if not ranked_list:
                        continue
                    rtype = ranked_list[0].retriever
                    contributed = any(ra.neuron_id in result_neuron_ids for ra in ranked_list)
                    await self._storage.save_retriever_outcome(  # type: ignore[attr-defined]
                        retriever_type=rtype,
                        contributed=contributed,
                    )
                # Periodic pruning: cap retriever_calibration per type (every ~100 saves)
                import random as _rnd

                if _rnd.random() < 0.01:  # ~1% chance per save → prunes ~every 100 saves
                    try:
                        await self._storage.prune_retriever_calibration()  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                logger.debug("Retriever outcome save failed (non-critical)", exc_info=True)

        # Record adaptive depth outcome (non-critical)
        if _depth_decision is not None:
            result.metadata["depth_selection"] = {
                "method": _depth_decision.method,
                "reason": _depth_decision.reason,
                "exploration": _depth_decision.exploration,
            }
            if self._adaptive_selector is not None:
                try:
                    # Infer agent_used_result from priming hit rate:
                    # If primed neurons appeared in result → agent is using the recall
                    _agent_signal: bool | None = None
                    if _primed_neuron_ids and activations:
                        _result_nids = set(activations.keys())
                        _agent_signal = bool(_primed_neuron_ids & _result_nids)
                    await self._adaptive_selector.record_outcome(
                        stimulus=stimulus,
                        depth_used=depth,
                        confidence=reconstruction.confidence,
                        fibers_matched=len(fibers_matched),
                        agent_used_result=_agent_signal,
                    )
                except Exception:
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

        # Post-recall reconsolidation: recalled memories absorb current context
        if getattr(self._config, "reconsolidation_enabled", True) and fibers_matched:
            try:
                from neural_memory.engine.reconsolidation import reconsolidate_on_recall

                query_tags = set(stimulus.keywords) if stimulus.keywords else set()
                query_entities = [e.text for e in stimulus.entities] if stimulus.entities else []
                brain_id = getattr(self._storage, "_brain_id", "")
                for fiber in fibers_matched[:5]:  # top 5 only
                    if fiber.anchor_neuron_id:
                        await reconsolidate_on_recall(
                            fiber_id=fiber.id,
                            anchor_neuron_id=fiber.anchor_neuron_id,
                            query_tags=query_tags,
                            query_entities=query_entities,
                            storage=self._storage,
                            config=self._config,
                            brain_id=brain_id,
                        )
            except Exception:
                logger.debug("Reconsolidation failed (non-critical)", exc_info=True)

        # Record session query (non-critical)
        if session_id:
            try:
                from neural_memory.engine.session_state import SessionManager

                session_mgr = SessionManager.get_instance()
                session = session_mgr.get_or_create(session_id)
                session.record_query(
                    query=query,
                    depth_used=depth.value,
                    confidence=reconstruction.confidence,
                    fibers_matched=len(fibers_matched),
                    entities=[e.text for e in stimulus.entities] if stimulus.entities else [],
                    keywords=list(stimulus.keywords) if stimulus.keywords else [],
                )
                # Attach session context to result metadata
                session_top_topics = session.get_top_topics()
                if session_top_topics:
                    result.metadata["session_topics"] = session_top_topics
                    result.metadata["session_query_count"] = session.query_count

                # Periodic session summary persist
                if session.needs_persist():
                    try:
                        summary = session.to_summary_dict()
                        await self._storage.save_session_summary(  # type: ignore[attr-defined]
                            session_id=session.session_id,
                            topics=summary["topics"],
                            topic_weights=summary["topic_weights"],
                            top_entities=summary["top_entities"],
                            query_count=summary["query_count"],
                            avg_confidence=summary["avg_confidence"],
                            avg_depth=summary["avg_depth"],
                            started_at=utcnow().isoformat(),
                            ended_at=utcnow().isoformat(),
                        )
                        session.mark_persisted()
                    except Exception:
                        logger.debug("Session summary persist failed (non-critical)", exc_info=True)
            except Exception:
                logger.debug("Session recording failed (non-critical)", exc_info=True)

        # Record surfaced fibers in attention set (anti-redundancy for next query)
        if fibers_matched and _session_state is not None:
            try:
                _session_state.record_surfaced([f.id for f in fibers_matched])
            except Exception:
                logger.debug("Record surfaced fibers failed (non-critical)", exc_info=True)

        # Compute unified confidence score (non-critical)
        try:
            from neural_memory.engine.confidence import ConfidenceWeights, compute_confidence

            _top_fiber = fibers_matched[0] if fibers_matched else None
            _fiber_meta = (_top_fiber.metadata or {}) if _top_fiber else {}
            _fidelity = str(_fiber_meta.get("_fidelity_layer", "detail"))
            _quality = float(_fiber_meta.get("_quality_score", 5.0))
            _is_fam = result.synthesis_method == "familiarity"

            _conf_weights = ConfidenceWeights(
                retrieval=getattr(self._config, "confidence_weight_retrieval", 0.35),
                content_quality=getattr(self._config, "confidence_weight_quality", 0.25),
                fidelity=getattr(self._config, "confidence_weight_fidelity", 0.20),
                freshness=getattr(self._config, "confidence_weight_freshness", 0.20),
            )
            result.confidence_score = compute_confidence(
                retrieval_score=result.confidence,
                sufficiency_confidence=_sufficiency.confidence,
                quality_score=_quality,
                fidelity_layer=_fidelity,
                created_at=_top_fiber.created_at if _top_fiber else None,
                is_familiarity_fallback=_is_fam,
                weights=_conf_weights,
            )
        except Exception:
            logger.debug("Confidence score computation failed (non-critical)", exc_info=True)

        return result

    async def _auto_select_strategy(self) -> str:
        """Auto-select activation strategy based on backend and graph density.

        InfinityDB with HNSW → hnsw_hybrid (vector-scoped cognitive).
        Sparse graph (avg <3 synapses/neuron) → classic BFS reaches more.
        Dense graph (avg >8 synapses/neuron) → PPR dampens hub noise.
        Medium → hybrid.
        """
        # hnsw_hybrid is available for InfinityDB but not auto-selected —
        # embedding call overhead negates the scoped activation benefit at <100K.
        # Use activation_strategy = "hnsw_hybrid" in config to opt in.

        try:
            # exclude_hubs=True: DREAM hub synapses are consolidation
            # artefacts; including them inflates density and misclassifies
            # organic-sparse graphs as dense.
            density = await self._storage.get_graph_density(exclude_hubs=True)  # type: ignore[attr-defined]
        except Exception:
            return "classic"  # Fallback if storage doesn't support it

        if density < 3.0:
            return "classic"
        elif density > 8.0:
            if self._ppr_activator is not None:
                return "ppr"
            return "classic"
        else:
            if self._ppr_activator is not None:
                return "hybrid"
            return "classic"

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

    async def _try_fiber_summary_tier(
        self,
        stimulus: Stimulus,
        depth: DepthLevel,
        max_tokens: int,
        start_time: float,
    ) -> RetrievalResult | None:
        """Step 2.8: Fiber summary first-pass retrieval.

        Searches fiber summaries via FTS5 before the full neuron pipeline.
        If results have sufficient confidence and enough context tokens,
        returns early without running the expensive activation pipeline.
        Returns None to fall through to the standard pipeline otherwise.
        """
        # Build search query from stimulus keywords + entities
        search_terms: list[str] = list(stimulus.keywords)
        for entity in stimulus.entities:
            search_terms.append(entity.text)
        if not search_terms:
            return None

        query_text = " ".join(search_terms)
        try:
            fibers = await self._storage.search_fiber_summaries(query_text, limit=10)
        except Exception:
            logger.debug("Fiber summary search failed, falling through", exc_info=True)
            return None

        if not fibers:
            return None

        # Build context from fiber summaries
        context_parts: list[str] = []
        tokens_used = 0
        for fiber in fibers:
            summary = fiber.summary or ""
            if not summary:
                continue
            estimated_tokens = len(summary) // 4
            if tokens_used + estimated_tokens > max_tokens:
                break
            context_parts.append(summary)
            tokens_used += estimated_tokens

        if not context_parts:
            return None

        # Compute confidence: based on number of matches and token coverage
        match_ratio = min(1.0, len(context_parts) / max(len(search_terms), 1))
        token_ratio = min(1.0, tokens_used / max(max_tokens * 0.3, 1))
        confidence = match_ratio * 0.6 + token_ratio * 0.4

        # Sufficiency gate: only return early if confidence exceeds threshold
        if confidence < self._config.sufficiency_threshold:
            logger.debug(
                "Fiber summary tier: confidence %.2f < threshold %.2f, continuing to neuron pipeline",
                confidence,
                self._config.sufficiency_threshold,
            )
            return None

        context = "\n\n".join(context_parts)
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            "Fiber summary tier sufficient: confidence=%.2f, fibers=%d, tokens=%d, latency=%.1fms",
            confidence,
            len(context_parts),
            tokens_used,
            latency_ms,
        )

        return RetrievalResult(
            answer=context_parts[0] if context_parts else None,
            confidence=confidence,
            depth_used=depth,
            neurons_activated=0,
            fibers_matched=[f.id for f in fibers[: len(context_parts)]],
            subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=[]),
            context=context,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            metadata={"fiber_summary_tier": True, "fibers_searched": len(fibers)},
        )

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
        anchor_activations: dict[str, float] | None = None,
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
                anchor_activations=anchor_activations,
                warm_activations=self._warm_activations,
            )
            return activations, intersections, []

        # --- Phase 1: Reflex activation (primary) ---
        reflex_activations, co_activations = await self._reflex_activator.activate_with_co_binding(
            anchor_sets=anchor_sets,
            fibers=fibers,
            reference_time=reference_time,
            anchor_activations=anchor_activations,
        )

        # --- Phase 2: Limited classic BFS (discovery) ---
        discovery_hops = max(1, self._config.max_spread_hops // 2)
        classic_activations, classic_intersections = await self._activator.activate_from_multiple(
            anchor_sets,
            max_hops=discovery_hops,
            anchor_activations=anchor_activations,
            warm_activations=self._warm_activations,
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
        pairs_to_check: list[tuple[str, str, float, float]] = []

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
                        activations[a].activation_level if activations and a in activations else 0.1
                    )
                    post_act = (
                        activations[b].activation_level if activations and b in activations else 0.1
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

    async def _familiarity_recall(
        self,
        stimulus: Stimulus,
        activations: dict[str, ActivationResult],
        anchor_sets: list[list[str]],
        co_activations: list[CoActivation],
        depth: DepthLevel,
        query: str,
        tags: set[str] | None,
        tag_mode: str,
        valid_at: datetime | None,
        as_of: datetime | None,
        max_tokens: int,
        start_time: float,
        phase_timings: dict[str, float],
        sufficiency_gate: str,
        goal_proximity: dict[str, float] | None = None,
        session_state: SessionState | None = None,
    ) -> RetrievalResult | None:
        """Familiarity-based recall — weaker signal, lower confidence.

        Dual-process theory (Yonelinas 1994): when recollection (full
        activation) fails, fall back to familiarity (relaxed thresholds,
        vague recognition). Returns None if familiarity also fails.
        """
        max_fibers = self._config.familiarity_max_fibers
        confidence_cap = self._config.familiarity_confidence_cap
        fibers_matched: list[Fiber] = []

        # Strategy A: We have activations but they were too weak for sufficiency.
        # Relax the activation threshold by 50% and try fiber matching.
        if activations and sufficiency_gate not in ("no_anchors", "empty_landscape"):
            relaxed_threshold = self._config.activation_threshold * 0.5
            # Filter activations above relaxed threshold
            relaxed_activations = {
                nid: act
                for nid, act in activations.items()
                if act.activation_level >= relaxed_threshold
            }
            if relaxed_activations:
                query_tokens = set(query.lower().split())
                fibers_matched = await self._find_matching_fibers(
                    relaxed_activations,
                    valid_at=valid_at,
                    tags=tags,
                    query_tokens=query_tokens,
                    tag_mode=tag_mode,
                    created_before=as_of,
                    goal_proximity=goal_proximity,
                    session_state=session_state,
                )

        # Strategy B: No activations at all (no_anchors / empty_landscape).
        # Try broader content-based anchor search with query keywords,
        # then run a lightweight activation pass.
        if not fibers_matched and stimulus.keywords:
            broader_anchor_ids: list[str] = []
            for kw in list(stimulus.keywords)[:3]:
                try:
                    found = await self._storage.find_neurons(content_contains=kw, limit=3)
                    broader_anchor_ids.extend(n.id for n in found)
                except Exception:
                    pass

            # Deduplicate
            broader_anchor_ids = list(dict.fromkeys(broader_anchor_ids))

            if broader_anchor_ids:
                try:
                    new_activations, _trace = await self._activator.activate(
                        broader_anchor_ids[:10],
                        min_activation=self._config.activation_threshold * 0.5,
                    )
                    if new_activations:
                        query_tokens = set(query.lower().split())
                        fibers_matched = await self._find_matching_fibers(
                            new_activations,
                            valid_at=valid_at,
                            tags=tags,
                            query_tokens=query_tokens,
                            tag_mode=tag_mode,
                            created_before=as_of,
                            goal_proximity=goal_proximity,
                            session_state=session_state,
                        )
                        # Update activations for subgraph extraction
                        activations = new_activations
                except Exception:
                    logger.debug("Familiarity broader activation failed", exc_info=True)

        # Strategy C: Column fiber summary search (cortical column pattern completion).
        # If both A and B failed, search column fiber summaries for query keywords.
        if not fibers_matched and stimulus.keywords:
            try:
                column_fibers = await self._storage.find_fibers(
                    metadata_key="_column",
                    limit=20,
                )
                if column_fibers:
                    _q_tokens = {kw.lower() for kw in stimulus.keywords}
                    scored_cols: list[tuple[float, Fiber]] = []
                    for cf in column_fibers:
                        if valid_at is not None and not _fiber_valid_at(cf, valid_at):
                            continue
                        summary = (cf.summary or "").lower()
                        if not summary:
                            continue
                        hits = sum(1 for t in _q_tokens if t in summary)
                        if hits > 0:
                            score = hits / max(len(_q_tokens), 1)
                            scored_cols.append((score, cf))
                    scored_cols.sort(key=lambda x: x[0], reverse=True)
                    fibers_matched = [f for _, f in scored_cols[:max_fibers]]

                    # Build minimal activations from column fiber neurons
                    if fibers_matched and not activations:
                        col_neuron_ids = set()
                        for f in fibers_matched:
                            col_neuron_ids.update(f.neuron_ids)
                        for nid in list(col_neuron_ids)[:30]:
                            activations[nid] = ActivationResult(
                                neuron_id=nid,
                                activation_level=0.3,
                                hop_distance=0,
                                path=[nid],
                                source_anchor=nid,
                            )
            except Exception:
                logger.debug("Column fiber summary search failed", exc_info=True)

        if not fibers_matched:
            return None

        # Cap fibers
        fibers_matched = fibers_matched[:max_fibers]

        phase_timings["familiarity"] = (time.perf_counter() - start_time) * 1000

        # Build result with capped confidence
        neuron_ids, synapse_ids = await self._activator.get_activated_subgraph(
            activations,
            min_activation=self._config.activation_threshold * 0.5,
            max_neurons=30,
        )
        subgraph = Subgraph(
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_ids=[a for anchors in anchor_sets for a in anchors],
        )

        reconstruction = await reconstruct_answer(
            self._storage,
            activations,
            [],  # no intersections for familiarity
            fibers_matched,
        )

        _encryptor = self._get_encryptor()
        _brain_id = self._storage.brain_id or "" if _encryptor else ""

        context, tokens_used = await format_context(
            self._storage,
            activations,
            fibers_matched,
            max_tokens,
            encryptor=_encryptor,
            brain_id=_brain_id,
        )

        _latency = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            answer=reconstruction.answer,
            confidence=min(reconstruction.confidence, confidence_cap),
            depth_used=depth,
            neurons_activated=len(activations),
            fibers_matched=[f.id for f in fibers_matched],
            subgraph=subgraph,
            context=context,
            latency_ms=_latency,
            tokens_used=tokens_used,
            co_activations=co_activations,
            contributing_neurons=reconstruction.contributing_neuron_ids,
            synthesis_method="familiarity",
            metadata={
                "query_intent": stimulus.intent.value,
                "familiarity_fallback": True,
                "original_gate": sufficiency_gate,
                "phase_timings_ms": phase_timings,
            },
        )

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
        # Density scaling: increase K for large graphs so that more winners
        # survive, preventing collateral suppression of correct neurons.
        # Homeostatic synaptic scaling — dense networks need more active units.
        if self._config.graph_density_scaling_enabled and len(activations) > k:
            k = max(k, min(k * 3, int(math.sqrt(len(activations)) * 2)))
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

    async def _apply_causal_semantics(
        self,
        activations: dict[str, ActivationResult],
    ) -> dict[str, ActivationResult]:
        """Apply causal role semantics to post-activation scores.

        After spreading activation computes raw scores, this method adjusts
        them based on synapse ROLES — not just weights. Each role has distinct
        behavior:

        - SUPERSESSION: follow chain to latest version, demote outdated
        - REINFORCEMENT: boost target activation additively
        - WEAKENING: halve target activation (capped, supersession-protected)
        - SEQUENTIAL: light priming boost to next step
        - PASSIVE/STRUCTURAL/LATERAL: no special handling (skipped)

        Directionality matters for SUPERSESSION:
        - RESOLVED_BY/FALSIFIED_BY: source=old → target=new (demote source)
        - SUPERSEDES/EVOLVES_FROM: source=new → target=old (demote target)

        Args:
            activations: Current activation results from spreading activation

        Returns:
            New activations dict with causal adjustments applied.
            Populates self._supersession_map for context formatting.
        """
        if not activations:
            return activations

        # Batch-fetch outgoing synapses for all activated neurons
        neuron_ids = list(activations.keys())
        synapses_by_source = await self._storage.get_synapses_for_neurons(
            neuron_ids, direction="out"
        )

        # Filter to only active-role synapses
        active_synapses: dict[str, list[Synapse]] = {}
        for nid, synapses in synapses_by_source.items():
            filtered = [s for s in synapses if s.type in ACTIVE_ROLE_TYPES]
            if filtered:
                active_synapses[nid] = filtered

        result = dict(activations)

        # No causal synapses → skip synapse processing but still apply habit boost
        has_active_synapses = bool(active_synapses)

        # Track supersession: outdated_id → latest_id
        supersession_map: dict[str, str] = {}

        if not has_active_synapses:
            # No synapse-role processing needed, jump to habit boost
            self._supersession_map = supersession_map
            return await self._apply_habit_boost(result)

        # --- SUPERSESSION: determine outdated vs latest, follow chains ---
        # First pass: collect all chain targets for batch prefetch
        chain_targets: set[str] = set()
        for synapses in active_synapses.values():
            for syn in synapses:
                if syn.type in SUPERSESSION_TYPES:
                    chain_targets.add(syn.target_id)

        # Batch-prefetch chain node synapses to avoid N+1
        chain_synapses_cache: dict[str, list[Synapse]] = {}
        if chain_targets:
            unfetched = [t for t in chain_targets if t not in synapses_by_source]
            if unfetched:
                chain_synapses_cache = await self._storage.get_synapses_for_neurons(
                    unfetched, direction="out"
                )

        def _get_outgoing_supersession(nid: str) -> list[Synapse]:
            """Get outgoing supersession synapses from cache."""
            syns = synapses_by_source.get(nid) or chain_synapses_cache.get(nid) or []
            return [s for s in syns if s.type in SUPERSESSION_TYPES]

        # Shared visited set across all supersession edges (H2 fix)
        chain_visited: set[str] = set()

        for source_id, synapses in active_synapses.items():
            supersession_synapses = [s for s in synapses if s.type in SUPERSESSION_TYPES]
            if not supersession_synapses:
                continue

            source_activation = result.get(source_id)
            if source_activation is None:
                continue

            for syn in supersession_synapses:
                # Determine directionality: who is outdated, who is latest?
                if syn.type in SUPERSESSION_SOURCE_IS_NEWER:
                    # SUPERSEDES/EVOLVES_FROM: source=NEW, target=OLD
                    outdated_id = syn.target_id
                    latest_id = source_id
                else:
                    # RESOLVED_BY/FALSIFIED_BY: source=OLD, target=NEW
                    outdated_id = source_id
                    latest_id = syn.target_id

                # Follow chain from latest to find the ULTIMATE latest (max depth 5)
                chain_depth = 0
                current = latest_id
                if current in chain_visited:
                    continue
                local_visited: set[str] = {outdated_id, current}

                while chain_depth < 5:
                    chain_depth += 1
                    next_syns = _get_outgoing_supersession(current)
                    next_hop = None
                    for ns in next_syns:
                        candidate = ns.target_id
                        if ns.type in SUPERSESSION_SOURCE_IS_NEWER:
                            # source=current is newer, target is older — wrong direction
                            continue
                        # RESOLVED_BY direction: target is newer
                        if candidate not in local_visited:
                            next_hop = candidate
                            break

                    # Also check: is current pointed TO by a SUPERSEDES?
                    if next_hop is None:
                        # Check if any node SUPERSEDES current (making current outdated)
                        for ns in next_syns:
                            if ns.type in SUPERSESSION_SOURCE_IS_NEWER:
                                # current SUPERSEDES ns.target → current IS the newer
                                continue
                        break  # current is the ultimate latest

                    if next_hop in local_visited:
                        break  # cycle detected
                    local_visited.add(next_hop)

                    # Prefetch if not cached
                    if next_hop not in synapses_by_source and next_hop not in chain_synapses_cache:
                        hop_syns = await self._storage.get_synapses(source_id=next_hop)
                        chain_synapses_cache[next_hop] = hop_syns

                    current = next_hop

                ultimate_latest = current
                chain_visited.update(local_visited)

                # Boost the latest version
                base_score = source_activation.activation_level
                boosted_score = min(base_score * 1.2, 1.0)

                if (
                    ultimate_latest not in result
                    or result[ultimate_latest].activation_level < boosted_score
                ):
                    result[ultimate_latest] = ActivationResult(
                        neuron_id=ultimate_latest,
                        activation_level=boosted_score,
                        hop_distance=source_activation.hop_distance,
                        path=[*source_activation.path, ultimate_latest],
                        source_anchor=source_activation.source_anchor,
                    )

                # Demote the outdated version to ghost level
                outdated_activation = result.get(outdated_id)
                if outdated_activation is not None:
                    ghost_level = outdated_activation.activation_level * 0.1
                    if ghost_level >= self._config.activation_threshold:
                        result[outdated_id] = ActivationResult(
                            neuron_id=outdated_id,
                            activation_level=ghost_level,
                            hop_distance=outdated_activation.hop_distance,
                            path=outdated_activation.path,
                            source_anchor=outdated_activation.source_anchor,
                        )
                    else:
                        result.pop(outdated_id, None)

                # Record for context formatting (H1 fix)
                supersession_map[outdated_id] = ultimate_latest

        # Store supersession map for context formatting
        self._supersession_map = supersession_map

        # Track which IDs are supersession-protected (cannot be weakened)
        supersession_protected: set[str] = set(supersession_map.values())

        # --- WEAKENING: demote target activation by 50% (capped, C2 fix) ---
        weakening_counts: dict[str, int] = {}
        for synapses in active_synapses.values():
            for syn in synapses:
                if syn.type not in WEAKENING_TYPES:
                    continue
                target_id = syn.target_id
                if target_id in supersession_protected:
                    continue  # C2: don't weaken supersession targets
                activation = result.get(target_id)
                if activation is None:
                    continue
                count = weakening_counts.get(target_id, 0)
                if count >= 1:
                    continue  # C2: cap at one weakening (x0.5 floor)
                weakening_counts[target_id] = count + 1
                demoted_level = activation.activation_level * 0.5
                if demoted_level >= self._config.activation_threshold:
                    result[target_id] = ActivationResult(
                        neuron_id=target_id,
                        activation_level=demoted_level,
                        hop_distance=activation.hop_distance,
                        path=activation.path,
                        source_anchor=activation.source_anchor,
                    )
                else:
                    result.pop(target_id, None)

        # --- REINFORCEMENT: boost target activation additively ---
        reinforcement_boosts: dict[str, float] = {}
        for synapses in active_synapses.values():
            for syn in synapses:
                if syn.type not in REINFORCEMENT_TYPES:
                    continue
                target_id = syn.target_id
                if target_id not in result:
                    continue  # Only boost already-activated neurons
                current_boost = reinforcement_boosts.get(target_id, 0.0)
                reinforcement_boosts[target_id] = min(current_boost + 0.15, 0.3)

        for target_id, boost in reinforcement_boosts.items():
            activation = result.get(target_id)
            if activation is None:
                continue
            new_level = min(activation.activation_level + boost, 1.0)
            result[target_id] = ActivationResult(
                neuron_id=target_id,
                activation_level=new_level,
                hop_distance=activation.hop_distance,
                path=activation.path,
                source_anchor=activation.source_anchor,
            )

        # --- SEQUENTIAL: light priming boost ---
        for synapses in active_synapses.values():
            for syn in synapses:
                if syn.type not in SEQUENTIAL_TYPES:
                    continue
                target_id = syn.target_id
                if target_id not in result:
                    continue  # Only prime already-activated neurons
                activation = result[target_id]
                primed_level = min(activation.activation_level + 0.1, 1.0)
                result[target_id] = ActivationResult(
                    neuron_id=target_id,
                    activation_level=primed_level,
                    hop_distance=activation.hop_distance,
                    path=activation.path,
                    source_anchor=activation.source_anchor,
                )

        # Store supersession map for context formatting
        self._supersession_map = supersession_map

        return await self._apply_habit_boost(result)

    async def _apply_habit_boost(
        self,
        activations: dict[str, ActivationResult],
    ) -> dict[str, ActivationResult]:
        """Boost activation of neurons with proven workflow frequency.

        Neurons with `_habit_frequency` metadata get a proportional boost:
        +0.05 per frequency unit, capped at +0.2 total.

        Args:
            activations: Current activation results

        Returns:
            Activations with habit boosts applied
        """
        result_ids = list(activations.keys())
        if not result_ids:
            return activations

        result = dict(activations)
        neurons_batch = await self._storage.get_neurons_batch(result_ids)
        for nid, neuron in neurons_batch.items():
            freq = neuron.metadata.get("_habit_frequency", 0)
            if not isinstance(freq, (int, float)) or freq <= 0:
                continue
            activation = result.get(nid)
            if activation is None:
                continue
            # Proportional boost: +0.05 per frequency unit, capped at +0.2
            habit_boost = min(float(freq) * 0.05, 0.2)
            boosted_level = min(activation.activation_level + habit_boost, 1.0)
            result[nid] = ActivationResult(
                neuron_id=nid,
                activation_level=boosted_level,
                hop_distance=activation.hop_distance,
                path=activation.path,
                source_anchor=activation.source_anchor,
            )

        return result

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

        Fast path: uses HNSW knn_search when available (O(log N)).
        Fallback: O(N) pairwise scan of stored embeddings.
        """
        if self._embedding_provider is None:
            return []

        try:
            query_vec = await self._embedding_provider.embed(query)
        except Exception:
            logger.debug("Embedding query failed (non-critical)", exc_info=True)
            return []

        # Fast path: HNSW knn_search (InfinityDB)
        if hasattr(self._storage, "knn_search"):
            try:
                knn_results = await self._storage.knn_search(query_vec, k=top_k * 2)
                threshold = self._config.embedding_similarity_threshold
                return [nid for nid, sim in knn_results if sim >= threshold][:top_k]
            except Exception:
                logger.debug("HNSW knn_search failed, falling back to scan", exc_info=True)

        # Fallback: O(N) scan with pairwise similarity
        probe = await self._storage.find_neurons(limit=20)
        has_embeddings = any(n.metadata.get("_embedding") for n in probe)
        if not has_embeddings:
            return []

        candidates = await self._storage.find_neurons(limit=1000)
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
        threshold = self._config.embedding_similarity_threshold
        scored = [(nid, sim) for nid, sim in results if sim >= threshold]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scored[:top_k]]

    async def _find_anchors_ranked(
        self,
        stimulus: Stimulus,
        *,
        exclude_ephemeral: bool = False,
        exclude_ids: set[str] | None = None,
        created_before: datetime | None = None,
    ) -> tuple[list[list[str]], list[list[RankedAnchor]]]:
        """Find anchor neurons with ranked results for RRF fusion.

        Returns both flat anchor_sets (for activation) and ranked lists
        (for RRF score fusion). The ranked lists preserve retriever
        identity and position so RRF can weight them appropriately.

        Priority order:
        1. Time neurons (weight 1.0) - temporal context
        2. Entity neurons (weight 0.8) - who/what
        3. Keyword neurons (weight 0.6) - expanded terms
        4. Embedding neurons (weight 1.0) - semantic similarity
        """
        anchor_sets: list[list[str]] = []
        ranked_lists: list[list[RankedAnchor]] = []
        ephemeral_filter: bool | None = False if exclude_ephemeral else None

        # 1. TIME ANCHORS FIRST (primary) — batch via asyncio.gather
        time_anchors: list[str] = []
        if stimulus.time_hints:
            time_tasks = [
                self._storage.find_neurons(
                    type=NeuronType.TIME,
                    time_range=(hint.absolute_start, hint.absolute_end),
                    limit=5,
                    ephemeral=ephemeral_filter,
                    created_before=created_before,
                )
                for hint in stimulus.time_hints
            ]
            time_results = await asyncio.gather(*time_tasks)
            for neurons in time_results:
                time_anchors.extend(n.id for n in neurons)

        if time_anchors:
            anchor_sets.append(time_anchors)
            ranked_lists.append(
                [
                    RankedAnchor(neuron_id=nid, rank=i + 1, retriever="time")
                    for i, nid in enumerate(time_anchors)
                ]
            )

        # 2 & 3. Entity + keyword anchors
        # Expand keywords for better recall (morphological + smart expansion)
        expanded_keywords = self._expand_query_terms(list(stimulus.keywords[:5]))

        # Smart query expansion: synonyms, abbreviations, cross-language
        from neural_memory.engine.query_expander import expand_terms
        from neural_memory.engine.token_normalizer import normalize_for_search

        expanded_keywords = expand_terms(
            expanded_keywords,
            enable_synonyms=self._config.query_expansion_synonyms,
            enable_abbreviations=self._config.query_expansion_abbreviations,
            enable_cross_language=(
                self._config.query_expansion_cross_language
                and self._embedding_provider is None  # skip if embedding handles cross-lang
            ),
            max_per_term=self._config.query_expansion_max_per_term,
            language=getattr(stimulus, "language", "auto"),
        )

        # Token normalization: add Vietnamese compound + diacritics-stripped variants
        normalized: list[str] = []
        seen_kw: set[str] = set()
        for kw in expanded_keywords[:12]:
            for variant in normalize_for_search(kw):
                if variant not in seen_kw:
                    normalized.append(variant)
                    seen_kw.add(variant)

        # IDF-weighted anchor limits: rare terms get more slots, common fewer
        kw_limits: dict[str, int] = {}
        _default_kw_limit = 2
        if self._config.idf_anchor_enabled:
            try:
                from neural_memory.engine.idf_anchor import compute_keyword_limits

                _kw_df = await self._storage.get_keyword_df_batch(
                    [k.lower() for k in normalized[:15]]
                )
                _total_fibers = await self._storage.get_total_fiber_count()
                kw_limits = compute_keyword_limits(
                    normalized[:15],
                    _kw_df,
                    _total_fibers,
                    min_limit=self._config.idf_anchor_min_limit,
                    max_limit=self._config.idf_anchor_max_limit,
                )
            except Exception:
                logger.debug("IDF anchor limit computation failed (non-critical)", exc_info=True)

        entity_anchors: list[str] = []
        keyword_anchors: list[str] = []

        # Batch path: use find_neurons_by_content_batch when available (InfinityDB)
        _has_batch = hasattr(self._storage, "find_neurons_by_content_batch")
        if _has_batch and (stimulus.entities or normalized):
            entity_terms = [e.text for e in stimulus.entities]
            kw_terms = list(normalized[:15])
            all_terms = entity_terms + kw_terms
            if all_terms:
                max_limit = max(
                    3,  # entity default
                    max(
                        (kw_limits.get(kw, _default_kw_limit) for kw in kw_terms),
                        default=_default_kw_limit,
                    ),
                )
                batch_results = await self._storage.find_neurons_by_content_batch(  # type: ignore[attr-defined]
                    terms=all_terms,
                    limit_per_term=max_limit,
                    ephemeral=ephemeral_filter,
                    created_before=created_before,
                )
                for et in entity_terms:
                    entity_anchors.extend(n.id for n in batch_results.get(et, [])[:3])
                for kw in kw_terms:
                    kw_limit = kw_limits.get(kw, _default_kw_limit)
                    keyword_anchors.extend(n.id for n in batch_results.get(kw, [])[:kw_limit])
        else:
            # Standard path: individual find_neurons calls (SQLite, etc.)
            entity_tasks = [
                self._storage.find_neurons(
                    content_contains=entity.text,
                    limit=3,
                    ephemeral=ephemeral_filter,
                    created_before=created_before,
                )
                for entity in stimulus.entities
            ]
            keyword_tasks = [
                self._storage.find_neurons(
                    content_contains=keyword,
                    limit=kw_limits.get(keyword, _default_kw_limit),
                    ephemeral=ephemeral_filter,
                    created_before=created_before,
                )
                for keyword in normalized[:15]
            ]
            all_tasks = entity_tasks + keyword_tasks
            if all_tasks:
                all_results = await asyncio.gather(*all_tasks)

                for neurons in all_results[: len(entity_tasks)]:
                    entity_anchors.extend(n.id for n in neurons)

                for neurons in all_results[len(entity_tasks) :]:
                    keyword_anchors.extend(n.id for n in neurons)

            if entity_anchors:
                anchor_sets.append(entity_anchors)
                ranked_lists.append(
                    [
                        RankedAnchor(neuron_id=nid, rank=i + 1, retriever="entity")
                        for i, nid in enumerate(entity_anchors)
                    ]
                )
            if keyword_anchors:
                anchor_sets.append(keyword_anchors)
                ranked_lists.append(
                    [
                        RankedAnchor(neuron_id=nid, rank=i + 1, retriever="keyword")
                        for i, nid in enumerate(keyword_anchors)
                    ]
                )

        # 3.5 FUZZY SEARCH — typo tolerance when keyword results are sparse
        if self._config.fuzzy_search_enabled and len(keyword_anchors) < 2:
            try:
                from neural_memory.engine.fuzzy_match import (
                    find_fuzzy_matches,
                    generate_prefix_variants,
                )

                fuzzy_anchors: list[str] = []
                fuzzy_seen: set[str] = set()
                # Use original keywords (not expanded) for fuzzy
                # Collect all prefix queries first, then run in parallel
                _fuzzy_kw_prefix_pairs: list[tuple[str, str]] = []
                for kw in list(stimulus.keywords[:3]):
                    for prefix in generate_prefix_variants(kw):
                        _fuzzy_kw_prefix_pairs.append((kw, prefix))

                _fuzzy_tasks = [
                    self._storage.find_neurons(
                        content_contains=prefix,
                        limit=self._config.fuzzy_search_max_candidates,
                        ephemeral=ephemeral_filter,
                        created_before=created_before,
                    )
                    for _kw, prefix in _fuzzy_kw_prefix_pairs
                ]
                _fuzzy_results = await asyncio.gather(*_fuzzy_tasks) if _fuzzy_tasks else []

                for (kw, _prefix), prefix_neurons in zip(
                    _fuzzy_kw_prefix_pairs, _fuzzy_results, strict=False
                ):
                    # Match against _raw_keywords metadata (short strings),
                    # falling back to content for neurons without raw keywords
                    keyword_to_id: dict[str, str] = {}
                    for n in prefix_neurons:
                        raw_kws = n.metadata.get("_raw_keywords", []) if n.metadata else []
                        if raw_kws:
                            for rk in raw_kws:
                                if rk and rk not in keyword_to_id:
                                    keyword_to_id[rk] = n.id
                        elif n.content and n.content not in keyword_to_id:
                            keyword_to_id[n.content] = n.id
                    matches = find_fuzzy_matches(
                        kw,
                        list(keyword_to_id.keys()),
                        max_distance=self._config.fuzzy_search_max_distance,
                    )
                    for match_content, _dist in matches:
                        nid = keyword_to_id.get(match_content)
                        if nid and nid not in fuzzy_seen:
                            fuzzy_anchors.append(nid)
                            fuzzy_seen.add(nid)

                if fuzzy_anchors:
                    anchor_sets.append(fuzzy_anchors)
                    ranked_lists.append(
                        [
                            RankedAnchor(neuron_id=nid, rank=i + 1, retriever="fuzzy")
                            for i, nid in enumerate(fuzzy_anchors)
                        ]
                    )
            except Exception:
                logger.debug("Fuzzy search failed (non-critical)", exc_info=True)

        # 4. EMBEDDING ANCHORS - parallel source (always, not just fallback)
        if self._embedding_provider is not None:
            embedding_anchors = await self._find_embedding_anchors(stimulus.raw_query)
            if embedding_anchors:
                anchor_sets.append(embedding_anchors)
                ranked_lists.append(
                    [
                        RankedAnchor(neuron_id=nid, rank=i + 1, retriever="embedding")
                        for i, nid in enumerate(embedding_anchors)
                    ]
                )

        # 4.5 BM25 TEXT RELEVANCE — Tantivy full-text search (InfinityDB)
        if hasattr(self._storage, "text_search"):
            try:
                bm25_results = await self._storage.text_search(stimulus.raw_query, limit=15)
                if bm25_results:
                    bm25_anchors = [nid for nid, _score in bm25_results]
                    anchor_sets.append(bm25_anchors)
                    ranked_lists.append(
                        [
                            RankedAnchor(
                                neuron_id=nid,
                                rank=i + 1,
                                retriever="text_relevance",
                                score=score,
                            )
                            for i, (nid, score) in enumerate(bm25_results)
                        ]
                    )
            except Exception:
                logger.debug("BM25 text search failed (non-critical)", exc_info=True)

        # 5. GRAPH EXPANSION — 1-hop neighbors of entity anchors as soft anchors
        if self._config.graph_expansion_enabled and entity_anchors:
            try:
                expansion_ids, expansion_ranked = await expand_via_graph(
                    self._storage,
                    seed_neuron_ids=entity_anchors,
                    max_expansions=self._config.graph_expansion_max,
                    min_synapse_weight=self._config.graph_expansion_min_weight,
                )
                if expansion_ids:
                    anchor_sets.append(expansion_ids)
                    ranked_lists.append(expansion_ranked)
            except Exception:
                logger.debug("Graph expansion failed (non-critical)", exc_info=True)

        # 6. Apply SimHash pre-filter exclusion
        if exclude_ids:
            anchor_sets = [
                [nid for nid in anchors if nid not in exclude_ids] for anchors in anchor_sets
            ]
            ranked_lists = [
                [ra for ra in ranked if ra.neuron_id not in exclude_ids] for ranked in ranked_lists
            ]
            # Remove empty lists
            anchor_sets = [a for a in anchor_sets if a]
            ranked_lists = [r for r in ranked_lists if r]

        return anchor_sets, ranked_lists

    async def _find_matching_fibers(
        self,
        activations: dict[str, ActivationResult],
        valid_at: datetime | None = None,
        tags: set[str] | None = None,
        query_tokens: set[str] | None = None,
        tag_mode: str = "and",
        session_topics: set[str] | None = None,
        created_before: datetime | None = None,
        goal_proximity: dict[str, float] | None = None,
        session_state: SessionState | None = None,
        is_preference_query: bool = False,
        temporal_event_anchors: set[str] | None = None,
        role_target: str | None = None,
        ranked_lists: list[list[RankedAnchor]] | None = None,
        query_intent: str | None = None,
    ) -> list[Fiber]:
        """Find fibers that contain activated neurons (batch query).

        A8 Phase 1 enhancements:
        - Topic affinity boost from session EMA (T1.2)
        - Recent-access boost for active project memories (T1.5)
        - MMR diversity re-ranking to reduce redundancy (T1.1)
        - Early SimHash dedup before cap (T1.3)
        """
        # Get highly activated neurons
        top_neurons = sorted(
            activations.values(),
            key=lambda a: a.activation_level,
            reverse=True,
        )[:20]

        top_neuron_ids = [a.neuron_id for a in top_neurons]
        fibers = await self._storage.find_fibers_batch(
            top_neuron_ids,
            limit_per_neuron=3,
            tags=tags,
            tag_mode=tag_mode,
            created_before=created_before,
        )

        # Apply point-in-time temporal filter
        if valid_at is not None:
            fibers = [f for f in fibers if _fiber_valid_at(f, valid_at)]

        # Sort by composite score: base quality * activation relevance * stage bonus
        # Doc-trained fibers start at lower salience (ceiling 0.5) and EPISODIC stage,
        # so lifecycle naturally handles ranking without retrieval-time hacks.
        fw = self._config.freshness_weight
        halflife = self._config.recency_halflife_hours
        tag_boost = self._config.tag_match_boost
        _topic_affinity_boost = self._config.topic_affinity_boost
        _recent_boost = self._config.recent_access_boost
        _recent_window_hrs = self._config.recent_access_window_days * 24.0
        _session_topics = session_topics or set()
        _goal_proximity = goal_proximity or {}
        _goal_proximity_boost = getattr(self._config, "goal_proximity_boost", 0.25)
        _anti_redundancy = getattr(self._config, "anti_redundancy_penalty", 0.3)
        _preference_boost = getattr(self._config, "preference_boost", 1.5)
        _preference_domain_boost = getattr(self._config, "preference_domain_boost", 0.2)
        _is_pref_query = is_preference_query
        _event_anchors = temporal_event_anchors or set()
        _event_anchor_boost = getattr(self._config, "temporal_event_anchor_boost", 0.3)
        _role_target = role_target
        _role_match_boost = getattr(self._config, "role_match_boost", 1.3)
        _role_mismatch_penalty = getattr(self._config, "role_mismatch_penalty", 0.9)
        _session_state = session_state
        _now = utcnow()

        # --- Hybrid retrieval fusion: pre-compute per-fiber fused scores ---
        _fusion_scores: dict[str, float] = {}
        _fusion_enabled = getattr(self._config, "retrieval_fusion_enabled", True)
        if _fusion_enabled and ranked_lists:
            from neural_memory.engine.retrieval_fusion import (
                FusionWeights,
                fuse_scores,
                select_weights,
            )

            # Build neuron-level scores per channel from ranked_lists
            _semantic_neuron_scores: dict[str, float] = {}
            _lexical_neuron_scores: dict[str, float] = {}
            for rlist in ranked_lists:
                for anchor in rlist:
                    # Use raw score if available, otherwise 1/(rank) as proxy
                    score = anchor.score if anchor.score > 0 else 1.0 / anchor.rank
                    if anchor.retriever == "embedding":
                        prev = _semantic_neuron_scores.get(anchor.neuron_id, 0.0)
                        _semantic_neuron_scores[anchor.neuron_id] = max(prev, score)
                    elif anchor.retriever in ("text_relevance", "keyword", "fuzzy"):
                        prev = _lexical_neuron_scores.get(anchor.neuron_id, 0.0)
                        _lexical_neuron_scores[anchor.neuron_id] = max(prev, score)

            # Build per-fiber channel scores (max neuron score per fiber per channel)
            _graph_fiber: dict[str, float] = {}
            _semantic_fiber: dict[str, float] = {}
            _lexical_fiber: dict[str, float] = {}
            for fiber in fibers:
                fid = fiber.id
                # Graph: max activation level of fiber's neurons
                act_levels = [
                    activations[nid].activation_level
                    for nid in fiber.neuron_ids
                    if nid in activations
                ]
                if act_levels:
                    _graph_fiber[fid] = max(act_levels)
                # Semantic: max embedding score of fiber's neurons
                sem_levels = [
                    _semantic_neuron_scores[nid]
                    for nid in fiber.neuron_ids
                    if nid in _semantic_neuron_scores
                ]
                if sem_levels:
                    _semantic_fiber[fid] = max(sem_levels)
                # Lexical: max keyword/BM25 score of fiber's neurons
                lex_levels = [
                    _lexical_neuron_scores[nid]
                    for nid in fiber.neuron_ids
                    if nid in _lexical_neuron_scores
                ]
                if lex_levels:
                    _lexical_fiber[fid] = max(lex_levels)

            # Select weights based on query intent or config
            _cfg_weights = dict(self._config.retrieval_fusion_weights)
            _weights = FusionWeights(
                graph=_cfg_weights.get("graph", 0.5),
                semantic=_cfg_weights.get("semantic", 0.3),
                lexical=_cfg_weights.get("lexical", 0.2),
            )
            if query_intent:
                _weights = select_weights(query_intent)

            fusion_results = fuse_scores(_graph_fiber, _semantic_fiber, _lexical_fiber, _weights)
            _fusion_scores = {r.fiber_id: r.fused_score for r in fusion_results}

        def _fiber_score(fiber: Fiber) -> float:
            # --- Base quality: salience * recency * conductivity ---
            recency = 0.5
            if fiber.last_conducted:
                hours_ago = (_now - fiber.last_conducted).total_seconds() / 3600
                recency = max(0.1, 1.0 / (1.0 + math.exp((hours_ago - halflife) / (halflife / 2))))

            base_score = fiber.salience * recency * fiber.conductivity

            # Creation-age freshness penalty (opt-in via freshness_weight > 0)
            if fw > 0.0 and fiber.created_at:
                from neural_memory.safety.freshness import evaluate_freshness

                age_result = evaluate_freshness(fiber.created_at)
                base_score *= (1.0 - fw) + fw * age_result.score

            # --- Activation relevance: how well does this fiber match the query? ---
            # When fusion is enabled, use fused tri-modal score as activation signal
            fused = _fusion_scores.get(fiber.id) if _fusion_scores else None
            if fused is not None:
                activation_signal = max(0.05, fused)
            else:
                activated = [nid for nid in fiber.neuron_ids if nid in activations]
                if activated:
                    coverage = len(activated) / max(len(fiber.neuron_ids), 1)
                    max_act = max(activations[nid].activation_level for nid in activated)
                    mean_act = sum(activations[nid].activation_level for nid in activated) / len(
                        activated
                    )
                    activation_signal = max_act * 0.5 + coverage * 0.3 + mean_act * 0.2
                    activation_signal = max(0.05, activation_signal)
                else:
                    activation_signal = 0.05

            # --- Stage bonus: semantic memories are more consolidated/reliable ---
            stage = getattr(fiber, "stage", None) or (fiber.metadata or {}).get("_stage")
            stage_multiplier = 1.1 if stage == "semantic" else 1.0

            score = base_score * activation_signal * stage_multiplier

            # --- Tag-aware scoring boost ---
            fiber_tags = set(fiber.metadata.get("tags", [])) if fiber.metadata else set()
            if tags and tag_boost > 0:
                if fiber_tags:
                    tag_overlap = len(tags & fiber_tags)
                    if tag_overlap > 0:
                        score += tag_boost * min(tag_overlap, 3) / 3  # cap at 3 matching tags
                    else:
                        score -= tag_boost * 0.5  # mild penalty for zero overlap

            # --- T1.2: Topic affinity from session EMA ---
            if _session_topics and _topic_affinity_boost > 0:
                # Use fiber.tags property (auto_tags | agent_tags) + metadata tags
                all_fiber_tags = fiber.tags | fiber_tags
                if all_fiber_tags:
                    topic_overlap = len(_session_topics & all_fiber_tags)
                    if topic_overlap > 0:
                        # Dice-style: normalize by the smaller set to avoid
                        # penalizing well-tagged fibers (review fix M1)
                        denom = min(len(_session_topics), len(all_fiber_tags))
                        affinity = topic_overlap / max(denom, 1)
                        score += affinity * _topic_affinity_boost

            # --- Goal-directed recall: proximity to active goals ---
            # Compound with prediction error: surprise near goals amplifies boost
            if _goal_proximity and _goal_proximity_boost > 0:
                goal_neurons = [nid for nid in fiber.neuron_ids if nid in _goal_proximity]
                if goal_neurons:
                    max_prox = max(_goal_proximity[nid] for nid in goal_neurons)
                    goal_boost = max_prox * _goal_proximity_boost
                    _surprise = (fiber.metadata or {}).get("_surprise_bonus", 0.0)
                    if isinstance(_surprise, (int, float)) and _surprise > 0:
                        goal_boost *= 1.0 + float(_surprise) * 0.3
                    score += goal_boost

            # --- Anti-redundancy: penalize previously surfaced fibers ---
            if _session_state and _anti_redundancy > 0:
                if _session_state.is_surfaced(fiber.id):
                    score *= _anti_redundancy

            # --- T1.5: Recent-access boost (multiplicative, review fix M3) ---
            if _recent_boost > 0 and fiber.last_conducted:
                hours_since = (_now - fiber.last_conducted).total_seconds() / 3600
                if hours_since <= _recent_window_hrs:
                    score *= 1.0 + _recent_boost

            # --- Arousal boost: emotionally charged memories are more memorable ---
            fiber_meta = fiber.metadata or {}
            arousal = fiber_meta.get("_arousal", 0.0)
            if isinstance(arousal, (int, float)) and arousal > 0.0:
                score *= 1.0 + float(arousal) * 0.2  # up to 20% boost at max arousal

            # --- T4.2: Stale penalty for outdated version references ---
            if fiber_meta.get("_stale"):
                score *= 0.8  # -20% penalty for outdated version references

            # --- Preference-aware boost: preference fibers rank higher for preference queries ---
            if _is_pref_query and _preference_boost > 1.0:
                if "preference" in fiber_tags:
                    score *= _preference_boost
                # Domain keyword overlap: additive boost for matching domains
                if _preference_domain_boost > 0 and query_tokens:
                    pref_domain = fiber_meta.get("_preference_domain")
                    if isinstance(pref_domain, list) and pref_domain:
                        domain_set = {d.lower() for d in pref_domain}
                        domain_overlap = len(query_tokens & domain_set)
                        if domain_overlap > 0:
                            score += _preference_domain_boost * min(domain_overlap, 3) / 3

            # --- Temporal event anchor boost: fibers mentioning query events rank higher ---
            if _event_anchors and _event_anchor_boost > 0:
                fiber_content_lower = (fiber.summary or "").lower()
                if not fiber_content_lower:
                    # Fall back to anchor neuron content via metadata
                    fiber_content_lower = str(fiber_meta.get("_content", "")).lower()
                anchor_hits = sum(1 for a in _event_anchors if a in fiber_content_lower)
                if anchor_hits > 0:
                    score += _event_anchor_boost * min(anchor_hits, 3) / 3

            # --- Role-aware scoring: boost fibers matching the query's role target ---
            if _role_target and _role_match_boost > 1.0:
                # Check fiber's role tag (set during benchmark ingest as "role:user"/"role:assistant")
                fiber_role = None
                if f"role:{_role_target}" in fiber_tags:
                    fiber_role = _role_target
                elif "role:assistant" in fiber_tags:
                    fiber_role = "assistant"
                elif "role:user" in fiber_tags:
                    fiber_role = "user"

                if fiber_role == _role_target:
                    score *= _role_match_boost
                elif fiber_role is not None:
                    score *= _role_mismatch_penalty

            # --- Column fiber boost: complete episodic traces rank higher ---
            if fiber_meta.get("_column"):
                score *= 1.3

            # --- Context-dependent retrieval: match encoding vs query context ---
            if getattr(self._config, "context_retrieval_enabled", True):
                stored_fp = fiber_meta.get("_context_fingerprint")
                if stored_fp and isinstance(stored_fp, dict) and query_tokens:
                    from neural_memory.engine.context_retrieval import (
                        ContextFingerprint,
                        context_match_score,
                    )

                    enc_ctx = ContextFingerprint.from_dict(stored_fp)
                    # Use session's top topic as project context for matching
                    _ret_project = ""
                    if session_state is not None:
                        top = session_state.get_topic_weights(limit=1)
                        if top:
                            _ret_project = next(iter(top))
                    ret_ctx = ContextFingerprint(
                        project_name=_ret_project,
                        dominant_topics=tuple(sorted(query_tokens)[:10]),
                    )
                    ctx_mult = context_match_score(enc_ctx, ret_ctx)
                    score *= ctx_mult

            # --- Adaptive instruction boost ---
            if fiber_meta.get("memory_type") == "instruction" or fiber_meta.get("type") in (
                "instruction",
                "workflow",
            ):
                exec_count = fiber_meta.get("execution_count", 0)
                success_rate = fiber_meta.get("success_rate")
                if exec_count > 0 and success_rate is not None:
                    confidence = min(1.0, exec_count / 10.0)
                    # boost is signed: positive for high success, negative for low
                    instruction_boost = (float(success_rate) - 0.5) * confidence * 0.3
                    score = max(0.0, score + instruction_boost)

                # Trigger pattern matching boost
                if query_tokens:
                    triggers = set(fiber_meta.get("trigger_patterns", []))
                    if triggers:
                        overlap = len(query_tokens & triggers) / max(len(triggers), 1)
                        if overlap > 0.3:
                            score += overlap * 0.2

            return score

        # Score all fibers and cache scores
        scored: list[tuple[float, Fiber]] = [(_fiber_score(f), f) for f in fibers]
        scored.sort(key=lambda x: x[0], reverse=True)
        score_cache: dict[str, float] = {f.id: s for s, f in scored}

        # --- T1.1 + T1.3: MMR diversity + SimHash dedup greedy selection ---
        # Instead of naive top-10, select greedily: skip fibers too similar to
        # already-selected ones (neuron overlap OR SimHash near-duplicate).
        from neural_memory.utils.simhash import is_near_duplicate

        overlap_threshold = self._config.diversity_overlap_threshold
        penalty_factor = self._config.diversity_penalty_factor

        # Batch-fetch anchor neurons for SimHash (T1.3)
        anchor_ids = list(dict.fromkeys(f.anchor_neuron_id for _, f in scored[:30]))
        anchor_neurons = await self._storage.get_neurons_batch(anchor_ids) if anchor_ids else {}

        selected: list[Fiber] = []
        selected_neuron_sets: list[set[str]] = []
        selected_hashes: list[int] = []

        # Stratum-aware diversity: track lifecycle stage + schema cluster counts
        from collections import Counter as _Counter

        stratum_counts: _Counter[str] = _Counter()
        schema_counts: _Counter[str] = _Counter()
        abstraction_counts: _Counter[str] = _Counter()
        _stratum_cap = getattr(self._config, "stratum_diversity_cap", 0.4)
        _target_count = 10

        for raw_score, fiber in scored:
            if len(selected) >= _target_count:
                break

            # T1.3: SimHash dedup — skip near-duplicate content
            anchor_neuron = anchor_neurons.get(fiber.anchor_neuron_id)
            fiber_hash = anchor_neuron.content_hash if anchor_neuron else 0
            if fiber_hash != 0 and any(
                h != 0 and is_near_duplicate(fiber_hash, h) for h in selected_hashes
            ):
                continue

            # T1.1: MMR diversity — penalize high neuron overlap
            if selected_neuron_sets:
                max_overlap = 0.0
                for sel_neurons in selected_neuron_sets:
                    if not sel_neurons:
                        continue
                    intersection = len(fiber.neuron_ids & sel_neurons)
                    union = len(fiber.neuron_ids | sel_neurons)
                    if union > 0:
                        overlap = intersection / union
                        max_overlap = max(max_overlap, overlap)
                if max_overlap > overlap_threshold:
                    # Apply penalty — fiber may still be selected if score is high enough
                    penalized_score = raw_score * (1.0 - max_overlap * penalty_factor)
                    # Skip if penalized score is below 50% of the lowest selected score
                    lowest_selected = score_cache.get(selected[-1].id, 0.0)
                    if penalized_score < lowest_selected * 0.5:
                        continue

            # Stratum-aware diversity: cap results per lifecycle stage
            fiber_meta = fiber.metadata or {}
            stratum = getattr(fiber, "stage", None) or fiber_meta.get("_stage") or "episodic"
            max_per_stratum = max(1, int(_target_count * _stratum_cap))
            if stratum_counts[stratum] >= max_per_stratum and len(selected) >= 3:
                # Allow first 3 selections unconstrained, then enforce cap
                continue

            # Schema-cluster diversity: cap fibers from same schema (when enabled)
            _schema_id = fiber_meta.get("_schema_id")
            if _schema_id and len(selected) >= 3:
                if schema_counts[_schema_id] >= max_per_stratum:
                    continue

            # Abstraction-cluster diversity (v4.52): cap fibers anchored on same
            # abstraction-induced CONCEPT neuron, or fibers carrying `_abstract_neuron_id`
            # from MERGE consolidation. Prevents one super-abstract from dominating top-K.
            _abstraction_id: str | None = None
            if anchor_neuron is not None and (anchor_neuron.metadata or {}).get(
                "_abstraction_induced"
            ):
                _abstraction_id = anchor_neuron.id
            elif fiber_meta.get("_abstract_neuron_id"):
                _abstraction_id = str(fiber_meta["_abstract_neuron_id"])
            if _abstraction_id and len(selected) >= 3:
                if abstraction_counts[_abstraction_id] >= max_per_stratum:
                    continue

            selected.append(fiber)
            selected_neuron_sets.append(set(fiber.neuron_ids))
            selected_hashes.append(fiber_hash)
            stratum_counts[stratum] += 1
            if _schema_id:
                schema_counts[_schema_id] += 1
            if _abstraction_id:
                abstraction_counts[_abstraction_id] += 1

        return selected

    async def query_with_stimulus(
        self,
        stimulus: Stimulus,
        depth: DepthLevel | None = None,
        max_tokens: int | None = None,
        as_of: datetime | None = None,
        simhash_threshold: int | None = None,
    ) -> RetrievalResult:
        """
        Execute retrieval with a pre-parsed stimulus.

        Useful when you want to control the parsing or reuse a stimulus.
        """
        return await self.query(
            stimulus.raw_query,
            depth=depth,
            max_tokens=max_tokens,
            as_of=as_of,
            simhash_threshold=simhash_threshold,
        )
