"""Brain container - the top-level memory structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from neural_memory.utils.timeutils import utcnow


@dataclass(frozen=True)
class BrainConfig:
    """
    Configuration for brain behavior.

    Attributes:
        decay_rate: Rate at which neuron activation decays (per day)
        reinforcement_delta: Amount to increase synapse weight on access
        activation_threshold: Minimum activation level to consider active
        max_spread_hops: Maximum hops in spreading activation
        max_context_tokens: Maximum tokens to include in context injection
        default_synapse_weight: Default weight for new synapses
    """

    decay_rate: float = 0.1
    reinforcement_delta: float = 0.05
    activation_threshold: float = 0.3
    max_spread_hops: int = 4
    max_context_tokens: int = 1500
    default_synapse_weight: float = 0.5
    hebbian_delta: float = 0.03
    hebbian_threshold: float = 0.5
    hebbian_initial_weight: float = 0.2
    consolidation_prune_threshold: float = 0.05
    prune_min_inactive_days: float = 7.0
    merge_overlap_threshold: float = 0.5
    sigmoid_steepness: float = 6.0
    default_firing_threshold: float = 0.3
    default_refractory_ms: float = 500.0
    lateral_inhibition_k: int = 10
    lateral_inhibition_factor: float = 0.3
    learning_rate: float = 0.05
    weight_normalization_budget: float = 5.0
    novelty_boost_max: float = 3.0
    novelty_decay_rate: float = 0.06
    co_activation_threshold: int = 3
    co_activation_window_days: int = 7
    max_inferences_per_run: int = 50
    emotional_decay_factor: float = 0.5
    emotional_weight_scale: float = 0.8
    sequential_window_seconds: float = 30.0
    dream_neuron_count: int = 5
    dream_decay_multiplier: float = 3.0
    habit_min_frequency: int = 3
    habit_suggestion_min_weight: float = 0.8
    habit_suggestion_min_count: int = 5
    embedding_enabled: bool = True
    embedding_provider: str = "sentence_transformer"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_similarity_threshold: float = 0.7
    embedding_activation_boost: float = 0.15
    freshness_weight: float = 0.15
    semantic_discovery_similarity_threshold: float = 0.7
    semantic_discovery_max_pairs: int = 100
    # Adaptive recall (Bayesian depth priors)
    adaptive_depth_enabled: bool = True
    adaptive_depth_epsilon: float = 0.05
    # Memory compression
    compression_enabled: bool = True
    compression_tier_thresholds: tuple[int, ...] = (7, 30, 90, 180)
    # Retrieval: Reciprocal Rank Fusion
    rrf_k: int = 60
    # Retrieval: Graph-based query expansion
    graph_expansion_enabled: bool = True
    graph_expansion_max: int = 10
    graph_expansion_min_weight: float = 0.3
    # Retrieval: Activation strategy
    activation_strategy: str = "classic"  # "ppr" | "classic" | "reflex" | "hybrid" | "auto"
    ppr_damping: float = 0.15
    ppr_iterations: int = 20
    ppr_epsilon: float = 1e-6
    # PPR: dampening factor for DREAM hub synapses (_hub=True) during push.
    # 1.0 = no dampening; 0.5 halves hub edge influence so synthesized hub
    # links don't hijack random walks at the expense of genuine edges.
    hub_edge_dampening: float = 0.5
    # Cascading retrieval: fiber summary tier + sufficiency gate
    fiber_summary_tier_enabled: bool = True
    sufficiency_threshold: float = 0.7
    # Lazy entity promotion: entities need N mentions to become neurons
    lazy_entity_enabled: bool = True
    lazy_entity_promotion_threshold: int = 2
    lazy_entity_prune_days: int = 90
    # Recall quality: recency sigmoid halflife (hours)
    recency_halflife_hours: float = 168.0  # 7 days (was hardcoded 72h)
    # Recall quality: tag-aware scoring boost
    tag_match_boost: float = 0.15
    # Prune: dead neuron minimum age (days) before auto-prune
    prune_dead_neuron_days: float = 14.0
    # Diminishing returns gate: stop spreading when new hops add little signal
    diminishing_returns_enabled: bool = True
    diminishing_returns_threshold: float = 0.15
    diminishing_returns_min_neurons: int = 2
    diminishing_returns_grace_hops: int = 1
    # Fidelity layers
    decay_floor: float = 0.05
    fidelity_enabled: bool = True
    fidelity_full_threshold: float = 0.6
    fidelity_summary_threshold: float = 0.3
    fidelity_essence_threshold: float = 0.1
    essence_generator: str = "extractive"  # "extractive" or "llm"
    # Fuzzy search (typo tolerance)
    fuzzy_search_enabled: bool = False
    fuzzy_search_max_distance: int = 2
    fuzzy_search_max_candidates: int = 50
    # IDF-weighted anchor selection
    idf_anchor_enabled: bool = False
    idf_anchor_min_limit: int = 1
    idf_anchor_max_limit: int = 5
    # SimHash pre-filter: exclude distant neurons before spreading activation
    simhash_prefilter_threshold: int = 0  # 0=disabled, 1-64=Hamming distance cutoff
    # Query expansion (synonym, abbreviation, cross-language)
    query_expansion_synonyms: bool = True
    query_expansion_abbreviations: bool = True
    query_expansion_cross_language: bool = True
    query_expansion_max_per_term: int = 5
    # Cross-encoder reranking (optional post-SA refinement)
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_overfetch_multiplier: int = 3
    reranker_blend_weight: float = 0.7  # Reranker weight (SA gets 1 - this)
    reranker_min_score: float = 0.15
    reranker_max_candidates: int = 30  # Safety cap on overfetch
    # Temporal binding (session-level auto-linking)
    temporal_binding_enabled: bool = True
    temporal_binding_window_seconds: float = 300.0  # 5-minute window
    # Arousal detection (emotional intensity for compression resistance)
    arousal_enabled: bool = True
    # Prediction error encoding (surprise signal boosts priority)
    prediction_error_enabled: bool = True
    # Retrieval reconsolidation (recalled memories absorb context)
    reconsolidation_enabled: bool = True
    reconsolidation_drift_threshold: float = 0.6
    # Context-dependent retrieval (project-scoped scoring)
    context_retrieval_enabled: bool = True
    # Hippocampal replay consolidation (LTP/LTD)
    replay_enabled: bool = True
    replay_ltp_factor: float = 1.1
    replay_ltd_factor: float = 0.98
    replay_window_hours: float = 24.0  # How far back to look for recent fibers
    replay_max_episodes: int = 20  # Max episodes per replay run
    # Working memory chunking (group retrieval output)
    chunking_enabled: bool = True
    max_chunks: int = 5
    # Schema assimilation (bottom-up knowledge organization)
    schema_assimilation_enabled: bool = False
    schema_min_cluster_size: int = 10
    # Familiarity fallback (dual-process: recollection vs familiarity)
    familiarity_fallback_enabled: bool = True
    familiarity_max_fibers: int = 5
    familiarity_confidence_cap: float = 0.4
    # Graph density scaling (homeostatic synaptic scaling for large graphs)
    graph_density_scaling_enabled: bool = True
    # Session cortical columns (episodic session-level retrieval)
    session_columns_enabled: bool = True
    # Interference forgetting (memory competition detection)
    interference_detection_enabled: bool = False
    fan_effect_threshold: int = 15
    # Precision recall (A8 Phase 1)
    recent_access_boost: float = 0.1
    recent_access_window_days: int = 7
    diversity_overlap_threshold: float = 0.6
    diversity_penalty_factor: float = 0.7
    topic_affinity_boost: float = 0.15
    # Goal-directed recall (prefrontal cortex top-down attention modulation)
    goal_proximity_boost: float = 0.25
    goal_max_hops: int = 3
    # Causal integration: auto-include causal chains + anti-redundancy
    causal_auto_include: bool = True
    causal_auto_include_max_hops: int = 2
    anti_redundancy_penalty: float = 0.3
    # Unified confidence scoring weights (sum should ≈ 1.0)
    confidence_weight_retrieval: float = 0.35
    confidence_weight_quality: float = 0.25
    confidence_weight_fidelity: float = 0.20
    confidence_weight_freshness: float = 0.20
    # Cascade staleness: propagate stale marks through causal graph
    cascade_staleness_enabled: bool = True
    # Stratum-aware MMR diversity: cap per lifecycle stage
    stratum_diversity_cap: float = 0.4  # max 40% of results from one stratum
    # Preference-aware retrieval (boost preference-establishing sessions)
    preference_detection_enabled: bool = True
    preference_boost: float = 1.5  # multiplier for preference-tagged fibers
    preference_domain_boost: float = 0.2  # additive boost for domain keyword overlap
    # Temporal query routing (event anchor boost for temporal questions)
    temporal_routing_enabled: bool = True
    temporal_event_anchor_boost: float = 0.3  # additive boost per event anchor match
    # Role-aware scoring (boost fibers matching query role target)
    role_aware_scoring_enabled: bool = True
    role_match_boost: float = 1.3  # multiplier for matching role
    role_mismatch_penalty: float = 0.9  # multiplier for mismatching role
    # Abstraction-level constraint (spreading activation gate)
    abstraction_constraint_enabled: bool = False
    abstraction_max_distance: int = 1
    # Reflex arc: always-on neurons injected into every recall
    max_reflexes: int = 20
    # Hybrid retrieval fusion (tri-modal: graph + semantic + lexical)
    retrieval_fusion_enabled: bool = True
    retrieval_fusion_weights: tuple[tuple[str, float], ...] = (
        ("graph", 0.5),
        ("semantic", 0.3),
        ("lexical", 0.2),
    )

    def with_updates(self, **kwargs: Any) -> BrainConfig:
        """Create a new config with updated values."""
        return dc_replace(self, **kwargs)


@dataclass(frozen=True)
class Brain:
    """
    A Brain is the top-level container for a memory system.

    It holds configuration, ownership, and statistics for a
    collection of neurons, synapses, and fibers.

    Attributes:
        id: Unique identifier
        name: Human-readable name
        config: Brain configuration settings
        owner_id: Optional owner identifier
        is_public: Whether this brain can be read by anyone
        shared_with: List of user IDs with access
        neuron_count: Number of neurons (computed)
        synapse_count: Number of synapses (computed)
        fiber_count: Number of fibers (computed)
        metadata: Additional brain-specific data
        created_at: When this brain was created
        updated_at: When this brain was last modified
    """

    id: str
    name: str
    config: BrainConfig = field(default_factory=BrainConfig)
    owner_id: str | None = None
    is_public: bool = False
    shared_with: list[str] = field(default_factory=list)
    neuron_count: int = 0
    synapse_count: int = 0
    fiber_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    @classmethod
    def create(
        cls,
        name: str,
        config: BrainConfig | None = None,
        owner_id: str | None = None,
        is_public: bool = False,
        brain_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Brain:
        """
        Factory method to create a new Brain.

        Args:
            name: Human-readable name
            config: Optional configuration (uses defaults if None)
            owner_id: Optional owner identifier
            is_public: Whether publicly accessible
            brain_id: Optional explicit ID
            metadata: Optional metadata

        Returns:
            A new Brain instance
        """
        return cls(
            id=brain_id or str(uuid4()),
            name=name,
            config=config or BrainConfig(),
            owner_id=owner_id,
            is_public=is_public,
            metadata=metadata or {},
            created_at=utcnow(),
            updated_at=utcnow(),
        )

    def share_with(self, user_id: str) -> Brain:
        """
        Create a new Brain shared with an additional user.

        Args:
            user_id: User ID to share with

        Returns:
            New Brain with updated shared_with list
        """
        if user_id in self.shared_with:
            return self

        return Brain(
            id=self.id,
            name=self.name,
            config=self.config,
            owner_id=self.owner_id,
            is_public=self.is_public,
            shared_with=[*self.shared_with, user_id],
            neuron_count=self.neuron_count,
            synapse_count=self.synapse_count,
            fiber_count=self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def unshare_with(self, user_id: str) -> Brain:
        """
        Create a new Brain with a user removed from sharing.

        Args:
            user_id: User ID to remove

        Returns:
            New Brain with updated shared_with list
        """
        return Brain(
            id=self.id,
            name=self.name,
            config=self.config,
            owner_id=self.owner_id,
            is_public=self.is_public,
            shared_with=[uid for uid in self.shared_with if uid != user_id],
            neuron_count=self.neuron_count,
            synapse_count=self.synapse_count,
            fiber_count=self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def make_public(self) -> Brain:
        """Create a new Brain that is publicly accessible."""
        return Brain(
            id=self.id,
            name=self.name,
            config=self.config,
            owner_id=self.owner_id,
            is_public=True,
            shared_with=self.shared_with,
            neuron_count=self.neuron_count,
            synapse_count=self.synapse_count,
            fiber_count=self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def make_private(self) -> Brain:
        """Create a new Brain that is private."""
        return Brain(
            id=self.id,
            name=self.name,
            config=self.config,
            owner_id=self.owner_id,
            is_public=False,
            shared_with=self.shared_with,
            neuron_count=self.neuron_count,
            synapse_count=self.synapse_count,
            fiber_count=self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def with_config(self, config: BrainConfig) -> Brain:
        """Create a new Brain with updated configuration."""
        return Brain(
            id=self.id,
            name=self.name,
            config=config,
            owner_id=self.owner_id,
            is_public=self.is_public,
            shared_with=self.shared_with,
            neuron_count=self.neuron_count,
            synapse_count=self.synapse_count,
            fiber_count=self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def with_stats(
        self,
        neuron_count: int | None = None,
        synapse_count: int | None = None,
        fiber_count: int | None = None,
    ) -> Brain:
        """Create a new Brain with updated statistics."""
        return Brain(
            id=self.id,
            name=self.name,
            config=self.config,
            owner_id=self.owner_id,
            is_public=self.is_public,
            shared_with=self.shared_with,
            neuron_count=neuron_count if neuron_count is not None else self.neuron_count,
            synapse_count=synapse_count if synapse_count is not None else self.synapse_count,
            fiber_count=fiber_count if fiber_count is not None else self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def can_access(self, user_id: str | None) -> bool:
        """
        Check if a user can access this brain.

        Args:
            user_id: User ID to check (None for anonymous)

        Returns:
            True if user has access
        """
        if self.is_public:
            return True
        if user_id is None:
            return False
        if self.owner_id == user_id:
            return True
        return user_id in self.shared_with

    def can_write(self, user_id: str | None) -> bool:
        """
        Check if a user can write to this brain.

        Args:
            user_id: User ID to check (None for anonymous)

        Returns:
            True if user has write access
        """
        if user_id is None:
            return False
        return self.owner_id == user_id


@dataclass(frozen=True)
class BrainSnapshot:
    """
    A serializable snapshot of a brain for export/import.

    Attributes:
        brain_id: ID of the original brain
        brain_name: Name of the brain
        exported_at: When this snapshot was created
        version: Schema version for compatibility
        neurons: List of serialized neurons
        synapses: List of serialized synapses
        fibers: List of serialized fibers
        config: Brain configuration
        metadata: Additional export metadata
    """

    brain_id: str
    brain_name: str
    exported_at: datetime
    version: str
    neurons: list[dict[str, Any]]
    synapses: list[dict[str, Any]]
    fibers: list[dict[str, Any]]
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
