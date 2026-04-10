"""Synapse data structures - connections between neurons."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from neural_memory.utils.timeutils import ensure_naive_utc, utcnow


class SynapseType(StrEnum):
    """Types of synaptic connections between neurons."""

    # Temporal relationships
    HAPPENED_AT = "happened_at"  # Event -> Time
    BEFORE = "before"  # Event A -> Event B (A happened before B)
    AFTER = "after"  # Event A -> Event B (A happened after B)
    DURING = "during"  # Event -> Period

    # Spatial relationships
    AT_LOCATION = "at_location"  # Event/Entity -> Place
    CONTAINS = "contains"  # Place -> Entity
    NEAR = "near"  # Place -> Place

    # Causal relationships
    CAUSED_BY = "caused_by"  # Effect -> Cause
    LEADS_TO = "leads_to"  # Cause -> Effect
    ENABLES = "enables"  # Condition -> Action
    PREVENTS = "prevents"  # Blocker -> Action

    # Associative relationships
    CO_OCCURS = "co_occurs"  # Entity -> Entity (appear together)
    RELATED_TO = "related_to"  # General association
    SIMILAR_TO = "similar_to"  # Semantic similarity

    # Semantic relationships
    IS_A = "is_a"  # Instance -> Category
    HAS_PROPERTY = "has_property"  # Entity -> Property
    INVOLVES = "involves"  # Event -> Entity

    # Emotional relationships
    FELT = "felt"  # Event -> Emotion
    EVOKES = "evokes"  # Stimulus -> Emotion

    # Conflict relationships
    CONTRADICTS = "contradicts"  # Memory A contradicts Memory B
    RESOLVED_BY = "resolved_by"  # Fix/fact that resolved an error

    # Tool relationships
    EFFECTIVE_FOR = "effective_for"  # Tool -> Task/Concept (tool is effective for task)
    USED_WITH = "used_with"  # Tool -> Tool (tools used together in same context)

    # Deduplication relationships
    ALIAS = "alias"  # New anchor -> Existing anchor (dedup reuse)

    # Cognitive layer — evidence relationships
    EVIDENCE_FOR = "evidence_for"  # Observation -> Hypothesis (supports it)
    EVIDENCE_AGAINST = "evidence_against"  # Observation -> Hypothesis (weakens it)

    # Cognitive layer — prediction relationships
    PREDICTED = "predicted"  # Prediction -> Hypothesis (derived from belief)
    VERIFIED_BY = "verified_by"  # Prediction -> Observation (outcome confirmed it)
    FALSIFIED_BY = "falsified_by"  # Prediction -> Observation (outcome disproved it)

    # Domain / Structured data relationships
    HAS_VALUE = "has_value"  # Cell -> Value neuron (structured data)
    MEASURED_AT = "measured_at"  # Metric -> FiscalPeriod (financial temporal)
    REGULATES = "regulates"  # Regulation -> Entity (legal scope)
    IN_ROW = "in_row"  # Cell -> Row header (table structure)
    IN_COLUMN = "in_column"  # Cell -> Column header (table structure)

    # Source tracking
    SOURCE_OF = "source_of"  # Source -> Neuron (provenance link)

    # Audit trail
    STORED_BY = "stored_by"  # Neuron -> Agent/User (who stored it)
    VERIFIED_AT = "verified_at"  # Neuron -> Verifier (verification event)
    APPROVED_BY = "approved_by"  # Neuron -> Approver (approval event)

    # Cognitive layer — schema relationships
    SUPERSEDES = "supersedes"  # Schema_v2 -> Schema_v1 (model evolution)
    DERIVED_FROM = "derived_from"  # Hypothesis/Prediction -> Schema (reasoning origin)

    # Decision intelligence — evolution tracking
    EVOLVES_FROM = "evolves_from"  # New decision -> Prior decision (same domain)

    # Goal hierarchy
    SUBGOAL_OF = "subgoal_of"  # Child goal -> Parent goal (decomposition)

    # Code-semantic relationships
    IMPORTS = "imports"  # Module -> Module (import dependency)
    CALLS = "calls"  # Function -> Function (invocation)
    DEPENDS_ON = "depends_on"  # Component -> Dependency (runtime/build)
    INHERITS = "inherits"  # Class -> Parent class (inheritance chain)
    IMPLEMENTS = "implements"  # Class -> Interface/Protocol
    DEFINED_IN = "defined_in"  # Symbol -> Module/File (definition location)
    RAISES = "raises"  # Function -> Exception (error contract)


class SynapseRole(StrEnum):
    """Causal role of a synapse — determines recall behavior.

    Instead of special-casing each synapse type during recall, classify
    all types into causal roles. One post-activation pass handles all roles.
    """

    SUPERSESSION = "supersession"  # old→new: demote source, boost target
    REINFORCEMENT = "reinforcement"  # evidence strengthens belief
    WEAKENING = "weakening"  # counter-evidence weakens belief
    SEQUENTIAL = "sequential"  # step N primes step N+1
    STRUCTURAL = "structural"  # context/hierarchy — standard traversal
    LATERAL = "lateral"  # co-occurrence, bidirectional
    PASSIVE = "passive"  # audit trail — skip during recall


# Mapping of every synapse type to its causal role.
# This drives recall behavior: _apply_causal_semantics() uses roles,
# not individual type checks.
SYNAPSE_ROLES: dict[SynapseType, SynapseRole] = {
    # ── Supersession: recall prefers target over source ───────────
    SynapseType.RESOLVED_BY: SynapseRole.SUPERSESSION,
    SynapseType.SUPERSEDES: SynapseRole.SUPERSESSION,
    SynapseType.EVOLVES_FROM: SynapseRole.SUPERSESSION,
    SynapseType.FALSIFIED_BY: SynapseRole.SUPERSESSION,
    # ── Reinforcement: evidence boosts hypothesis ─────────────────
    SynapseType.EVIDENCE_FOR: SynapseRole.REINFORCEMENT,
    SynapseType.VERIFIED_BY: SynapseRole.REINFORCEMENT,
    SynapseType.EFFECTIVE_FOR: SynapseRole.REINFORCEMENT,
    # ── Weakening: counter-evidence demotes hypothesis ────────────
    SynapseType.EVIDENCE_AGAINST: SynapseRole.WEAKENING,
    SynapseType.CONTRADICTS: SynapseRole.WEAKENING,
    SynapseType.PREVENTS: SynapseRole.WEAKENING,
    # ── Sequential: step N primes step N+1 ────────────────────────
    SynapseType.BEFORE: SynapseRole.SEQUENTIAL,
    SynapseType.AFTER: SynapseRole.SEQUENTIAL,
    SynapseType.LEADS_TO: SynapseRole.SEQUENTIAL,
    SynapseType.CALLS: SynapseRole.SEQUENTIAL,
    SynapseType.ENABLES: SynapseRole.SEQUENTIAL,
    SynapseType.CAUSED_BY: SynapseRole.SEQUENTIAL,
    # ── Structural: context/hierarchy — standard weight traversal ─
    SynapseType.IS_A: SynapseRole.STRUCTURAL,
    SynapseType.HAS_PROPERTY: SynapseRole.STRUCTURAL,
    SynapseType.INVOLVES: SynapseRole.STRUCTURAL,
    SynapseType.CONTAINS: SynapseRole.STRUCTURAL,
    SynapseType.AT_LOCATION: SynapseRole.STRUCTURAL,
    SynapseType.NEAR: SynapseRole.STRUCTURAL,
    SynapseType.INHERITS: SynapseRole.STRUCTURAL,
    SynapseType.IMPLEMENTS: SynapseRole.STRUCTURAL,
    SynapseType.DEFINED_IN: SynapseRole.STRUCTURAL,
    SynapseType.IMPORTS: SynapseRole.STRUCTURAL,
    SynapseType.DEPENDS_ON: SynapseRole.STRUCTURAL,
    SynapseType.RAISES: SynapseRole.STRUCTURAL,
    SynapseType.HAS_VALUE: SynapseRole.STRUCTURAL,
    SynapseType.IN_ROW: SynapseRole.STRUCTURAL,
    SynapseType.IN_COLUMN: SynapseRole.STRUCTURAL,
    SynapseType.SOURCE_OF: SynapseRole.STRUCTURAL,
    SynapseType.PREDICTED: SynapseRole.STRUCTURAL,
    SynapseType.DERIVED_FROM: SynapseRole.STRUCTURAL,
    SynapseType.REGULATES: SynapseRole.STRUCTURAL,
    SynapseType.SUBGOAL_OF: SynapseRole.STRUCTURAL,
    # ── Lateral: bidirectional co-occurrence ───────────────────────
    SynapseType.CO_OCCURS: SynapseRole.LATERAL,
    SynapseType.RELATED_TO: SynapseRole.LATERAL,
    SynapseType.SIMILAR_TO: SynapseRole.LATERAL,
    SynapseType.USED_WITH: SynapseRole.LATERAL,
    # ── Passive: audit/metadata — skip during recall ──────────────
    SynapseType.STORED_BY: SynapseRole.PASSIVE,
    SynapseType.VERIFIED_AT: SynapseRole.PASSIVE,
    SynapseType.APPROVED_BY: SynapseRole.PASSIVE,
    SynapseType.ALIAS: SynapseRole.PASSIVE,
    SynapseType.FELT: SynapseRole.PASSIVE,
    SynapseType.EVOKES: SynapseRole.PASSIVE,
    SynapseType.HAPPENED_AT: SynapseRole.PASSIVE,
    SynapseType.DURING: SynapseRole.PASSIVE,
    SynapseType.MEASURED_AT: SynapseRole.PASSIVE,
}

# Convenience frozensets for fast membership checks
SUPERSESSION_TYPES: frozenset[SynapseType] = frozenset(
    st for st, role in SYNAPSE_ROLES.items() if role == SynapseRole.SUPERSESSION
)
REINFORCEMENT_TYPES: frozenset[SynapseType] = frozenset(
    st for st, role in SYNAPSE_ROLES.items() if role == SynapseRole.REINFORCEMENT
)
WEAKENING_TYPES: frozenset[SynapseType] = frozenset(
    st for st, role in SYNAPSE_ROLES.items() if role == SynapseRole.WEAKENING
)
SEQUENTIAL_TYPES: frozenset[SynapseType] = frozenset(
    st for st, role in SYNAPSE_ROLES.items() if role == SynapseRole.SEQUENTIAL
)
PASSIVE_TYPES: frozenset[SynapseType] = frozenset(
    st for st, role in SYNAPSE_ROLES.items() if role == SynapseRole.PASSIVE
)
# Active roles = types that should be fetched for causal processing
ACTIVE_ROLE_TYPES: frozenset[SynapseType] = (
    SUPERSESSION_TYPES | REINFORCEMENT_TYPES | WEAKENING_TYPES | SEQUENTIAL_TYPES
)

# Supersession types where source=NEW, target=OLD (inverted directionality).
# For these, the SOURCE is the latest version and should be BOOSTED (not demoted).
# SUPERSEDES: "new_schema SUPERSEDES old_schema" → source=new, target=old
# EVOLVES_FROM: "new_decision EVOLVES_FROM old_decision" → source=new, target=old
SUPERSESSION_SOURCE_IS_NEWER: frozenset[SynapseType] = frozenset(
    {
        SynapseType.SUPERSEDES,
        SynapseType.EVOLVES_FROM,
    }
)


class Direction(StrEnum):
    """Direction of synapse connection."""

    UNIDIRECTIONAL = "uni"  # One-way: source -> target
    BIDIRECTIONAL = "bi"  # Two-way: source <-> target


# Synapse types that are typically bidirectional
BIDIRECTIONAL_TYPES: frozenset[SynapseType] = frozenset(
    {
        SynapseType.CO_OCCURS,
        SynapseType.RELATED_TO,
        SynapseType.SIMILAR_TO,
        SynapseType.NEAR,
        SynapseType.USED_WITH,
    }
)

# Synapse types with inverse relationships
INVERSE_TYPES: dict[SynapseType, SynapseType] = {
    SynapseType.BEFORE: SynapseType.AFTER,
    SynapseType.AFTER: SynapseType.BEFORE,
    SynapseType.CAUSED_BY: SynapseType.LEADS_TO,
    SynapseType.LEADS_TO: SynapseType.CAUSED_BY,
    SynapseType.CONTAINS: SynapseType.AT_LOCATION,
    SynapseType.AT_LOCATION: SynapseType.CONTAINS,
    # Note: VERIFIED_BY and FALSIFIED_BY are NOT inverses — they are
    # alternative truth-value edges on the same direction (Prediction → Observation).
    # SUPERSEDES and DERIVED_FROM are intentionally unidirectional with no inverse.
    # Code-semantic inverses
    SynapseType.IMPORTS: SynapseType.DEPENDS_ON,  # A imports B ↔ B is depended on by A
    SynapseType.DEPENDS_ON: SynapseType.IMPORTS,
    SynapseType.INHERITS: SynapseType.IS_A,  # A inherits B ↔ B is a parent of A
    SynapseType.IS_A: SynapseType.INHERITS,
}


@dataclass(frozen=True)
class Synapse:
    """
    A synapse represents a connection between two neurons.

    Synapses have semantic meaning (type) and strength (weight).
    They can be reinforced through use or decay over time.

    Attributes:
        id: Unique identifier
        source_id: ID of the source neuron
        target_id: ID of the target neuron
        type: The semantic type of this connection
        weight: Connection strength (0.0 - 1.0)
        direction: Whether connection is uni or bidirectional
        metadata: Additional connection-specific data
        reinforced_count: How many times this connection was reinforced
        last_activated: When this synapse was last used
        created_at: When this synapse was created
    """

    id: str
    source_id: str
    target_id: str
    type: SynapseType
    weight: float = 0.5
    direction: Direction = Direction.UNIDIRECTIONAL
    metadata: dict[str, Any] = field(default_factory=dict)
    reinforced_count: int = 0
    last_activated: datetime | None = None
    created_at: datetime = field(default_factory=utcnow)

    @classmethod
    def create(
        cls,
        source_id: str,
        target_id: str,
        type: SynapseType,
        weight: float = 0.5,
        direction: Direction | None = None,
        metadata: dict[str, Any] | None = None,
        synapse_id: str | None = None,
    ) -> Synapse:
        """
        Factory method to create a new Synapse.

        Args:
            source_id: ID of source neuron
            target_id: ID of target neuron
            type: Synapse type
            weight: Initial weight (default 0.5)
            direction: Connection direction (auto-detected if None)
            metadata: Optional metadata
            synapse_id: Optional explicit ID

        Returns:
            A new Synapse instance
        """
        # Auto-detect direction based on type
        if direction is None:
            direction = (
                Direction.BIDIRECTIONAL if type in BIDIRECTIONAL_TYPES else Direction.UNIDIRECTIONAL
            )

        return cls(
            id=synapse_id or str(uuid4()),
            source_id=source_id,
            target_id=target_id,
            type=type,
            weight=max(0.0, min(1.0, weight)),
            direction=direction,
            metadata=metadata or {},
            created_at=utcnow(),
        )

    def reinforce(
        self,
        delta: float = 0.05,
        pre_activation: float | None = None,
        post_activation: float | None = None,
        now: datetime | None = None,
    ) -> Synapse:
        """
        Create a new Synapse with reinforced weight.

        When pre/post activation levels are provided, uses the formal
        Hebbian learning rule: Δw = η_eff * pre * post * (w_max - w).
        Otherwise falls back to direct delta addition (backward compatible).

        Args:
            delta: Amount to increase weight by (used as learning rate for Hebbian)
            pre_activation: Pre-synaptic neuron activation level [0, 1]
            post_activation: Post-synaptic neuron activation level [0, 1]
            now: Reference time (default: utcnow)

        Returns:
            New Synapse with increased weight (capped at 1.0)
        """
        now = now or utcnow()

        if pre_activation is not None and post_activation is not None:
            # Validate activation levels are in bounds [0, 1]
            pre_activation = max(0.0, min(1.0, pre_activation))
            post_activation = max(0.0, min(1.0, post_activation))

            from neural_memory.engine.learning_rule import LearningConfig, hebbian_update

            config = LearningConfig(learning_rate=delta)
            update = hebbian_update(
                current_weight=self.weight,
                pre_activation=pre_activation,
                post_activation=post_activation,
                reinforced_count=self.reinforced_count,
                config=config,
            )
            new_weight = update.new_weight
        else:
            # Backward-compatible: direct delta addition
            new_weight = min(1.0, self.weight + delta)

        return Synapse(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            type=self.type,
            weight=new_weight,
            direction=self.direction,
            metadata=self.metadata,
            reinforced_count=self.reinforced_count + 1,
            last_activated=now,
            created_at=self.created_at,
        )

    def decay(self, factor: float = 0.95) -> Synapse:
        """
        Create a new Synapse with decayed weight.

        Args:
            factor: Decay multiplier (0.0 - 1.0)

        Returns:
            New Synapse with decreased weight
        """
        factor = max(0.0, min(1.0, factor))
        return Synapse(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            type=self.type,
            weight=self.weight * factor,
            direction=self.direction,
            metadata=self.metadata,
            reinforced_count=self.reinforced_count,
            last_activated=self.last_activated,
            created_at=self.created_at,
        )

    def time_decay(self, reference_time: datetime | None = None) -> Synapse:
        """Decay weight based on time since last activation.

        Uses sigmoid with reinforcement-modulated half-life: reinforced
        connections decay slower and have a higher floor.

        Unreinforced (count=0): ~0.98 at 1d, ~0.50 at 60d, floor 0.30
        Reinforced 5x: half-life 210d, floor 0.55
        Reinforced 10x: half-life 360d, floor 0.80

        Args:
            reference_time: Reference time for age calculation (default: now)

        Returns:
            New Synapse with time-decayed weight
        """
        if reference_time is None:
            reference_time = utcnow()
        else:
            reference_time = ensure_naive_utc(reference_time)

        if self.last_activated:
            hours_since = (reference_time - self.last_activated).total_seconds() / 3600
        else:
            hours_since = (reference_time - self.created_at).total_seconds() / 3600

        hours_since = max(0, hours_since)

        # Adaptive half-life: reinforced connections last longer
        base_half_life = 1440.0  # 60 days in hours
        reinforcement_factor = 1.0 + self.reinforced_count * 0.5
        effective_half_life = base_half_life * reinforcement_factor
        spread = effective_half_life / 2.0

        # Sigmoid decay centered at effective half-life
        exponent = (hours_since - effective_half_life) / spread
        exponent = max(-100.0, min(100.0, exponent))
        factor = 1.0 / (1.0 + math.exp(exponent))

        # Adaptive floor: reinforced connections have higher minimum
        floor = 0.3 + min(0.5, self.reinforced_count * 0.05)
        factor = max(floor, factor)

        new_weight = self.weight * factor
        return Synapse(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            type=self.type,
            weight=new_weight,
            direction=self.direction,
            metadata=self.metadata,
            reinforced_count=self.reinforced_count,
            last_activated=self.last_activated,
            created_at=self.created_at,
        )

    @property
    def is_bidirectional(self) -> bool:
        """Check if this synapse allows traversal in both directions."""
        return self.direction == Direction.BIDIRECTIONAL

    def get_inverse_type(self) -> SynapseType | None:
        """Get the inverse synapse type if one exists."""
        return INVERSE_TYPES.get(self.type)

    def connects(self, neuron_id: str) -> bool:
        """Check if this synapse connects to a given neuron."""
        return self.source_id == neuron_id or self.target_id == neuron_id

    def other_end(self, neuron_id: str) -> str | None:
        """Get the ID of the neuron at the other end of this synapse."""
        if self.source_id == neuron_id:
            return self.target_id
        if self.target_id == neuron_id:
            return self.source_id
        return None
