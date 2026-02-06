"""Synapse data structures - connections between neurons."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


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
}


@dataclass
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
    created_at: datetime = field(default_factory=datetime.utcnow)

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
            created_at=datetime.utcnow(),
        )

    def reinforce(self, delta: float = 0.05) -> Synapse:
        """
        Create a new Synapse with reinforced weight.

        Includes dormancy bonus: synapses reactivated after long dormancy
        get an extra boost (re-learning effect).

        Args:
            delta: Amount to increase weight by

        Returns:
            New Synapse with increased weight (capped at 1.0)
        """
        dormancy_bonus = 0.0
        if self.last_activated:
            hours_dormant = (datetime.utcnow() - self.last_activated).total_seconds() / 3600
            if hours_dormant > 168:  # > 1 week dormant
                dormancy_bonus = min(0.1, 0.02 * math.log1p(hours_dormant / 168))

        effective_delta = delta + dormancy_bonus
        return Synapse(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            type=self.type,
            weight=min(1.0, self.weight + effective_delta),
            direction=self.direction,
            metadata=self.metadata,
            reinforced_count=self.reinforced_count + 1,
            last_activated=datetime.utcnow(),
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

        Uses sigmoid: recent synapses barely decay, old ones decay more.
        Synapses that were never activated decay fastest.

        ~0.98 at 1 day, ~0.90 at 7 days, ~0.70 at 30 days, ~0.50 at 60 days.

        Args:
            reference_time: Reference time for age calculation (default: now)

        Returns:
            New Synapse with time-decayed weight (floor at 30% of original)
        """
        if reference_time is None:
            reference_time = datetime.utcnow()

        if self.last_activated:
            hours_since = (reference_time - self.last_activated).total_seconds() / 3600
        else:
            hours_since = (reference_time - self.created_at).total_seconds() / 3600

        # Sigmoid decay: center at 1440h (60 days), spread 720h
        factor = 1.0 / (1.0 + math.exp((hours_since - 1440) / 720))
        factor = max(0.3, factor)  # Floor: never decay below 30% of original

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
