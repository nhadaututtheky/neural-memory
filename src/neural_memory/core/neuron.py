"""Neuron data structures - the basic units of memory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

from neural_memory.utils.timeutils import utcnow


class NeuronType(StrEnum):
    """Types of neurons in the memory system."""

    TIME = "time"  # Temporal markers: "3pm", "yesterday"
    SPATIAL = "spatial"  # Locations: "coffee shop", "office"
    ENTITY = "entity"  # Named entities: "Alice", "FastAPI"
    ACTION = "action"  # Verbs/actions: "discussed", "completed"
    STATE = "state"  # Emotional/mental states: "happy", "frustrated"
    CONCEPT = "concept"  # Abstract ideas: "API design", "authentication"
    SENSORY = "sensory"  # Sensory experiences: "loud", "bright"
    INTENT = "intent"  # Goals/intentions: "learn", "build"

    # Cognitive layer types
    HYPOTHESIS = "hypothesis"  # Evolving beliefs: "PostgreSQL is better for this project"
    PREDICTION = "prediction"  # Falsifiable claims: "API will fail at 1000 concurrent"
    SCHEMA = "schema"  # Mental model snapshots: versioned understanding of a domain


@dataclass(frozen=True)
class Neuron:
    """
    A neuron represents a single unit of memory.

    Neurons are immutable - they represent facts that don't change.
    The activation state is stored separately in NeuronState.

    Attributes:
        id: Unique identifier (UUID or content-hash)
        type: Category of information this neuron represents
        content: The raw value/text of this memory unit
        metadata: Type-specific additional information
        created_at: When this neuron was created
    """

    id: str
    type: NeuronType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: int = 0
    created_at: datetime = field(default_factory=utcnow)
    ephemeral: bool = False

    @property
    def grounded(self) -> bool:
        """Whether this neuron is a grounded truth (resists decay and conflicts)."""
        return bool(self.metadata.get("_grounded", False))

    @property
    def confidence(self) -> float:
        """Confidence level (0.0-1.0). Grounded neurons default to 1.0."""
        return float(self.metadata.get("_confidence", 0.5))

    # -- Goal neuron properties (metadata-backed, zero migration) --

    @property
    def goal_state(self) -> str | None:
        """Goal state: 'active', 'paused', or 'completed'. None if not a goal."""
        return self.metadata.get("_goal_state")

    @property
    def goal_priority(self) -> int:
        """Goal priority (1-10, default 5)."""
        return int(self.metadata.get("_goal_priority", 5))

    @property
    def goal_keywords(self) -> list[str]:
        """Keywords extracted from goal for EMA seeding."""
        kw = self.metadata.get("_goal_keywords")
        return list(kw) if isinstance(kw, (list, tuple)) else []

    @property
    def is_active_goal(self) -> bool:
        """Whether this neuron is an active goal."""
        return self.goal_state == "active"

    @property
    def parent_goal_id(self) -> str | None:
        """ID of the parent goal (via SUBGOAL_OF synapse). Cached in metadata."""
        return self.metadata.get("_parent_goal_id")

    def with_goal_state(
        self,
        state: str,
        priority: int = 5,
        keywords: list[str] | None = None,
        parent_goal_id: str | None = None,
    ) -> Neuron:
        """Create a new Neuron with updated goal state.

        Args:
            state: Goal state ('active', 'paused', 'completed')
            priority: Goal priority 1-10
            keywords: Keywords for EMA seeding
            parent_goal_id: Optional parent goal ID for hierarchy

        Returns:
            New Neuron with goal metadata
        """
        _valid_states = ("active", "paused", "completed")
        if state not in _valid_states:
            msg = f"Invalid goal state '{state}', must be one of {_valid_states}"
            raise ValueError(msg)
        updates: dict[str, Any] = {
            "_goal_state": state,
            "_goal_priority": max(1, min(10, priority)),
        }
        if keywords is not None:
            updates["_goal_keywords"] = keywords
        if parent_goal_id is not None:
            updates["_parent_goal_id"] = parent_goal_id
        return self.with_metadata(**updates)

    @classmethod
    def create(
        cls,
        type: NeuronType,
        content: str,
        metadata: dict[str, Any] | None = None,
        neuron_id: str | None = None,
        content_hash: int = 0,
        ephemeral: bool = False,
        grounded: bool = False,
        confidence: float | None = None,
    ) -> Neuron:
        """
        Factory method to create a new Neuron.

        Args:
            type: The type of neuron
            content: The content/value
            metadata: Optional metadata dict
            neuron_id: Optional explicit ID (generates UUID if not provided)
            content_hash: SimHash fingerprint for near-duplicate detection
            ephemeral: If True, this neuron is session-scoped and auto-expires

        Returns:
            A new Neuron instance
        """
        final_metadata = dict(metadata) if metadata else {}
        if grounded:
            final_metadata["_grounded"] = True
            final_metadata["_confidence"] = 1.0 if confidence is None else confidence
        elif confidence is not None:
            final_metadata["_confidence"] = confidence
        return cls(
            id=neuron_id or str(uuid4()),
            type=type,
            content=content,
            metadata=final_metadata,
            content_hash=content_hash,
            created_at=utcnow(),
            ephemeral=ephemeral,
        )

    def with_metadata(self, **kwargs: Any) -> Neuron:
        """
        Create a new Neuron with updated metadata.

        Args:
            **kwargs: Metadata key-value pairs to add/update

        Returns:
            New Neuron with merged metadata
        """
        return Neuron(
            id=self.id,
            type=self.type,
            content=self.content,
            metadata={**self.metadata, **kwargs},
            content_hash=self.content_hash,
            created_at=self.created_at,
            ephemeral=self.ephemeral,
        )

    def with_grounded(self, grounded: bool = True, confidence: float = 1.0) -> Neuron:
        """Create a new Neuron with updated grounding status."""
        updates: dict[str, Any] = {"_grounded": grounded, "_confidence": confidence}
        if not grounded:
            # Remove grounding keys when ungrounding
            new_meta = {
                k: v for k, v in self.metadata.items() if k not in ("_grounded", "_confidence")
            }
            return Neuron(
                id=self.id,
                type=self.type,
                content=self.content,
                metadata=new_meta,
                content_hash=self.content_hash,
                created_at=self.created_at,
                ephemeral=self.ephemeral,
            )
        return self.with_metadata(**updates)


@dataclass(frozen=True)
class NeuronState:
    """
    Mutable activation state for a neuron.

    Separated from Neuron to allow state changes without
    modifying the immutable neuron data.

    Attributes:
        neuron_id: Reference to the associated Neuron
        activation_level: Current activation (0.0 - 1.0)
        access_frequency: How many times this neuron has been activated
        last_activated: When this neuron was last activated
        decay_rate: How fast activation decays over time
        created_at: When this state was created
    """

    neuron_id: str
    activation_level: float = 0.0
    access_frequency: int = 0
    last_activated: datetime | None = None
    decay_rate: float = 0.1
    created_at: datetime = field(default_factory=utcnow)
    firing_threshold: float = 0.3
    refractory_until: datetime | None = None
    refractory_period_ms: float = 500.0
    homeostatic_target: float = 0.5

    def activate(
        self,
        level: float = 1.0,
        now: datetime | None = None,
        sigmoid_steepness: float = 6.0,
    ) -> NeuronState:
        """
        Create a new state with sigmoid-gated activation.

        Applies a sigmoid function to produce bio-realistic nonlinear
        gating. If the neuron is in its refractory period, returns self
        unchanged (no activation, no frequency increment).

        Args:
            level: Activation level to set (clamped to 0.0-1.0 before sigmoid)
            now: Current time (for testability; defaults to utcnow)
            sigmoid_steepness: Steepness of sigmoid curve (default 6.0)

        Returns:
            New NeuronState with updated activation
        """
        now = now or utcnow()

        # Refractory check: neuron cannot fire during cooldown
        if self.refractory_until is not None and now < self.refractory_until:
            return self

        clamped_level = max(0.0, min(1.0, level))
        sigmoid_level = 1.0 / (1.0 + math.exp(-sigmoid_steepness * (clamped_level - 0.5)))

        # Set refractory period if neuron fires
        new_refractory = self.refractory_until
        if sigmoid_level >= self.firing_threshold:
            new_refractory = now + timedelta(milliseconds=self.refractory_period_ms)

        return NeuronState(
            neuron_id=self.neuron_id,
            activation_level=sigmoid_level,
            access_frequency=self.access_frequency + 1,
            last_activated=now,
            decay_rate=self.decay_rate,
            created_at=self.created_at,
            firing_threshold=self.firing_threshold,
            refractory_until=new_refractory,
            refractory_period_ms=self.refractory_period_ms,
            homeostatic_target=self.homeostatic_target,
        )

    def decay(self, time_delta_seconds: float) -> NeuronState:
        """
        Apply decay to activation based on time elapsed.

        Uses exponential decay: new_level = old_level * e^(-decay_rate * time)

        Args:
            time_delta_seconds: Time elapsed since last update

        Returns:
            New NeuronState with decayed activation
        """
        if time_delta_seconds <= 0:
            return self  # No decay for non-positive time deltas

        days_elapsed = time_delta_seconds / 86400  # Convert to days
        decay_factor = math.exp(-self.decay_rate * days_elapsed)
        new_level = self.activation_level * decay_factor

        return NeuronState(
            neuron_id=self.neuron_id,
            activation_level=new_level,
            access_frequency=self.access_frequency,
            last_activated=self.last_activated,
            decay_rate=self.decay_rate,
            created_at=self.created_at,
            firing_threshold=self.firing_threshold,
            refractory_until=self.refractory_until,
            refractory_period_ms=self.refractory_period_ms,
            homeostatic_target=self.homeostatic_target,
        )

    @property
    def is_active(self) -> bool:
        """Check if neuron is currently active (above threshold)."""
        return self.activation_level > 0.1

    @property
    def fired(self) -> bool:
        """Check if neuron activation meets firing threshold."""
        return self.activation_level >= self.firing_threshold

    @property
    def in_refractory(self) -> bool:
        """Check if neuron is in refractory cooldown period."""
        if self.refractory_until is None:
            return False
        return utcnow() < self.refractory_until
