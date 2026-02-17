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

    @classmethod
    def create(
        cls,
        type: NeuronType,
        content: str,
        metadata: dict[str, Any] | None = None,
        neuron_id: str | None = None,
        content_hash: int = 0,
    ) -> Neuron:
        """
        Factory method to create a new Neuron.

        Args:
            type: The type of neuron
            content: The content/value
            metadata: Optional metadata dict
            neuron_id: Optional explicit ID (generates UUID if not provided)
            content_hash: SimHash fingerprint for near-duplicate detection

        Returns:
            A new Neuron instance
        """
        return cls(
            id=neuron_id or str(uuid4()),
            type=type,
            content=content,
            metadata=metadata or {},
            content_hash=content_hash,
            created_at=utcnow(),
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
        )


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
