"""Brain container - the top-level memory structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


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
    activation_threshold: float = 0.2
    max_spread_hops: int = 4
    max_context_tokens: int = 1500
    default_synapse_weight: float = 0.5
    hebbian_delta: float = 0.03
    hebbian_threshold: float = 0.5
    hebbian_initial_weight: float = 0.2

    def with_updates(self, **kwargs: Any) -> BrainConfig:
        """Create a new config with updated values."""
        return BrainConfig(
            decay_rate=kwargs.get("decay_rate", self.decay_rate),
            reinforcement_delta=kwargs.get("reinforcement_delta", self.reinforcement_delta),
            activation_threshold=kwargs.get("activation_threshold", self.activation_threshold),
            max_spread_hops=kwargs.get("max_spread_hops", self.max_spread_hops),
            max_context_tokens=kwargs.get("max_context_tokens", self.max_context_tokens),
            default_synapse_weight=kwargs.get(
                "default_synapse_weight", self.default_synapse_weight
            ),
            hebbian_delta=kwargs.get("hebbian_delta", self.hebbian_delta),
            hebbian_threshold=kwargs.get("hebbian_threshold", self.hebbian_threshold),
            hebbian_initial_weight=kwargs.get(
                "hebbian_initial_weight", self.hebbian_initial_weight
            ),
        )


@dataclass
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
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
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
            updated_at=datetime.utcnow(),
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
            updated_at=datetime.utcnow(),
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
            updated_at=datetime.utcnow(),
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
            updated_at=datetime.utcnow(),
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
            updated_at=datetime.utcnow(),
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
            updated_at=datetime.utcnow(),
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


@dataclass
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
