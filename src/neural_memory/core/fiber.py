"""Fiber data structures - memory clusters of related neurons."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from neural_memory.utils.timeutils import utcnow


@dataclass(frozen=True)
class Fiber:
    """
    A fiber represents a signal pathway through related neurons.

    Fibers are not just memory clusters - they are ordered pathways
    that conduct activation signals. Like neural fibers in the brain,
    they have conductivity that affects signal transmission quality.

    Attributes:
        id: Unique identifier
        neuron_ids: Set of neuron IDs in this fiber
        synapse_ids: Set of synapse IDs connecting neurons in this fiber
        anchor_neuron_id: Primary entry point neuron for this fiber
        pathway: Ordered sequence of neuron IDs forming the signal path
        conductivity: Signal transmission quality (0.0 - 1.0)
        last_conducted: When this fiber last conducted a signal
        time_start: Earliest timestamp in this memory
        time_end: Latest timestamp in this memory
        coherence: How tightly connected the neurons are (0.0 - 1.0)
        salience: Importance/relevance score (0.0 - 1.0)
        frequency: Number of times this fiber has been accessed
        summary: Optional compressed text summary
        auto_tags: Tags generated automatically (entity/keyword extraction)
        agent_tags: Tags provided by the calling agent
        metadata: Additional fiber-specific data
        created_at: When this fiber was created
    """

    id: str
    neuron_ids: set[str]
    synapse_ids: set[str]
    anchor_neuron_id: str
    # Signal pathway fields
    pathway: list[str] = field(default_factory=list)
    conductivity: float = 1.0
    last_conducted: datetime | None = None
    # Temporal bounds
    time_start: datetime | None = None
    time_end: datetime | None = None
    # Quality metrics
    coherence: float = 0.0
    salience: float = 0.0
    frequency: int = 0
    summary: str | None = None
    # Tag origin tracking (v0.14.0)
    auto_tags: set[str] = field(default_factory=set)
    agent_tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    compression_tier: int = 0
    pinned: bool = False
    created_at: datetime = field(default_factory=utcnow)
    # Lazy pathway index cache (not part of constructor/repr/compare)
    _pathway_index: dict[str, int] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def tags(self) -> set[str]:
        """Union of auto_tags and agent_tags (backward compatible)."""
        return self.auto_tags | self.agent_tags

    @classmethod
    def create(
        cls,
        neuron_ids: set[str],
        synapse_ids: set[str],
        anchor_neuron_id: str,
        pathway: list[str] | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        summary: str | None = None,
        tags: set[str] | None = None,
        auto_tags: set[str] | None = None,
        agent_tags: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
        fiber_id: str | None = None,
    ) -> Fiber:
        """
        Factory method to create a new Fiber.

        Args:
            neuron_ids: Set of neuron IDs
            synapse_ids: Set of synapse IDs
            anchor_neuron_id: Primary entry point
            pathway: Ordered sequence of neuron IDs forming signal path
            time_start: Optional start time
            time_end: Optional end time
            summary: Optional text summary
            tags: Legacy param — if provided alone, assigned to agent_tags
            auto_tags: Tags from automatic extraction (entity/keyword)
            agent_tags: Tags from the calling agent
            metadata: Optional metadata
            fiber_id: Optional explicit ID

        Returns:
            A new Fiber instance
        """
        if anchor_neuron_id not in neuron_ids:
            raise ValueError(f"Anchor neuron {anchor_neuron_id} must be in neuron_ids")

        # Default pathway starts with anchor
        if pathway is None:
            pathway = [anchor_neuron_id]

        # Tag origin resolution: if caller uses legacy 'tags' param,
        # treat them as agent-provided tags for backward compatibility
        effective_auto = auto_tags or set()
        effective_agent = agent_tags or set()
        if tags is not None and not auto_tags and not agent_tags:
            effective_agent = tags

        return cls(
            id=fiber_id or str(uuid4()),
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_neuron_id=anchor_neuron_id,
            pathway=pathway,
            conductivity=1.0,
            last_conducted=None,
            time_start=time_start,
            time_end=time_end,
            summary=summary,
            auto_tags=effective_auto,
            agent_tags=effective_agent,
            metadata=metadata or {},
            created_at=utcnow(),
        )

    def access(self) -> Fiber:
        """
        Create a new Fiber with incremented access frequency.

        Returns:
            New Fiber with frequency + 1
        """
        return replace(self, frequency=self.frequency + 1)

    def with_salience(self, salience: float) -> Fiber:
        """
        Create a new Fiber with updated salience.

        Args:
            salience: New salience value (clamped to 0.0-1.0)

        Returns:
            New Fiber with updated salience
        """
        return replace(self, salience=max(0.0, min(1.0, salience)))

    def with_summary(self, summary: str) -> Fiber:
        """
        Create a new Fiber with a summary.

        Args:
            summary: The summary text

        Returns:
            New Fiber with summary
        """
        return replace(self, summary=summary)

    def add_tags(self, *new_tags: str) -> Fiber:
        """
        Create a new Fiber with additional agent tags.

        External tag additions are treated as agent-origin by default.

        Args:
            *new_tags: Tags to add

        Returns:
            New Fiber with merged agent_tags
        """
        return replace(self, agent_tags=self.agent_tags | set(new_tags))

    def add_auto_tags(self, *new_tags: str) -> Fiber:
        """
        Create a new Fiber with additional auto-generated tags.

        Args:
            *new_tags: Tags to add to auto_tags

        Returns:
            New Fiber with merged auto_tags
        """
        return replace(self, auto_tags=self.auto_tags | set(new_tags))

    def conduct(
        self,
        conducted_at: datetime | None = None,
        reinforce: bool = True,
    ) -> Fiber:
        """
        Create a new Fiber after conducting a signal through it.

        Conducting a fiber:
        - Updates last_conducted timestamp
        - Optionally increases conductivity (reinforcement)
        - Increases frequency

        Args:
            conducted_at: When the signal was conducted (default: now)
            reinforce: If True, slightly increase conductivity

        Returns:
            New Fiber with updated conduction state
        """
        new_conductivity = self.conductivity
        if reinforce:
            new_conductivity = min(1.0, self.conductivity + 0.02)

        return replace(
            self,
            conductivity=new_conductivity,
            last_conducted=conducted_at or utcnow(),
            frequency=self.frequency + 1,
        )

    def with_conductivity(self, conductivity: float) -> Fiber:
        """
        Create a new Fiber with updated conductivity.

        Args:
            conductivity: New conductivity value (clamped to 0.0-1.0)

        Returns:
            New Fiber with updated conductivity
        """
        return replace(self, conductivity=max(0.0, min(1.0, conductivity)))

    @property
    def neuron_count(self) -> int:
        """Number of neurons in this fiber."""
        return len(self.neuron_ids)

    @property
    def synapse_count(self) -> int:
        """Number of synapses in this fiber."""
        return len(self.synapse_ids)

    @property
    def time_span(self) -> float | None:
        """
        Duration of this memory in seconds.

        Returns None if time bounds are not set.
        """
        if self.time_start and self.time_end:
            return (self.time_end - self.time_start).total_seconds()
        return None

    def contains_neuron(self, neuron_id: str) -> bool:
        """Check if this fiber contains a specific neuron."""
        return neuron_id in self.neuron_ids

    def overlaps_time(self, start: datetime, end: datetime) -> bool:
        """
        Check if this fiber's time range overlaps with given range.

        Args:
            start: Query start time
            end: Query end time

        Returns:
            True if there is any overlap
        """
        if self.time_start is None or self.time_end is None:
            return False

        return self.time_start <= end and self.time_end >= start

    @property
    def pathway_length(self) -> int:
        """Number of neurons in the signal pathway."""
        return len(self.pathway)

    def _ensure_pathway_index(self) -> dict[str, int]:
        """Build pathway index lazily on first access."""
        if self._pathway_index is None:
            index = {nid: i for i, nid in enumerate(self.pathway)}
            object.__setattr__(self, "_pathway_index", index)
            return index
        return self._pathway_index

    def pathway_position(self, neuron_id: str) -> int | None:
        """Get the position of a neuron in the pathway. O(1) after first call."""
        return self._ensure_pathway_index().get(neuron_id)

    def is_in_pathway(self, neuron_id: str) -> bool:
        """Check if a neuron is in the signal pathway. O(1) after first call."""
        return neuron_id in self._ensure_pathway_index()
