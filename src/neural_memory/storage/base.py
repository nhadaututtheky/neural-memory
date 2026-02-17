"""Abstract base class for neural storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from neural_memory.core.brain import Brain, BrainSnapshot
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
    from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
    from neural_memory.core.synapse import Synapse, SynapseType
    from neural_memory.engine.brain_versioning import BrainVersion
    from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage


class NeuralStorage(ABC):
    """
    Abstract interface for neural graph storage.

    Implementations must provide all methods for storing and
    retrieving neurons, synapses, fibers, and brain metadata.
    """

    @property
    def current_brain_id(self) -> str | None:
        """The active brain ID, or None if not set."""
        return getattr(self, "_current_brain_id", None)

    # ========== Batch Operations ==========

    def disable_auto_save(self) -> None:  # noqa: B027
        """Disable auto-save for batch operations. No-op by default."""

    def enable_auto_save(self) -> None:  # noqa: B027
        """Re-enable auto-save after batch operations. No-op by default."""

    async def batch_save(self) -> None:  # noqa: B027
        """Flush pending writes from batch mode. No-op by default."""

    # ========== Neuron Operations ==========

    @abstractmethod
    async def add_neuron(self, neuron: Neuron) -> str:
        """
        Add a neuron to storage.

        Args:
            neuron: The neuron to add

        Returns:
            The neuron ID

        Raises:
            ValueError: If neuron with same ID already exists
        """
        ...

    @abstractmethod
    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        """
        Get a neuron by ID.

        Args:
            neuron_id: The neuron ID

        Returns:
            The neuron if found, None otherwise
        """
        ...

    async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        """Get multiple neurons by ID in a single operation.

        Default implementation falls back to sequential get_neuron.
        Backends should override for batch efficiency.

        Args:
            neuron_ids: List of neuron IDs to fetch

        Returns:
            Dict mapping neuron_id to Neuron for found neurons
        """
        results: dict[str, Neuron] = {}
        for nid in neuron_ids:
            neuron = await self.get_neuron(nid)
            if neuron is not None:
                results[nid] = neuron
        return results

    @abstractmethod
    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
    ) -> list[Neuron]:
        """
        Find neurons matching criteria.

        Args:
            type: Filter by neuron type
            content_contains: Filter by content substring (case-insensitive)
            content_exact: Filter by exact content match
            time_range: Filter by created_at within range
            limit: Maximum results to return

        Returns:
            List of matching neurons
        """
        ...

    @abstractmethod
    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: NeuronType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest neurons matching a prefix, ranked by relevance + frequency.

        Returns list of dicts with keys: neuron_id, content, type,
        access_frequency, activation_level, score.
        """
        ...

    @abstractmethod
    async def update_neuron(self, neuron: Neuron) -> None:
        """
        Update an existing neuron.

        Args:
            neuron: The updated neuron (must have existing ID)

        Raises:
            ValueError: If neuron doesn't exist
        """
        ...

    @abstractmethod
    async def delete_neuron(self, neuron_id: str) -> bool:
        """
        Delete a neuron and its connected synapses.

        Args:
            neuron_id: The neuron ID to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    # ========== Neuron State Operations ==========

    @abstractmethod
    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        """
        Get the activation state for a neuron.

        Args:
            neuron_id: The neuron ID

        Returns:
            The state if found, None otherwise
        """
        ...

    @abstractmethod
    async def update_neuron_state(self, state: NeuronState) -> None:
        """
        Update or create neuron activation state.

        Args:
            state: The state to save
        """
        ...

    async def get_all_neuron_states(self) -> list[NeuronState]:
        """Get all neuron states for the current brain.

        Returns:
            List of all neuron states
        """
        raise NotImplementedError

    async def get_neuron_states_batch(self, neuron_ids: list[str]) -> dict[str, NeuronState]:
        """Get activation states for multiple neurons in one call.

        Default implementation falls back to sequential get_neuron_state.
        Backends should override for batch efficiency.

        Args:
            neuron_ids: List of neuron IDs to fetch states for

        Returns:
            Dict mapping neuron_id to NeuronState for found states
        """
        result: dict[str, NeuronState] = {}
        for nid in neuron_ids:
            state = await self.get_neuron_state(nid)
            if state is not None:
                result[nid] = state
        return result

    # ========== Synapse Operations ==========

    @abstractmethod
    async def add_synapse(self, synapse: Synapse) -> str:
        """
        Add a synapse to storage.

        Args:
            synapse: The synapse to add

        Returns:
            The synapse ID

        Raises:
            ValueError: If synapse with same ID exists, or neurons don't exist
        """
        ...

    @abstractmethod
    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        """
        Get a synapse by ID.

        Args:
            synapse_id: The synapse ID

        Returns:
            The synapse if found, None otherwise
        """
        ...

    @abstractmethod
    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
    ) -> list[Synapse]:
        """
        Find synapses matching criteria.

        Args:
            source_id: Filter by source neuron
            target_id: Filter by target neuron
            type: Filter by synapse type
            min_weight: Filter by minimum weight

        Returns:
            List of matching synapses
        """
        ...

    @abstractmethod
    async def update_synapse(self, synapse: Synapse) -> None:
        """
        Update an existing synapse.

        Args:
            synapse: The updated synapse

        Raises:
            ValueError: If synapse doesn't exist
        """
        ...

    @abstractmethod
    async def delete_synapse(self, synapse_id: str) -> bool:
        """
        Delete a synapse.

        Args:
            synapse_id: The synapse ID to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    async def get_all_synapses(self) -> list[Synapse]:
        """Get all synapses for the current brain.

        Returns:
            List of all synapses
        """
        raise NotImplementedError

    async def get_synapses_for_neurons(
        self,
        neuron_ids: list[str],
        direction: str = "out",
    ) -> dict[str, list[Synapse]]:
        """Get all synapses for multiple neurons in one call.

        Default implementation falls back to sequential get_synapses.
        Backends should override for batch efficiency.

        Args:
            neuron_ids: List of neuron IDs
            direction: "out" for outgoing, "in" for incoming synapses

        Returns:
            Dict mapping neuron_id to list of synapses
        """
        result: dict[str, list[Synapse]] = {}
        for nid in neuron_ids:
            if direction == "out":
                result[nid] = await self.get_synapses(source_id=nid)
            else:
                result[nid] = await self.get_synapses(target_id=nid)
        return result

    # ========== Graph Traversal ==========

    @abstractmethod
    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        """
        Get neighboring neurons connected by synapses.

        Args:
            neuron_id: The central neuron ID
            direction: Which direction to traverse
                - "out": Only follow outgoing synapses
                - "in": Only follow incoming synapses
                - "both": Follow both directions
            synapse_types: Only follow these synapse types
            min_weight: Only follow synapses with weight >= this

        Returns:
            List of (neighbor_neuron, connecting_synapse) tuples
        """
        ...

    @abstractmethod
    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
    ) -> list[tuple[Neuron, Synapse]] | None:
        """
        Find shortest path between two neurons.

        Args:
            source_id: Starting neuron ID
            target_id: Target neuron ID
            max_hops: Maximum path length

        Returns:
            List of (neuron, synapse) pairs representing path, or None if no path
        """
        ...

    # ========== Fiber Operations ==========

    @abstractmethod
    async def add_fiber(self, fiber: Fiber) -> str:
        """
        Add a fiber to storage.

        Args:
            fiber: The fiber to add

        Returns:
            The fiber ID

        Raises:
            ValueError: If fiber with same ID exists
        """
        ...

    @abstractmethod
    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        """
        Get a fiber by ID.

        Args:
            fiber_id: The fiber ID

        Returns:
            The fiber if found, None otherwise
        """
        ...

    @abstractmethod
    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        limit: int = 100,
    ) -> list[Fiber]:
        """
        Find fibers matching criteria.

        Args:
            contains_neuron: Filter by containing this neuron ID
            time_overlaps: Filter by time range overlap
            tags: Filter by having all these tags
            min_salience: Filter by minimum salience
            limit: Maximum results

        Returns:
            List of matching fibers
        """
        ...

    @abstractmethod
    async def update_fiber(self, fiber: Fiber) -> None:
        """
        Update an existing fiber.

        Args:
            fiber: The updated fiber

        Raises:
            ValueError: If fiber doesn't exist
        """
        ...

    @abstractmethod
    async def delete_fiber(self, fiber_id: str) -> bool:
        """
        Delete a fiber.

        Args:
            fiber_id: The fiber ID to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    async def find_fibers_batch(
        self,
        neuron_ids: list[str],
        limit_per_neuron: int = 10,
    ) -> list[Fiber]:
        """Find fibers containing ANY of the given neurons, deduplicated.

        Default implementation falls back to sequential find_fibers.
        Backends should override for batch efficiency.

        Args:
            neuron_ids: List of neuron IDs to search for
            limit_per_neuron: Max fibers per neuron in fallback

        Returns:
            Deduplicated list of fibers containing any of the neurons
        """
        seen: set[str] = set()
        result: list[Fiber] = []
        for nid in neuron_ids:
            for f in await self.find_fibers(contains_neuron=nid, limit=limit_per_neuron):
                if f.id not in seen:
                    seen.add(f.id)
                    result.append(f)
        return result

    @abstractmethod
    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        """
        Get fibers with ordering.

        Args:
            limit: Maximum results
            order_by: Field to order by
            descending: Sort descending if True

        Returns:
            List of fibers
        """
        ...

    # ========== Brain Operations ==========

    @abstractmethod
    async def save_brain(self, brain: Brain) -> None:
        """
        Save or update brain metadata.

        Args:
            brain: The brain to save
        """
        ...

    @abstractmethod
    async def get_brain(self, brain_id: str) -> Brain | None:
        """
        Get brain metadata by ID.

        Args:
            brain_id: The brain ID

        Returns:
            The brain if found, None otherwise
        """
        ...

    async def find_brain_by_name(self, name: str) -> Brain | None:
        """Find a brain by its name.

        Args:
            name: The brain name to search for

        Returns:
            The brain if found, None otherwise
        """
        raise NotImplementedError

    @abstractmethod
    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        """
        Export entire brain as a snapshot.

        Args:
            brain_id: The brain ID to export

        Returns:
            Complete snapshot of the brain

        Raises:
            ValueError: If brain doesn't exist
        """
        ...

    @abstractmethod
    async def import_brain(
        self,
        snapshot: BrainSnapshot,
        target_brain_id: str | None = None,
    ) -> str:
        """
        Import a brain snapshot.

        Args:
            snapshot: The snapshot to import
            target_brain_id: Optional ID for the imported brain

        Returns:
            The ID of the imported brain
        """
        ...

    # ========== Statistics ==========

    @abstractmethod
    async def get_stats(self, brain_id: str) -> dict[str, int]:
        """
        Get statistics for a brain.

        Args:
            brain_id: The brain ID

        Returns:
            Dict with keys: neuron_count, synapse_count, fiber_count
        """
        ...

    @abstractmethod
    async def get_enhanced_stats(self, brain_id: str) -> dict[str, Any]:
        """
        Get enhanced statistics for a brain.

        Includes hot neurons, DB size, daily activity, synapse stats,
        neuron type breakdown, and memory time range.

        Args:
            brain_id: The brain ID

        Returns:
            Dict with enhanced statistics
        """
        ...

    # ========== Typed Memory Operations ==========

    async def add_typed_memory(self, typed_memory: TypedMemory) -> str:
        """Add a typed memory to storage.

        Args:
            typed_memory: The typed memory to add

        Returns:
            The fiber ID used as the typed memory key
        """
        raise NotImplementedError

    async def get_typed_memory(self, fiber_id: str) -> TypedMemory | None:
        """Get a typed memory by its fiber ID.

        Args:
            fiber_id: The fiber ID

        Returns:
            The typed memory if found, None otherwise
        """
        raise NotImplementedError

    async def find_typed_memories(
        self,
        memory_type: MemoryType | None = None,
        min_priority: Priority | None = None,
        include_expired: bool = False,
        project_id: str | None = None,
        tags: set[str] | None = None,
        limit: int = 100,
    ) -> list[TypedMemory]:
        """Find typed memories matching criteria.

        Args:
            memory_type: Filter by memory type
            min_priority: Filter by minimum priority
            include_expired: Include expired memories
            project_id: Filter by project ID
            tags: Filter by having all these tags
            limit: Maximum results to return

        Returns:
            List of matching typed memories
        """
        raise NotImplementedError

    async def update_typed_memory(self, typed_memory: TypedMemory) -> None:
        """Update an existing typed memory.

        Args:
            typed_memory: The updated typed memory

        Raises:
            ValueError: If typed memory doesn't exist
        """
        raise NotImplementedError

    async def delete_typed_memory(self, fiber_id: str) -> bool:
        """Delete a typed memory by its fiber ID.

        Args:
            fiber_id: The fiber ID to delete

        Returns:
            True if deleted, False if not found
        """
        raise NotImplementedError

    async def get_expired_memories(self, limit: int = 100) -> list[TypedMemory]:
        """Get expired typed memories for the current brain.

        Args:
            limit: Maximum number of expired memories to return.

        Returns:
            List of expired typed memories
        """
        raise NotImplementedError

    async def get_expired_memory_count(self) -> int:
        """Get count of expired typed memories for the current brain.

        Cheap COUNT query for health pulse — avoids materializing full objects.

        Returns:
            Number of expired typed memories
        """
        raise NotImplementedError

    async def get_expiring_memories_for_fibers(
        self,
        fiber_ids: list[str],
        within_days: int = 7,
    ) -> list[TypedMemory]:
        """Get typed memories expiring within N days for given fibers.

        Returns only memories that are NOT yet expired but WILL expire
        within the specified window.

        Args:
            fiber_ids: Fiber IDs to check
            within_days: Number of days from now to check

        Returns:
            List of TypedMemory objects expiring within the window
        """
        raise NotImplementedError

    async def get_expiring_memory_count(self, within_days: int = 7) -> int:
        """Count typed memories expiring within N days for current brain.

        Cheap COUNT query — avoids materializing full objects.

        Args:
            within_days: Days from now

        Returns:
            Count of soon-to-expire memories
        """
        raise NotImplementedError

    async def get_stale_fiber_count(self, brain_id: str, stale_days: int = 90) -> int:
        """Get count of fibers that are stale (unused for a long time).

        A fiber is stale if it was never conducted and created_at is older
        than stale_days, or if last_conducted is older than stale_days.

        Args:
            brain_id: The brain ID
            stale_days: Number of days after which a fiber is considered stale

        Returns:
            Number of stale fibers
        """
        raise NotImplementedError

    # ========== Maturation Operations ==========

    async def save_maturation(self, record: MaturationRecord) -> None:  # noqa: B027
        """Save or update a maturation record.

        Default no-op — backends that support maturation should override.

        Args:
            record: The maturation record to save
        """

    async def get_maturation(self, fiber_id: str) -> MaturationRecord | None:
        """Get a maturation record for a fiber.

        Args:
            fiber_id: The fiber ID

        Returns:
            The maturation record if found, None otherwise
        """
        return None

    async def find_maturations(
        self,
        stage: MemoryStage | None = None,
        min_rehearsal_count: int = 0,
    ) -> list[MaturationRecord]:
        """Find maturation records matching criteria.

        Args:
            stage: Optional stage filter
            min_rehearsal_count: Minimum rehearsal count filter

        Returns:
            List of matching maturation records
        """
        return []

    # ========== Co-Activation Operations ==========

    async def record_co_activation(
        self,
        neuron_a: str,
        neuron_b: str,
        binding_strength: float,
        source_anchor: str | None = None,
    ) -> str:
        """Record a co-activation event between two neurons.

        Implementations must store pairs in canonical order (a < b).

        Args:
            neuron_a: First neuron ID
            neuron_b: Second neuron ID
            binding_strength: Strength of the co-activation (0.0-1.0)
            source_anchor: Optional anchor neuron that triggered this

        Returns:
            The event ID
        """
        raise NotImplementedError

    async def get_co_activation_counts(
        self,
        since: datetime | None = None,
        min_count: int = 1,
    ) -> list[tuple[str, str, int, float]]:
        """Get aggregated co-activation counts for neuron pairs.

        Args:
            since: Only count events after this time
            min_count: Minimum co-activation count to include

        Returns:
            List of (neuron_a, neuron_b, count, avg_binding_strength) tuples
        """
        raise NotImplementedError

    async def prune_co_activations(self, older_than: datetime) -> int:
        """Remove co-activation events older than the given time.

        Args:
            older_than: Remove events created before this time

        Returns:
            Number of events pruned
        """
        raise NotImplementedError

    # ========== Action Event Operations ==========

    async def record_action(
        self,
        action_type: str,
        action_context: str = "",
        tags: tuple[str, ...] | list[str] = (),
        session_id: str | None = None,
        fiber_id: str | None = None,
    ) -> str:
        """Record an action event for habit learning.

        Args:
            action_type: Type of action (e.g., "remember", "recall")
            action_context: Optional context string
            tags: Tags for categorization
            session_id: Optional session grouping
            fiber_id: Optional associated fiber

        Returns:
            The action event ID
        """
        raise NotImplementedError

    async def get_action_sequences(
        self,
        session_id: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[Any]:
        """Get action events ordered by time.

        Args:
            session_id: Filter by session
            since: Only events after this time
            limit: Maximum events to return

        Returns:
            List of ActionEvent objects ordered by created_at
        """
        raise NotImplementedError

    async def prune_action_events(self, older_than: datetime) -> int:
        """Remove action events older than the given time.

        Args:
            older_than: Remove events created before this time

        Returns:
            Number of events pruned
        """
        raise NotImplementedError

    # ========== Version Operations ==========

    async def save_version(
        self,
        brain_id: str,
        version: BrainVersion,
        snapshot_json: str,
    ) -> None:
        """Save a brain version with its snapshot data.

        Args:
            brain_id: Brain ID
            version: The version metadata
            snapshot_json: Serialized snapshot JSON
        """
        raise NotImplementedError

    async def get_version(
        self,
        brain_id: str,
        version_id: str,
    ) -> tuple[BrainVersion, str] | None:
        """Get a version and its snapshot JSON by ID.

        Args:
            brain_id: Brain ID
            version_id: Version ID

        Returns:
            Tuple of (BrainVersion, snapshot_json) or None
        """
        raise NotImplementedError

    async def list_versions(
        self,
        brain_id: str,
        limit: int = 20,
    ) -> list[BrainVersion]:
        """List versions for a brain, most recent first.

        Args:
            brain_id: Brain ID
            limit: Maximum versions to return

        Returns:
            List of BrainVersion, newest first
        """
        raise NotImplementedError

    async def get_next_version_number(self, brain_id: str) -> int:
        """Get the next version number for a brain.

        Args:
            brain_id: Brain ID

        Returns:
            Next auto-incrementing version number
        """
        raise NotImplementedError

    async def delete_version(self, brain_id: str, version_id: str) -> bool:
        """Delete a specific version.

        Args:
            brain_id: Brain ID
            version_id: Version ID

        Returns:
            True if deleted, False if not found
        """
        raise NotImplementedError

    # ========== Cleanup ==========

    @abstractmethod
    async def clear(self, brain_id: str) -> None:
        """
        Clear all data for a brain.

        Args:
            brain_id: The brain ID to clear
        """
        ...
