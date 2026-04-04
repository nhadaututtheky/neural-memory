"""Abstract base class for neural storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from neural_memory.core.alert import Alert
    from neural_memory.core.brain import Brain, BrainSnapshot
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
    from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
    from neural_memory.core.review_schedule import ReviewSchedule
    from neural_memory.core.synapse import Synapse, SynapseType
    from neural_memory.engine.brain_versioning import BrainVersion
    from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage


class NeuralStorage(ABC):
    """
    Abstract interface for neural graph storage.

    Implementations must provide all methods for storing and
    retrieving neurons, synapses, fibers, and brain metadata.
    """

    _current_brain_id: str | None

    @property
    def brain_id(self) -> str | None:
        """The active brain ID, or None if not set."""
        return getattr(self, "_current_brain_id", None)

    @property
    def current_brain_id(self) -> str | None:
        """Alias for brain_id (backward compatibility)."""
        return self.brain_id

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

    async def find_neurons_exact_batch(
        self,
        contents: list[str],
        type: NeuronType | None = None,
        ephemeral: bool | None = None,
    ) -> dict[str, Neuron]:
        """Find neurons by exact content match for multiple contents at once.

        Default falls back to sequential find_neurons calls.
        Backends should override for batch efficiency.

        Args:
            contents: List of exact content strings to match
            type: Optional neuron type filter
            ephemeral: Filter by ephemeral flag (None=all, True=only ephemeral, False=only permanent)

        Returns:
            Dict mapping content string to first matching Neuron
        """
        results: dict[str, Neuron] = {}
        for content in contents:
            matches = await self.find_neurons(
                type=type, content_exact=content, limit=1, ephemeral=ephemeral
            )
            if matches:
                results[content] = matches[0]
        return results

    async def has_neuron_by_content_hash(self, content_hash: int) -> bool:
        """Check if a neuron with this content hash exists.

        Default returns False (no dedup). SQLite backend overrides with fast query.
        """
        return False

    @abstractmethod
    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
        offset: int = 0,
        ephemeral: bool | None = None,
    ) -> list[Neuron]:
        """
        Find neurons matching criteria.

        Args:
            type: Filter by neuron type
            content_contains: Filter by content substring (case-insensitive)
            content_exact: Filter by exact content match
            time_range: Filter by created_at within range
            limit: Maximum results to return
            offset: Number of rows to skip (for pagination)
            ephemeral: Filter by ephemeral flag (None=all, True=only ephemeral, False=only permanent)

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

    async def update_neuron_states_batch(self, states: list[NeuronState]) -> None:
        """Update multiple neuron states in one batch operation.

        Default implementation falls back to sequential update_neuron_state.
        Backends should override for batch efficiency.

        Args:
            states: List of neuron states to update
        """
        for s in states:
            await self.update_neuron_state(s)

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

    async def update_synapses_batch(self, synapses: list[Synapse]) -> None:
        """Update multiple synapses in one batch operation.

        Default implementation falls back to sequential update_synapse.
        Backends should override for batch efficiency.

        Args:
            synapses: List of synapses to update
        """
        for syn in synapses:
            await self.update_synapse(syn)

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
        bidirectional: bool = False,
    ) -> list[tuple[Neuron, Synapse]] | None:
        """
        Find shortest path between two neurons.

        Args:
            source_id: Starting neuron ID
            target_id: Target neuron ID
            max_hops: Maximum path length
            bidirectional: Traverse both incoming and outgoing edges

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

    async def search_fiber_summaries(self, query: str, *, limit: int = 10) -> list[Fiber]:
        """Search fiber summaries using full-text search.

        Default implementation returns empty list (opt-in for storage backends).
        """
        return []

    @abstractmethod
    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        metadata_key: str | None = None,
        limit: int = 100,
        tag_mode: str = "and",
    ) -> list[Fiber]:
        """
        Find fibers matching criteria.

        Args:
            contains_neuron: Filter by containing this neuron ID
            time_overlaps: Filter by time range overlap
            tags: Filter by tags ("and" = all must match, "or" = any matches)
            min_salience: Filter by minimum salience
            metadata_key: Filter by metadata containing this key (non-null value)
            limit: Maximum results
            tag_mode: "and" (all tags must match) or "or" (any tag matches)

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

    async def batch_update_ghost_shown(self, fiber_ids: list[str], timestamp: datetime) -> int:
        """Batch update last_ghost_shown_at for multiple fibers.

        Args:
            fiber_ids: Fiber IDs to update
            timestamp: Timestamp to set

        Returns:
            Number of fibers updated
        """
        raise NotImplementedError

    async def update_fiber_metadata(self, fiber_id: str, metadata: dict[str, Any]) -> None:
        """Merge metadata into a fiber's existing metadata JSON.

        Default implementation fetches the fiber, merges metadata, then calls
        update_fiber.  Backends may override for efficiency.

        Args:
            fiber_id: The fiber to update.
            metadata: New metadata values to merge (existing keys are overwritten).
        """
        from dataclasses import replace as dc_replace

        fiber = await self.get_fiber(fiber_id)
        if fiber is None:
            return
        updated_meta = {**fiber.metadata, **metadata}
        updated_fiber = dc_replace(fiber, metadata=updated_meta)
        await self.update_fiber(updated_fiber)

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
        tags: set[str] | None = None,
        tag_mode: str = "and",
    ) -> list[Fiber]:
        """Find fibers containing ANY of the given neurons, deduplicated.

        Default implementation falls back to sequential find_fibers.
        Backends should override for batch efficiency.

        Args:
            neuron_ids: List of neuron IDs to search for
            limit_per_neuron: Max fibers per neuron in fallback
            tags: Optional set of tags to filter by
            tag_mode: "and" (all tags must match) or "or" (any tag matches)

        Returns:
            Deduplicated list of fibers containing any of the neurons
        """
        seen: set[str] = set()
        result: list[Fiber] = []
        for nid in neuron_ids:
            for f in await self.find_fibers(
                contains_neuron=nid, limit=limit_per_neuron, tags=tags, tag_mode=tag_mode
            ):
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

    # ========== Lifecycle ==========

    async def close(self) -> None:  # noqa: B027
        """Close storage connections. No-op by default."""

    # ========== Brain Operations ==========

    def set_brain(self, brain_id: str) -> None:
        """Set the active brain context.

        Args:
            brain_id: The brain ID to activate
        """
        raise NotImplementedError

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
        tier: str | None = None,
    ) -> list[TypedMemory]:
        """Find typed memories matching criteria.

        Args:
            memory_type: Filter by memory type
            min_priority: Filter by minimum priority
            include_expired: Include expired memories
            project_id: Filter by project ID
            tags: Filter by having all these tags
            limit: Maximum results to return
            tier: Filter by loading tier ("hot", "warm", "cold")

        Returns:
            List of matching typed memories
        """
        raise NotImplementedError

    async def count_typed_memories(
        self,
        tier: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> int:
        """Count typed memories matching criteria (no limit cap).

        Args:
            tier: Filter by loading tier ("hot", "warm", "cold")
            memory_type: Filter by memory type

        Returns:
            Total count of matching typed memories
        """
        raise NotImplementedError

    async def count_typed_memories_grouped(
        self,
    ) -> list[tuple[str, str, int]]:
        """Count typed memories grouped by (memory_type, tier).

        Returns:
            List of (memory_type, tier, count) tuples.
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

    async def update_typed_memory_source(self, fiber_id: str, source: str) -> bool:
        """Update only the source field on a typed memory."""
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

    async def get_promotion_candidates(
        self,
        min_frequency: int = 5,
        source_type: str = "context",
    ) -> list[dict[str, Any]]:
        """Find typed memories eligible for auto-promotion.

        Returns context memories whose fibers have frequency >= min_frequency.
        """
        raise NotImplementedError

    async def promote_memory_type(
        self,
        fiber_id: str,
        new_type: MemoryType,
        new_expires_at: str | None = None,
    ) -> bool:
        """Promote a memory's type and update its expiry.

        Stores the original type in metadata for audit trail.
        Returns True if the promotion was applied.
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

    async def get_fiber_stage_counts(self, brain_id: str) -> dict[str, int]:
        """Get count of fibers grouped by maturation stage.

        Returns:
            Dict mapping stage names to counts, e.g.
            {"stm": 10, "working": 5, "episodic": 80, "semantic": 5}
        """
        raise NotImplementedError

    async def get_total_fiber_count(self) -> int:
        """Get total number of fibers for the current brain."""
        raise NotImplementedError

    async def get_keyword_df_batch(self, keywords: list[str]) -> dict[str, int]:
        """Get document frequency (fiber count) for a batch of keywords.

        Args:
            keywords: List of keyword strings (lowercased).

        Returns:
            Dict mapping keyword to its fiber_count. Missing keywords omitted.
        """
        raise NotImplementedError

    async def increment_keyword_df(self, keywords: list[str]) -> None:
        """Increment document frequency for each keyword by 1 (UPSERT).

        Args:
            keywords: List of keyword strings (lowercased) from the encoded memory.
        """
        raise NotImplementedError

    # ========== Entity Ref Operations (Lazy Entity Promotion) ==========

    async def add_entity_ref(
        self, entity_text: str, fiber_id: str, created_at: datetime | None = None
    ) -> None:
        """Record an entity mention for lazy promotion tracking."""
        raise NotImplementedError

    async def count_entity_refs(self, entity_text: str) -> int:
        """Count unpromoted mentions of an entity."""
        raise NotImplementedError

    async def get_entity_ref_fiber_ids(self, entity_text: str) -> list[str]:
        """Get fiber/anchor IDs that reference this entity."""
        raise NotImplementedError

    async def mark_entity_refs_promoted(self, entity_text: str) -> int:
        """Mark all refs for an entity as promoted."""
        raise NotImplementedError

    async def prune_old_entity_refs(self, max_age_days: int = 90) -> int:
        """Remove unpromoted entity refs older than max_age_days."""
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

    async def cleanup_orphaned_maturations(self) -> int:
        """Delete maturation records whose fiber no longer exists.

        Default no-op — backends that support maturation should override.

        Returns:
            Number of orphaned records removed.
        """
        return 0

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

    # ========== Review Schedule Operations ==========

    async def add_review_schedule(self, schedule: ReviewSchedule) -> str:
        """Insert or update a review schedule (upsert by fiber_id + brain_id).

        Args:
            schedule: The review schedule to save

        Returns:
            The fiber ID
        """
        raise NotImplementedError

    async def get_review_schedule(self, fiber_id: str) -> ReviewSchedule | None:
        """Get a review schedule by fiber ID.

        Args:
            fiber_id: The fiber ID

        Returns:
            The review schedule if found, None otherwise
        """
        raise NotImplementedError

    async def get_due_reviews(self, limit: int = 20) -> list[ReviewSchedule]:
        """Get review schedules that are due (next_review <= now).

        Args:
            limit: Maximum results (capped at 100)

        Returns:
            List of due review schedules, oldest first
        """
        raise NotImplementedError

    async def delete_review_schedule(self, fiber_id: str) -> bool:
        """Delete a review schedule.

        Args:
            fiber_id: The fiber ID

        Returns:
            True if deleted, False if not found
        """
        raise NotImplementedError

    async def get_review_stats(self) -> dict[str, int]:
        """Get review statistics for the current brain.

        Returns:
            Dict with keys: total, due, box_1..box_5
        """
        raise NotImplementedError

    # ========== Depth Prior Operations ==========

    async def get_depth_priors_batch(
        self,
        entity_texts: list[str],
    ) -> dict[str, list[Any]]:
        """Batch-fetch Bayesian depth priors for multiple entities.

        Args:
            entity_texts: Entity text strings to look up

        Returns:
            Dict mapping entity_text to list of DepthPrior objects
        """
        raise NotImplementedError

    async def upsert_depth_prior(self, prior: Any) -> None:
        """Insert or update a single depth prior.

        Args:
            prior: DepthPrior instance to persist
        """
        raise NotImplementedError

    async def get_stale_priors(self, older_than: datetime) -> list[Any]:
        """Find depth priors not updated since a given date.

        Args:
            older_than: Cutoff datetime; priors with last_updated before this are returned

        Returns:
            List of stale DepthPrior objects
        """
        raise NotImplementedError

    async def delete_depth_priors(self, entity_text: str) -> int:
        """Delete all depth priors for an entity.

        Args:
            entity_text: The entity text whose priors should be removed

        Returns:
            Number of rows deleted
        """
        raise NotImplementedError

    # ========== Compression Backup Operations ==========

    async def save_compression_backup(
        self,
        fiber_id: str,
        original_content: str,
        compression_tier: int,
        original_token_count: int,
        compressed_token_count: int,
    ) -> None:
        """Save (upsert) a pre-compression content backup for a fiber.

        Only called for reversible tiers (1-2).  Backends that do not
        support compression may leave this as a no-op.

        Args:
            fiber_id: The fiber whose content is being backed up.
            original_content: Full original text before compression.
            compression_tier: The tier to which the fiber is being compressed.
            original_token_count: Approximate token count before compression.
            compressed_token_count: Approximate token count after compression.
        """
        raise NotImplementedError

    async def get_compression_backup(self, fiber_id: str) -> dict[str, Any] | None:
        """Retrieve the compression backup for a fiber, if any.

        Args:
            fiber_id: The fiber ID to look up.

        Returns:
            Dict with backup fields, or None if no backup exists.
        """
        raise NotImplementedError

    async def delete_compression_backup(self, fiber_id: str) -> bool:
        """Delete the compression backup for a fiber.

        Args:
            fiber_id: The fiber ID whose backup should be removed.

        Returns:
            True if a row was deleted, False if no backup existed.
        """
        raise NotImplementedError

    async def get_compression_stats(self) -> dict[str, Any]:
        """Return aggregate compression statistics for the current brain.

        Returns:
            Dict with keys: ``total_backups``, ``by_tier``, ``total_tokens_saved``.
        """
        raise NotImplementedError

    # ========== Neuron Snapshot Operations (Tier 3-4 recovery) ==========

    async def save_neuron_snapshot(
        self,
        neuron_id: str,
        brain_id: str,
        original_content: str,
        compressed_at: str,
        tier: int,
    ) -> None:
        """Save (upsert) a pre-compression content snapshot for a neuron.

        Called before destructive Tier 3-4 compression so content can be
        recovered later via ``get_neuron_snapshot``.

        Args:
            neuron_id: The neuron whose content is being snapshotted.
            brain_id: Brain that owns the neuron.
            original_content: Full original text before compression.
            compressed_at: ISO timestamp of when compression occurred.
            tier: The compression tier being applied (3 or 4).
        """
        raise NotImplementedError

    async def get_neuron_snapshot(self, neuron_id: str) -> dict[str, Any] | None:
        """Retrieve the snapshot for a neuron, if any.

        Args:
            neuron_id: The neuron ID to look up.

        Returns:
            Dict with snapshot fields (neuron_id, original_content, compressed_at, tier),
            or None if no snapshot exists.
        """
        raise NotImplementedError

    async def delete_neuron_snapshot(self, neuron_id: str) -> bool:
        """Delete the snapshot for a neuron.

        Args:
            neuron_id: The neuron ID whose snapshot should be removed.

        Returns:
            True if a row was deleted, False if no snapshot existed.
        """
        raise NotImplementedError

    # ========== Ephemeral Operations ==========

    async def cleanup_ephemeral_neurons(self, max_age_hours: float = 24.0) -> int:
        """Delete ephemeral neurons older than max_age_hours.

        Returns:
            Number of deleted neurons.
        """
        raise NotImplementedError

    # ========== Access Tracking Operations ==========

    async def batch_update_last_accessed(self, neuron_ids: list[str]) -> None:
        """Update last_accessed_at for neurons in batch (single SQL UPDATE).

        Called after successful recall to track which neurons were activated.
        Default is a no-op; SQLite backend overrides for efficiency.

        Args:
            neuron_ids: List of neuron IDs whose last_accessed_at should be updated.
        """
        return  # No-op default; SQLite backend overrides.

    # ========== Lifecycle State Operations ==========

    async def update_neuron_lifecycle(
        self,
        neuron_id: str,
        lifecycle_state: str,
    ) -> None:
        """Update the lifecycle_state for a neuron.

        Args:
            neuron_id: The neuron ID to update.
            lifecycle_state: New lifecycle state string.
        """
        raise NotImplementedError

    async def update_neuron_frozen(self, neuron_id: str, frozen: bool) -> None:
        """Set or clear the frozen flag for a neuron.

        Args:
            neuron_id: The neuron ID to update.
            frozen: True to prevent compression, False to resume normal lifecycle.
        """
        raise NotImplementedError

    async def update_neuron_ephemeral(self, neuron_id: str, ephemeral: bool) -> None:
        """Set or clear the ephemeral flag for a neuron."""
        raise NotImplementedError

    async def update_neurons_ephemeral_batch(self, neuron_ids: list[str], ephemeral: bool) -> None:
        """Batch-set ephemeral flag for multiple neurons."""
        raise NotImplementedError

    async def get_lifecycle_distribution(self) -> dict[str, int]:
        """Return count of neurons by lifecycle_state for the current brain.

        Returns:
            Dict mapping state name to count, e.g. {"active": 100, "warm": 20, ...}.
        """
        raise NotImplementedError

    # ========== Change Log Operations ==========

    async def record_change(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        device_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a change to the log. Returns the sequence number (id).

        Args:
            entity_type: Type of entity changed ("neuron", "synapse", "fiber").
            entity_id: ID of the changed entity.
            operation: Operation performed ("insert", "update", "delete").
            device_id: ID of the device that made the change.
            payload: Optional dict of changed field values.

        Returns:
            Auto-incremented sequence number (id) of the new log entry.
        """
        raise NotImplementedError

    async def get_changes_since(self, sequence: int = 0, limit: int = 1000) -> list[Any]:
        """Get changes after a given sequence number, ordered by id ASC.

        Args:
            sequence: Return only entries with id > this value (0 = all).
            limit: Maximum number of entries to return (capped server-side).

        Returns:
            List of ChangeEntry objects ordered by id ascending.
        """
        raise NotImplementedError

    async def get_unsynced_changes(self, limit: int = 1000) -> list[Any]:
        """Get all unsynced changes, ordered by id ASC.

        Args:
            limit: Maximum number of entries to return (capped server-side).

        Returns:
            List of unsynced ChangeEntry objects ordered by id ascending.
        """
        raise NotImplementedError

    async def mark_synced(self, up_to_sequence: int) -> int:
        """Mark all changes up to a sequence number as synced.

        Args:
            up_to_sequence: Mark all entries with id <= this value as synced.

        Returns:
            Count of rows updated.
        """
        raise NotImplementedError

    async def prune_synced_changes(self, older_than_days: int = 30) -> int:
        """Delete synced changes older than N days.

        Args:
            older_than_days: Delete synced entries whose changed_at is older
                than this many days from now.

        Returns:
            Count of rows deleted.
        """
        raise NotImplementedError

    async def seed_change_log(self, device_id: str = "") -> dict[str, int]:
        """Seed the change log with all existing entities as 'insert' entries.

        Enables initial sync for brains created before sync was enabled.
        Only adds entities not already tracked in the change log.

        Args:
            device_id: Device ID to tag the seeded entries with.

        Returns:
            Dict with counts: neurons, synapses, fibers seeded.
        """
        raise NotImplementedError

    async def get_change_log_stats(self) -> dict[str, Any]:
        """Get change log statistics for the current brain.

        Returns:
            Dict with keys: ``total``, ``pending``, ``synced``, ``last_sequence``.
        """
        raise NotImplementedError

    # ========== Device Registry Operations ==========

    async def register_device(self, device_id: str, device_name: str = "") -> Any:
        """Register a device for the current brain (upsert).

        Args:
            device_id: Unique device identifier.
            device_name: Human-readable device name.

        Returns:
            DeviceRecord for the registered device.
        """
        raise NotImplementedError

    async def get_device(self, device_id: str) -> Any | None:
        """Get device info for a specific device.

        Args:
            device_id: The device ID to look up.

        Returns:
            DeviceRecord if found, None otherwise.
        """
        raise NotImplementedError

    async def list_devices(self) -> list[Any]:
        """List all registered devices for the current brain.

        Returns:
            List of DeviceRecord objects ordered by registration time ascending.
        """
        raise NotImplementedError

    async def update_device_sync(self, device_id: str, last_sync_sequence: int) -> None:
        """Update the last sync timestamp and sequence for a device.

        Args:
            device_id: The device to update.
            last_sync_sequence: The highest change log sequence the device has synced.
        """
        raise NotImplementedError

    async def remove_device(self, device_id: str) -> bool:
        """Remove a device from the registry.

        Args:
            device_id: The device to remove.

        Returns:
            True if a row was deleted, False if device was not registered.
        """
        raise NotImplementedError

    # ========== Merkle Hash Operations ==========

    async def compute_merkle_root(self, entity_type: str, *, is_pro: bool = False) -> str | None:
        """Compute and cache the Merkle root hash for an entity type."""
        raise NotImplementedError

    async def get_merkle_tree(self, entity_type: str, *, is_pro: bool = False) -> dict[str, str]:
        """Return cached {prefix: hash} map for an entity type."""
        raise NotImplementedError

    async def invalidate_merkle_prefix(
        self, entity_type: str, entity_id: str, *, is_pro: bool = False
    ) -> None:
        """Delete cached hashes for the bucket containing entity_id."""
        raise NotImplementedError

    async def get_merkle_root(self, *, is_pro: bool = False) -> str | None:
        """Get combined root hash across all entity types."""
        raise NotImplementedError

    async def get_bucket_entity_ids(
        self, entity_type: str, prefix: str, *, is_pro: bool = False
    ) -> list[str]:
        """Return all entity IDs in the given bucket prefix for delete detection."""
        raise NotImplementedError

    # ========== Alert Operations ==========

    def _get_brain_id(self) -> str:
        """Return current brain ID, raising ValueError if not set."""
        raise NotImplementedError

    async def record_alert(self, alert: Alert) -> str:
        """Insert a new alert. Returns alert ID if inserted, empty string if suppressed."""
        raise NotImplementedError

    async def get_active_alerts(self, limit: int = 50) -> list[Alert]:
        """Get active/seen/acknowledged alerts (not resolved)."""
        raise NotImplementedError

    async def count_pending_alerts(self) -> int:
        """Count active + seen alerts (not acknowledged or resolved)."""
        raise NotImplementedError

    async def mark_alerts_seen(self, alert_ids: list[str]) -> int:
        """Mark alerts as seen. Returns count of updated rows."""
        raise NotImplementedError

    async def mark_alert_acknowledged(self, alert_id: str) -> bool:
        """Mark a single alert as acknowledged. Returns True if updated."""
        raise NotImplementedError

    async def resolve_alerts_by_type(self, alert_types: list[str]) -> int:
        """Resolve all active/seen alerts of given types. Returns count."""
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

    # ========== Cognitive State (optional, provided by SQLiteCognitiveMixin) ==========

    async def upsert_cognitive_state(
        self,
        neuron_id: str,
        *,
        confidence: float = 0.5,
        evidence_for_count: int = 0,
        evidence_against_count: int = 0,
        status: str = "active",
        predicted_at: str | None = None,
        resolved_at: str | None = None,
        schema_version: int = 1,
        parent_schema_id: str | None = None,
        last_evidence_at: str | None = None,
    ) -> None:
        """Insert or update a cognitive state record."""
        raise NotImplementedError

    async def get_cognitive_state(self, neuron_id: str) -> dict[str, Any] | None:
        """Get cognitive state for a neuron."""
        raise NotImplementedError

    async def list_cognitive_states(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List cognitive states, optionally filtered by status."""
        raise NotImplementedError

    async def update_cognitive_evidence(
        self,
        neuron_id: str,
        *,
        confidence: float,
        evidence_for_count: int,
        evidence_against_count: int,
        status: str,
        resolved_at: str | None = None,
        last_evidence_at: str | None = None,
    ) -> None:
        """Update only evidence-related fields of a cognitive state."""
        raise NotImplementedError

    async def list_predictions(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List predictions (cognitive states with predicted_at set)."""
        raise NotImplementedError

    async def get_calibration_stats(self) -> dict[str, int]:
        """Get prediction calibration statistics."""
        raise NotImplementedError

    async def refresh_hot_index(self, items: list[dict[str, Any]]) -> int:
        """Replace the hot index with freshly scored items."""
        raise NotImplementedError

    async def get_hot_index(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the current hot index items."""
        raise NotImplementedError

    async def add_knowledge_gap(
        self,
        *,
        topic: str,
        detection_source: str,
        priority: float = 0.5,
        related_neuron_ids: list[str] | None = None,
    ) -> str:
        """Create a new knowledge gap record."""
        raise NotImplementedError

    async def list_knowledge_gaps(
        self,
        *,
        include_resolved: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List knowledge gaps."""
        raise NotImplementedError

    async def get_knowledge_gap(self, gap_id: str) -> dict[str, Any] | None:
        """Get a single knowledge gap by ID."""
        raise NotImplementedError

    async def resolve_knowledge_gap(
        self,
        gap_id: str,
        *,
        resolved_by_neuron_id: str | None = None,
    ) -> bool:
        """Mark a knowledge gap as resolved."""
        raise NotImplementedError

    async def get_schema_history(
        self,
        neuron_id: str,
        *,
        max_depth: int = 20,
    ) -> list[dict[str, Any]]:
        """Walk the version chain for a hypothesis."""
        raise NotImplementedError

    # ========== Source Registry ==========

    async def add_source(self, source: Any) -> str:
        """Insert a source record. Returns the source ID."""
        raise NotImplementedError

    async def get_source(self, source_id: str) -> Any:
        """Get a source by ID within the current brain."""
        raise NotImplementedError

    async def list_sources(
        self,
        source_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """List sources for the current brain."""
        raise NotImplementedError

    async def update_source(
        self,
        source_id: str,
        status: str | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a source. Returns True if modified."""
        raise NotImplementedError

    async def delete_source(self, source_id: str) -> bool:
        """Delete a source. Returns True if deleted."""
        raise NotImplementedError

    async def count_neurons_for_source(self, source_id: str) -> int:
        """Count neurons linked to a source."""
        raise NotImplementedError

    async def find_source_by_name(self, name: str) -> Any:
        """Find a source by exact name."""
        raise NotImplementedError
