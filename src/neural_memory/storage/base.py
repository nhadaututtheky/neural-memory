"""Abstract base class for neural storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from neural_memory.core.brain import Brain, BrainSnapshot
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
    from neural_memory.core.synapse import Synapse, SynapseType


class NeuralStorage(ABC):
    """
    Abstract interface for neural graph storage.

    Implementations must provide all methods for storing and
    retrieving neurons, synapses, fibers, and brain metadata.
    """

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

    # ========== Cleanup ==========

    @abstractmethod
    async def clear(self, brain_id: str) -> None:
        """
        Clear all data for a brain.

        Args:
            brain_id: The brain ID to clear
        """
        ...
