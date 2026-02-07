"""Deferred write queue for batching non-critical writes after response assembly."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.neuron import NeuronState
    from neural_memory.core.synapse import Synapse
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


class DeferredWriteQueue:
    """Collects non-critical writes and flushes them in batch after response.

    During retrieval, writes like fiber conductivity updates, Hebbian
    strengthening, and reinforcement are deferred until after the response
    is assembled. This moves ~200-500ms of blocking writes to the end.
    """

    def __init__(self) -> None:
        self._fiber_updates: list[Fiber] = []
        self._synapse_updates: list[Synapse] = []
        self._synapse_creates: list[Synapse] = []
        self._state_updates: list[NeuronState] = []

    def defer_fiber_update(self, fiber: Fiber) -> None:
        """Queue a fiber update for later flush."""
        self._fiber_updates.append(fiber)

    def defer_synapse_update(self, synapse: Synapse) -> None:
        """Queue a synapse update for later flush."""
        self._synapse_updates.append(synapse)

    def defer_synapse_create(self, synapse: Synapse) -> None:
        """Queue a synapse creation for later flush."""
        self._synapse_creates.append(synapse)

    def defer_state_update(self, state: NeuronState) -> None:
        """Queue a neuron state update for later flush."""
        self._state_updates.append(state)

    @property
    def pending_count(self) -> int:
        """Number of pending writes."""
        return (
            len(self._fiber_updates)
            + len(self._synapse_updates)
            + len(self._synapse_creates)
            + len(self._state_updates)
        )

    async def flush(self, storage: NeuralStorage) -> int:
        """Flush all pending writes to storage.

        Args:
            storage: Storage backend to write to

        Returns:
            Count of items written
        """
        count = 0

        for fiber in self._fiber_updates:
            try:
                await storage.update_fiber(fiber)
                count += 1
            except Exception:
                logger.debug("Deferred fiber update failed", exc_info=True)

        for synapse in self._synapse_creates:
            try:
                await storage.add_synapse(synapse)
                count += 1
            except Exception:
                logger.debug("Deferred synapse create failed", exc_info=True)

        for synapse in self._synapse_updates:
            try:
                await storage.update_synapse(synapse)
                count += 1
            except Exception:
                logger.debug("Deferred synapse update failed", exc_info=True)

        for state in self._state_updates:
            try:
                await storage.update_neuron_state(state)
                count += 1
            except Exception:
                logger.debug("Deferred state update failed", exc_info=True)

        self.clear()
        return count

    def clear(self) -> None:
        """Discard all pending writes."""
        self._fiber_updates.clear()
        self._synapse_updates.clear()
        self._synapse_creates.clear()
        self._state_updates.clear()
