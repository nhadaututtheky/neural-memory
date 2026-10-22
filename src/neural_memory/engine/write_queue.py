"""Deferred write queue for batching non-critical writes after response assembly."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.storage.write_schedule import run_writes

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.neuron import NeuronState
    from neural_memory.core.synapse import Synapse
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


async def _gather_count(
    storage: object,
    coros: list[Any],
    error_label: str,
) -> int:
    """Run writes, return how many succeeded, and log the failures.

    Postgres overlaps the writes. SQLite runs them one at a time so the
    single writer connection is not asked to commit from two tasks at once.
    """

    async def _one(coro: Any) -> int:
        try:
            await coro
        except Exception as exc:
            logger.warning("%s: %s", error_label, exc)
            return 0
        return 1

    counts = await run_writes(storage, [_one(coro) for coro in coros])
    return sum(counts)


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
        self._co_activation_records: list[tuple[str, str, float, str | None]] = []

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

    def defer_co_activation(
        self,
        neuron_a: str,
        neuron_b: str,
        binding_strength: float,
        source_anchor: str | None = None,
    ) -> None:
        """Queue a co-activation event for later flush."""
        self._co_activation_records.append((neuron_a, neuron_b, binding_strength, source_anchor))

    @property
    def pending_count(self) -> int:
        """Number of pending writes."""
        return (
            len(self._fiber_updates)
            + len(self._synapse_updates)
            + len(self._synapse_creates)
            + len(self._state_updates)
            + len(self._co_activation_records)
        )

    async def flush(self, storage: NeuralStorage) -> int:
        """Flush all pending writes to storage.

        Categories run one after another (creates before updates). Inside a
        category, Postgres overlaps the writes and SQLite runs them one at
        a time.

        Args:
            storage: Storage backend to write to

        Returns:
            Count of items successfully written
        """
        count = 0

        # Fiber updates — parallel
        if self._fiber_updates:
            count += await _gather_count(
                storage,
                [storage.update_fiber(f) for f in self._fiber_updates],
                "Deferred fiber update failed",
            )

        # Synapse creates — parallel (before updates to avoid update-before-create)
        if self._synapse_creates:
            count += await _gather_count(
                storage,
                [storage.add_synapse(s) for s in self._synapse_creates],
                "Deferred synapse create failed",
            )

        # Synapse updates — parallel
        if self._synapse_updates:
            count += await _gather_count(
                storage,
                [storage.update_synapse(s) for s in self._synapse_updates],
                "Deferred synapse update failed",
            )

        # Neuron state updates — parallel
        if self._state_updates:
            count += await _gather_count(
                storage,
                [storage.update_neuron_state(s) for s in self._state_updates],
                "Deferred state update failed",
            )

        # Co-activation records — parallel
        if self._co_activation_records:
            count += await _gather_count(
                storage,
                [
                    storage.record_co_activation(a, b, strength, anchor)
                    for a, b, strength, anchor in self._co_activation_records
                ],
                "Deferred co-activation record failed",
            )

        self.clear()
        return count

    def clear(self) -> None:
        """Discard all pending writes."""
        self._fiber_updates.clear()
        self._synapse_updates.clear()
        self._synapse_creates.clear()
        self._state_updates.clear()
        self._co_activation_records.clear()
