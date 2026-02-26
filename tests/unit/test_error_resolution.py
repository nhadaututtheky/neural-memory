"""Tests for Error Resolution Learning.

When a new memory (FACT/INSIGHT/DECISION) contradicts an existing ERROR memory,
the system should create a RESOLVED_BY synapse linking error → resolution,
demote the error fiber's salience, and mark the error as resolved.

This enables agents to recall both the error AND its fix, instead of just
the stale error (which causes agents to stubbornly refuse retrying).
"""

from __future__ import annotations

from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.conflict_detection import (
    Conflict,
    ConflictType,
    resolve_conflicts,
)


class _MockStorage:
    """Minimal mock storage for error resolution tests."""

    def __init__(self) -> None:
        self._neurons: dict[str, Neuron] = {}
        self._states: dict[str, NeuronState] = {}
        self._synapses: list[object] = []
        self._fibers: dict[str, object] = {}

    async def find_neurons(self, **kwargs: object) -> list[Neuron]:
        results = list(self._neurons.values())
        content_contains = kwargs.get("content_contains")
        if content_contains and isinstance(content_contains, str):
            results = [n for n in results if content_contains.lower() in n.content.lower()]
        limit = kwargs.get("limit", 100)
        if isinstance(limit, int):
            results = results[:limit]
        return results

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        return self._neurons.get(neuron_id)

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        return self._states.get(neuron_id)

    async def update_neuron_state(self, state: NeuronState) -> None:
        self._states[state.neuron_id] = state

    async def update_neuron(self, neuron: Neuron) -> None:
        self._neurons[neuron.id] = neuron

    async def add_synapse(self, synapse: object) -> str:
        self._synapses.append(synapse)
        return getattr(synapse, "id", "")

    async def get_fiber_by_anchor(self, anchor_neuron_id: str) -> object | None:
        return self._fibers.get(anchor_neuron_id)

    async def update_fiber(self, fiber: object) -> None:
        anchor_id = getattr(fiber, "anchor_neuron_id", None)
        if anchor_id:
            self._fibers[anchor_id] = fiber

    def add_neuron_for_test(
        self,
        neuron: Neuron,
        state: NeuronState | None = None,
    ) -> None:
        self._neurons[neuron.id] = neuron
        if state is not None:
            self._states[neuron.id] = state

    def add_fiber_for_test(self, fiber: object) -> None:
        anchor_id = getattr(fiber, "anchor_neuron_id", None)
        if anchor_id:
            self._fibers[anchor_id] = fiber


# ========== SynapseType tests ==========


class TestResolvedBySynapseType:
    """RESOLVED_BY should exist as a SynapseType."""

    def test_resolved_by_exists(self) -> None:
        """RESOLVED_BY should be a valid SynapseType."""
        assert hasattr(SynapseType, "RESOLVED_BY")
        assert SynapseType.RESOLVED_BY == "resolved_by"

    def test_resolved_by_is_unidirectional(self) -> None:
        """RESOLVED_BY is a one-way link (fix → error), not in INVERSE_TYPES."""
        from neural_memory.core.synapse import BIDIRECTIONAL_TYPES, INVERSE_TYPES

        # RESOLVED_BY is intentionally not in INVERSE_TYPES
        # (CAUSED_BY already maps to LEADS_TO, can't be symmetric)
        assert SynapseType.RESOLVED_BY not in INVERSE_TYPES
        assert SynapseType.RESOLVED_BY not in BIDIRECTIONAL_TYPES


# ========== Error Resolution in resolve_conflicts ==========


class TestErrorResolutionLearning:
    """When new memory contradicts an ERROR, create RESOLVED_BY synapse."""

    async def test_creates_resolved_by_when_error_contradicted(self) -> None:
        """When a FACT contradicts an ERROR, should create RESOLVED_BY synapse."""
        storage = _MockStorage()
        error_neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="ImportError: module 'foo' has no attribute 'bar'",
            metadata={"type": "error", "is_anchor": True},
            neuron_id="error-1",
        )
        state = NeuronState(neuron_id="error-1", activation_level=0.85)
        storage.add_neuron_for_test(error_neuron, state)

        conflict = Conflict(
            type=ConflictType.FACTUAL_CONTRADICTION,
            existing_neuron_id="error-1",
            existing_content="ImportError: module 'foo' has no attribute 'bar'",
            new_content="Module 'foo' works correctly after upgrading to v2.0",
            confidence=0.8,
        )

        resolutions = await resolve_conflicts(
            [conflict],
            new_neuron_id="fix-1",
            storage=storage,
            existing_memory_type="error",
        )

        assert len(resolutions) == 1
        # Should have RESOLVED_BY synapse (in addition to CONTRADICTS)
        all_synapses = storage._synapses
        synapse_types = {getattr(s, "type", None) for s in all_synapses}
        assert SynapseType.RESOLVED_BY in synapse_types

        # Find the RESOLVED_BY synapse and verify direction
        resolved_by = [
            s for s in all_synapses if getattr(s, "type", None) == SynapseType.RESOLVED_BY
        ]
        assert len(resolved_by) == 1
        # Direction: fix → error (fix RESOLVED_BY the error)
        assert resolved_by[0].source_id == "fix-1"
        assert resolved_by[0].target_id == "error-1"

    async def test_marks_error_as_resolved(self) -> None:
        """Resolved errors should have _conflict_resolved metadata."""
        storage = _MockStorage()
        error_neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="ConnectionError: database timeout",
            metadata={"type": "error", "is_anchor": True},
            neuron_id="error-1",
        )
        state = NeuronState(neuron_id="error-1", activation_level=0.7)
        storage.add_neuron_for_test(error_neuron, state)

        conflict = Conflict(
            type=ConflictType.FACTUAL_CONTRADICTION,
            existing_neuron_id="error-1",
            existing_content="ConnectionError: database timeout",
            new_content="Database connection works after increasing pool size",
            confidence=0.8,
        )

        await resolve_conflicts(
            [conflict],
            new_neuron_id="fix-1",
            storage=storage,
            existing_memory_type="error",
        )

        updated = await storage.get_neuron("error-1")
        assert updated is not None
        assert updated.metadata.get("_conflict_resolved") is True
        assert updated.metadata.get("_resolved_by") == "fix-1"

    async def test_no_resolved_by_for_non_error_conflicts(self) -> None:
        """Normal conflicts (non-error) should NOT create RESOLVED_BY synapse."""
        storage = _MockStorage()
        existing = Neuron.create(
            type=NeuronType.CONCEPT,
            content="We use PostgreSQL for our database",
            metadata={"type": "decision", "is_anchor": True},
            neuron_id="existing-1",
        )
        state = NeuronState(neuron_id="existing-1", activation_level=0.8)
        storage.add_neuron_for_test(existing, state)

        conflict = Conflict(
            type=ConflictType.FACTUAL_CONTRADICTION,
            existing_neuron_id="existing-1",
            existing_content="We use PostgreSQL for our database",
            new_content="We use MySQL for our database",
            confidence=0.8,
        )

        await resolve_conflicts(
            [conflict],
            new_neuron_id="new-1",
            storage=storage,
            existing_memory_type="decision",
        )

        all_synapses = storage._synapses
        resolved_by = [
            s for s in all_synapses if getattr(s, "type", None) == SynapseType.RESOLVED_BY
        ]
        assert len(resolved_by) == 0

    async def test_error_activation_strongly_demoted(self) -> None:
        """Error neuron activation should be more aggressively reduced than normal conflicts."""
        storage = _MockStorage()
        error_neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="RuntimeError: cannot connect to API",
            metadata={"type": "error", "is_anchor": True},
            neuron_id="error-1",
        )
        state = NeuronState(neuron_id="error-1", activation_level=0.85)
        storage.add_neuron_for_test(error_neuron, state)

        conflict = Conflict(
            type=ConflictType.FACTUAL_CONTRADICTION,
            existing_neuron_id="error-1",
            existing_content="RuntimeError: cannot connect to API",
            new_content="API connection restored after fixing auth token",
            confidence=0.9,
        )

        await resolve_conflicts(
            [conflict],
            new_neuron_id="fix-1",
            storage=storage,
            existing_memory_type="error",
        )

        updated_state = await storage.get_neuron_state("error-1")
        assert updated_state is not None
        # Error should be strongly demoted (at least 50% reduction)
        assert updated_state.activation_level <= 0.85 * 0.5

    async def test_resolved_by_synapse_has_resolution_metadata(self) -> None:
        """RESOLVED_BY synapse should include error_resolution metadata."""
        storage = _MockStorage()
        error_neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="TypeError: undefined is not a function",
            metadata={"type": "error", "is_anchor": True},
            neuron_id="error-1",
        )
        state = NeuronState(neuron_id="error-1", activation_level=0.7)
        storage.add_neuron_for_test(error_neuron, state)

        conflict = Conflict(
            type=ConflictType.FACTUAL_CONTRADICTION,
            existing_neuron_id="error-1",
            existing_content="TypeError: undefined is not a function",
            new_content="Fixed by importing the correct module",
            confidence=0.8,
        )

        await resolve_conflicts(
            [conflict],
            new_neuron_id="fix-1",
            storage=storage,
            existing_memory_type="error",
        )

        resolved_by = [
            s for s in storage._synapses if getattr(s, "type", None) == SynapseType.RESOLVED_BY
        ]
        assert len(resolved_by) == 1
        metadata = resolved_by[0].metadata
        assert metadata.get("error_resolution") is True
        assert "resolved_at" in metadata
