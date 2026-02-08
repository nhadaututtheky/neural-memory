"""Tests for workflow suggestion engine."""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.workflow_suggest import WorkflowSuggestion, suggest_next_action
from neural_memory.storage.memory_store import InMemoryStorage

# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """Storage with a brain context, ready for test data."""
    storage = InMemoryStorage()
    brain = Brain.create(name="workflow-test", config=BrainConfig(), owner_id="test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


# ── Data Structure Tests ─────────────────────────────────────────


class TestWorkflowSuggestion:
    """WorkflowSuggestion data structure tests."""

    def test_frozen(self) -> None:
        suggestion = WorkflowSuggestion(
            action_type="edit",
            confidence=0.9,
            source_habit="recall-edit",
            sequential_count=10,
        )
        with pytest.raises(AttributeError):
            suggestion.confidence = 0.5  # type: ignore[misc]

    def test_defaults(self) -> None:
        suggestion = WorkflowSuggestion(action_type="edit", confidence=0.85)
        assert suggestion.source_habit is None
        assert suggestion.sequential_count == 0

    def test_fields(self) -> None:
        suggestion = WorkflowSuggestion(
            action_type="commit",
            confidence=0.95,
            source_habit="edit-commit",
            sequential_count=12,
        )
        assert suggestion.action_type == "commit"
        assert suggestion.confidence == 0.95
        assert suggestion.source_habit == "edit-commit"
        assert suggestion.sequential_count == 12


# ── suggest_next_action Tests ────────────────────────────────────


class TestSuggestNextAction:
    """Tests for suggest_next_action function."""

    async def test_no_action_neuron_returns_empty(self, store: InMemoryStorage) -> None:
        """When no ACTION neuron matches current_action, return empty list."""
        config = BrainConfig(habit_suggestion_min_weight=0.8, habit_suggestion_min_count=5)
        result = await suggest_next_action(store, "nonexistent", config)
        assert result == []

    async def test_no_before_synapses_returns_empty(self, store: InMemoryStorage) -> None:
        """ACTION neuron exists but has no outgoing BEFORE synapses."""
        recall = Neuron.create(type=NeuronType.ACTION, content="recall", neuron_id="n-recall")
        await store.add_neuron(recall)

        config = BrainConfig(habit_suggestion_min_weight=0.8, habit_suggestion_min_count=5)
        result = await suggest_next_action(store, "recall", config)
        assert result == []

    async def test_weight_below_threshold_returns_empty(self, store: InMemoryStorage) -> None:
        """BEFORE synapse exists but weight is below habit_suggestion_min_weight."""
        recall = Neuron.create(type=NeuronType.ACTION, content="recall", neuron_id="n-recall")
        edit = Neuron.create(type=NeuronType.ACTION, content="edit", neuron_id="n-edit")
        await store.add_neuron(recall)
        await store.add_neuron(edit)

        synapse = Synapse.create(
            source_id="n-recall",
            target_id="n-edit",
            type=SynapseType.BEFORE,
            weight=0.5,  # Below 0.8 threshold
            metadata={"sequential_count": 10},
        )
        await store.add_synapse(synapse)

        config = BrainConfig(habit_suggestion_min_weight=0.8, habit_suggestion_min_count=5)
        result = await suggest_next_action(store, "recall", config)
        assert result == []

    async def test_sequential_count_below_threshold_returns_empty(
        self, store: InMemoryStorage
    ) -> None:
        """BEFORE synapse has sufficient weight but sequential_count is too low."""
        recall = Neuron.create(type=NeuronType.ACTION, content="recall", neuron_id="n-recall")
        edit = Neuron.create(type=NeuronType.ACTION, content="edit", neuron_id="n-edit")
        await store.add_neuron(recall)
        await store.add_neuron(edit)

        synapse = Synapse.create(
            source_id="n-recall",
            target_id="n-edit",
            type=SynapseType.BEFORE,
            weight=0.9,
            metadata={"sequential_count": 3},  # Below 5 threshold
        )
        await store.add_synapse(synapse)

        config = BrainConfig(habit_suggestion_min_weight=0.8, habit_suggestion_min_count=5)
        result = await suggest_next_action(store, "recall", config)
        assert result == []

    async def test_valid_suggestion_returned(self, store: InMemoryStorage) -> None:
        """Synapse with weight >= 0.8 and sequential_count >= 5 yields a suggestion."""
        recall = Neuron.create(type=NeuronType.ACTION, content="recall", neuron_id="n-recall")
        edit = Neuron.create(type=NeuronType.ACTION, content="edit", neuron_id="n-edit")
        await store.add_neuron(recall)
        await store.add_neuron(edit)

        synapse = Synapse.create(
            source_id="n-recall",
            target_id="n-edit",
            type=SynapseType.BEFORE,
            weight=0.9,
            metadata={"sequential_count": 10},
        )
        await store.add_synapse(synapse)

        config = BrainConfig(habit_suggestion_min_weight=0.8, habit_suggestion_min_count=5)
        result = await suggest_next_action(store, "recall", config)
        assert len(result) == 1
        assert result[0].action_type == "edit"
        assert result[0].confidence == 0.9
        assert result[0].sequential_count == 10

    async def test_multiple_suggestions_sorted_by_confidence(self, store: InMemoryStorage) -> None:
        """Multiple valid suggestions are returned sorted by confidence descending."""
        recall = Neuron.create(type=NeuronType.ACTION, content="recall", neuron_id="n-recall")
        edit = Neuron.create(type=NeuronType.ACTION, content="edit", neuron_id="n-edit")
        commit = Neuron.create(type=NeuronType.ACTION, content="commit", neuron_id="n-commit")
        push = Neuron.create(type=NeuronType.ACTION, content="push", neuron_id="n-push")
        await store.add_neuron(recall)
        await store.add_neuron(edit)
        await store.add_neuron(commit)
        await store.add_neuron(push)

        # Three BEFORE synapses with different weights
        synapses = [
            Synapse.create(
                source_id="n-recall",
                target_id="n-edit",
                type=SynapseType.BEFORE,
                weight=0.85,
                metadata={"sequential_count": 8},
            ),
            Synapse.create(
                source_id="n-recall",
                target_id="n-commit",
                type=SynapseType.BEFORE,
                weight=0.95,
                metadata={"sequential_count": 15},
            ),
            Synapse.create(
                source_id="n-recall",
                target_id="n-push",
                type=SynapseType.BEFORE,
                weight=0.80,
                metadata={"sequential_count": 6},
            ),
        ]
        for s in synapses:
            await store.add_synapse(s)

        config = BrainConfig(habit_suggestion_min_weight=0.8, habit_suggestion_min_count=5)
        result = await suggest_next_action(store, "recall", config)
        assert len(result) == 3
        assert result[0].action_type == "commit"
        assert result[0].confidence == 0.95
        assert result[1].action_type == "edit"
        assert result[1].confidence == 0.85
        assert result[2].action_type == "push"
        assert result[2].confidence == 0.80

    async def test_source_habit_populated_from_fiber(self, store: InMemoryStorage) -> None:
        """When a habit fiber contains both neurons, source_habit is populated."""
        recall = Neuron.create(type=NeuronType.ACTION, content="recall", neuron_id="n-recall")
        edit = Neuron.create(type=NeuronType.ACTION, content="edit", neuron_id="n-edit")
        await store.add_neuron(recall)
        await store.add_neuron(edit)

        synapse = Synapse.create(
            source_id="n-recall",
            target_id="n-edit",
            type=SynapseType.BEFORE,
            weight=0.9,
            metadata={"sequential_count": 10},
        )
        await store.add_synapse(synapse)

        # Create a fiber that contains both neurons and is a habit pattern
        fiber = Fiber.create(
            neuron_ids={"n-recall", "n-edit"},
            synapse_ids={synapse.id},
            anchor_neuron_id="n-recall",
            summary="recall-edit",
            metadata={"_habit_pattern": True},
            fiber_id="f-habit",
        )
        await store.add_fiber(fiber)

        config = BrainConfig(habit_suggestion_min_weight=0.8, habit_suggestion_min_count=5)
        result = await suggest_next_action(store, "recall", config)
        assert len(result) == 1
        assert result[0].source_habit == "recall-edit"
        assert result[0].action_type == "edit"

    async def test_source_habit_none_without_habit_fiber(self, store: InMemoryStorage) -> None:
        """Without a habit fiber, source_habit should be None."""
        recall = Neuron.create(type=NeuronType.ACTION, content="recall", neuron_id="n-recall")
        edit = Neuron.create(type=NeuronType.ACTION, content="edit", neuron_id="n-edit")
        await store.add_neuron(recall)
        await store.add_neuron(edit)

        synapse = Synapse.create(
            source_id="n-recall",
            target_id="n-edit",
            type=SynapseType.BEFORE,
            weight=0.9,
            metadata={"sequential_count": 10},
        )
        await store.add_synapse(synapse)

        # Fiber exists but is NOT a habit pattern
        fiber = Fiber.create(
            neuron_ids={"n-recall", "n-edit"},
            synapse_ids={synapse.id},
            anchor_neuron_id="n-recall",
            summary="regular-fiber",
            metadata={},
            fiber_id="f-regular",
        )
        await store.add_fiber(fiber)

        config = BrainConfig(habit_suggestion_min_weight=0.8, habit_suggestion_min_count=5)
        result = await suggest_next_action(store, "recall", config)
        assert len(result) == 1
        assert result[0].source_habit is None
