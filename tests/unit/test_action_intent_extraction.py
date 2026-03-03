"""Tests for ACTION and INTENT neuron extraction (Issue #13 — diversity)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.neuron import NeuronType
from neural_memory.engine.pipeline_steps import (
    ExtractActionNeuronsStep,
    ExtractIntentNeuronsStep,
    _action_pattern,
    _intent_pattern,
)


def _make_ctx(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        language="en",
        action_neurons=[],
        intent_neurons=[],
        neurons_created=[],
    )


class TestActionPattern:
    def test_matches_decided(self) -> None:
        matches = _action_pattern().findall("I decided to use FastAPI for the backend.")
        assert len(matches) >= 1
        assert "use FastAPI for the backend" in matches[0]

    def test_matches_implemented(self) -> None:
        matches = _action_pattern().findall("We implemented the caching layer.")
        assert len(matches) >= 1

    def test_matches_deployed(self) -> None:
        matches = _action_pattern().findall("deployed the v2 release to production")
        assert len(matches) >= 1

    def test_no_match_plain_text(self) -> None:
        matches = _action_pattern().findall("The weather is nice today")
        assert len(matches) == 0


class TestIntentPattern:
    def test_matches_want_to(self) -> None:
        matches = _intent_pattern().findall("I want to improve test coverage.")
        assert len(matches) >= 1

    def test_matches_plan_to(self) -> None:
        matches = _intent_pattern().findall("We plan to migrate to PostgreSQL.")
        assert len(matches) >= 1

    def test_matches_goal_prefix(self) -> None:
        matches = _intent_pattern().findall("goal: reduce orphan rate below 20%")
        assert len(matches) >= 1

    def test_no_match_plain_text(self) -> None:
        matches = _intent_pattern().findall("The system runs on port 8090")
        assert len(matches) == 0


class TestExtractActionNeuronsStep:
    @pytest.mark.asyncio
    async def test_extracts_action_neurons(self) -> None:
        step = ExtractActionNeuronsStep()
        storage = AsyncMock()
        config = AsyncMock()

        ctx = _make_ctx("I decided to use Redis. Then implemented the cache layer.")
        result = await step.execute(ctx, storage, config)

        assert len(result.action_neurons) >= 1
        assert all(n.type == NeuronType.ACTION for n in result.action_neurons)
        assert len(result.neurons_created) == len(result.action_neurons)

    @pytest.mark.asyncio
    async def test_deduplicates_within_encode(self) -> None:
        step = ExtractActionNeuronsStep()
        storage = AsyncMock()
        config = AsyncMock()

        ctx = _make_ctx("Fixed the bug. Then fixed the bug again.")
        result = await step.execute(ctx, storage, config)

        contents = [n.content.lower() for n in result.action_neurons]
        assert len(contents) == len(set(contents))

    @pytest.mark.asyncio
    async def test_caps_at_max_actions(self) -> None:
        step = ExtractActionNeuronsStep()
        storage = AsyncMock()
        config = AsyncMock()

        actions = ". ".join(f"implemented feature{i}" for i in range(10))
        ctx = _make_ctx(actions)
        result = await step.execute(ctx, storage, config)

        assert len(result.action_neurons) <= step.MAX_ACTIONS

    @pytest.mark.asyncio
    async def test_no_actions_no_neurons(self) -> None:
        step = ExtractActionNeuronsStep()
        storage = AsyncMock()
        config = AsyncMock()

        ctx = _make_ctx("The system uses Python 3.11.")
        result = await step.execute(ctx, storage, config)

        assert len(result.action_neurons) == 0
        storage.add_neuron.assert_not_called()


class TestExtractIntentNeuronsStep:
    @pytest.mark.asyncio
    async def test_extracts_intent_neurons(self) -> None:
        step = ExtractIntentNeuronsStep()
        storage = AsyncMock()
        config = AsyncMock()

        ctx = _make_ctx("I want to improve performance. We need to fix the memory leak.")
        result = await step.execute(ctx, storage, config)

        assert len(result.intent_neurons) >= 1
        assert all(n.type == NeuronType.INTENT for n in result.intent_neurons)

    @pytest.mark.asyncio
    async def test_caps_at_max_intents(self) -> None:
        step = ExtractIntentNeuronsStep()
        storage = AsyncMock()
        config = AsyncMock()

        intents = ". ".join(f"I want to do thing{i}" for i in range(10))
        ctx = _make_ctx(intents)
        result = await step.execute(ctx, storage, config)

        assert len(result.intent_neurons) <= step.MAX_INTENTS

    @pytest.mark.asyncio
    async def test_no_intents_no_neurons(self) -> None:
        step = ExtractIntentNeuronsStep()
        storage = AsyncMock()
        config = AsyncMock()

        ctx = _make_ctx("Redis runs on port 6379.")
        result = await step.execute(ctx, storage, config)

        assert len(result.intent_neurons) == 0
        storage.add_neuron.assert_not_called()
