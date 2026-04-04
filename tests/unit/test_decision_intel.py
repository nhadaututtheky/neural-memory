"""Tests for decision intelligence — component extraction and overlap scoring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.decision_intel import (
    DecisionComponents,
    DecisionOverlap,
    _classify_relationship,
    _token_overlap_score,
    extract_decision_components,
    find_overlapping_decisions,
)

# ── extract_decision_components ──


class TestExtractFromContext:
    """Extraction from structured context dict."""

    def test_full_context(self) -> None:
        result = extract_decision_components(
            "Chose FastAPI",
            context={
                "chosen": "FastAPI",
                "alternatives": ["Flask", "Django"],
                "reason": "Better async support",
                "confidence": "high",
                "tags": ["python", "web"],
                "project": "myapi",
            },
        )
        assert result is not None
        assert result.chosen == "FastAPI"
        assert result.rejected_alternatives == ["Flask", "Django"]
        assert result.reasoning == "Better async support"
        assert result.confidence == "high"
        assert "python" in result.context_tags
        assert "myapi" in result.context_tags

    def test_minimal_context(self) -> None:
        result = extract_decision_components(
            "Chose X",
            context={"chosen": "PostgreSQL"},
        )
        assert result is not None
        assert result.chosen == "PostgreSQL"
        assert result.rejected_alternatives == []
        assert result.reasoning == ""

    def test_alternatives_as_string(self) -> None:
        result = extract_decision_components(
            "",
            context={"chosen": "A", "alternatives": "B, C, D"},
        )
        assert result is not None
        assert result.rejected_alternatives == ["B", "C", "D"]

    def test_rejected_key_alias(self) -> None:
        result = extract_decision_components(
            "",
            context={"chosen": "React", "rejected": ["Vue", "Svelte"]},
        )
        assert result is not None
        assert result.rejected_alternatives == ["Vue", "Svelte"]

    def test_empty_chosen_returns_none(self) -> None:
        result = extract_decision_components(
            "",
            context={"chosen": "", "reason": "whatever"},
        )
        assert result is None

    def test_none_context_falls_through(self) -> None:
        """None context should fall through to heuristic extraction."""
        result = extract_decision_components(
            "chose FastAPI over Flask",
            context=None,
        )
        assert result is not None
        assert result.chosen == "FastAPI"


class TestExtractFromContent:
    """Extraction via regex heuristics on content."""

    def test_chose_over_pattern(self) -> None:
        result = extract_decision_components(
            "Chose Postgres over MySQL because of JSONB support.",
        )
        assert result is not None
        assert result.chosen == "Postgres"
        assert "MySQL" in result.rejected_alternatives

    def test_decided_because_pattern(self) -> None:
        result = extract_decision_components(
            "Decided to use TypeScript because it catches bugs at compile time.",
        )
        assert result is not None
        assert "TypeScript" in result.chosen or "use TypeScript" in result.chosen
        assert "catches bugs" in result.reasoning

    def test_going_with_pattern(self) -> None:
        result = extract_decision_components(
            "Going with Redis instead of Memcached.",
        )
        assert result is not None
        assert result.chosen == "Redis"
        assert "Memcached" in result.rejected_alternatives

    def test_rejected_pattern(self) -> None:
        result = extract_decision_components(
            "Rejected MongoDB because of consistency issues.",
        )
        assert result is not None
        assert "MongoDB" in result.rejected_alternatives
        assert "consistency" in result.reasoning

    def test_no_decision_returns_none(self) -> None:
        result = extract_decision_components(
            "The weather is nice today.",
        )
        assert result is None

    def test_empty_content_returns_none(self) -> None:
        result = extract_decision_components("")
        assert result is None


# ── DecisionComponents ──


class TestDecisionComponents:
    def test_to_dict(self) -> None:
        dc = DecisionComponents(
            chosen="X",
            rejected_alternatives=["Y"],
            reasoning="faster",
            confidence="high",
            context_tags=["web"],
        )
        d = dc.to_dict()
        assert d["chosen"] == "X"
        assert d["rejected_alternatives"] == ["Y"]
        assert d["confidence"] == "high"

    def test_frozen(self) -> None:
        dc = DecisionComponents(chosen="X")
        with pytest.raises(AttributeError):
            dc.chosen = "Y"  # type: ignore[misc]


# ── Overlap scoring helpers ──


class TestTokenOverlapScore:
    def test_identical(self) -> None:
        score = _token_overlap_score("fastapi", set(), "fastapi", set())
        assert score > 0.5

    def test_no_overlap(self) -> None:
        score = _token_overlap_score("redis", set(), "postgres", set())
        assert score == 0.0

    def test_empty_both(self) -> None:
        score = _token_overlap_score("", set(), "", set())
        assert score == 0.0


class TestClassifyRelationship:
    def test_confirms_same_chosen(self) -> None:
        rel = _classify_relationship("fastapi", set(), "fastapi", set())
        assert rel == "confirms"

    def test_contradicts_new_was_rejected(self) -> None:
        rel = _classify_relationship("flask", set(), "fastapi", {"flask"})
        assert rel == "contradicts"

    def test_contradicts_old_now_rejected(self) -> None:
        rel = _classify_relationship("fastapi", {"flask"}, "flask", set())
        assert rel == "contradicts"

    def test_evolves_different_chosen(self) -> None:
        rel = _classify_relationship("fastapi", set(), "django", set())
        assert rel == "evolves"

    def test_evolves_empty_chosen(self) -> None:
        rel = _classify_relationship("", set(), "django", set())
        assert rel == "evolves"


# ── find_overlapping_decisions (async) ──


class TestFindOverlappingDecisions:
    @pytest.fixture
    def storage(self) -> AsyncMock:
        return AsyncMock()

    async def test_no_existing_decisions(self, storage: AsyncMock) -> None:
        storage.find_typed_memories = AsyncMock(return_value=[])
        result = await find_overlapping_decisions(
            storage,
            DecisionComponents(chosen="X"),
        )
        assert result == []

    async def test_finds_confirming_overlap(self, storage: AsyncMock) -> None:
        # Create a mock existing decision
        mem = MagicMock()
        mem.fiber_id = "fiber-1"
        mem.tags = {"python", "web"}
        storage.find_typed_memories = AsyncMock(return_value=[mem])

        fiber = MagicMock()
        fiber.anchor_neuron_id = "neuron-1"
        storage.get_fiber = AsyncMock(return_value=fiber)

        neuron = MagicMock()
        neuron.content = "Chose FastAPI for the web server"
        neuron.metadata = {
            "_decision": {
                "chosen": "FastAPI",
                "rejected_alternatives": ["Flask"],
                "context_tags": ["python", "web"],
            }
        }
        storage.get_neuron = AsyncMock(return_value=neuron)

        result = await find_overlapping_decisions(
            storage,
            DecisionComponents(
                chosen="FastAPI",
                context_tags=["python", "web"],
            ),
            new_tags={"python", "web"},
        )
        assert len(result) >= 1
        assert result[0].relationship == "confirms"
        assert result[0].fiber_id == "fiber-1"

    async def test_finds_contradicting_overlap(self, storage: AsyncMock) -> None:
        mem = MagicMock()
        mem.fiber_id = "fiber-2"
        mem.tags = {"db"}
        storage.find_typed_memories = AsyncMock(return_value=[mem])

        fiber = MagicMock()
        fiber.anchor_neuron_id = "neuron-2"
        storage.get_fiber = AsyncMock(return_value=fiber)

        neuron = MagicMock()
        neuron.content = "Chose Postgres, rejected MySQL"
        neuron.metadata = {
            "_decision": {
                "chosen": "Postgres",
                "rejected_alternatives": ["MySQL"],
                "context_tags": ["db"],
            }
        }
        storage.get_neuron = AsyncMock(return_value=neuron)

        result = await find_overlapping_decisions(
            storage,
            DecisionComponents(
                chosen="MySQL",  # was previously rejected!
                context_tags=["db"],
            ),
            new_tags={"db"},
        )
        assert len(result) >= 1
        assert result[0].relationship == "contradicts"

    async def test_skips_missing_fiber(self, storage: AsyncMock) -> None:
        mem = MagicMock()
        mem.fiber_id = "missing"
        mem.tags = set()
        storage.find_typed_memories = AsyncMock(return_value=[mem])
        storage.get_fiber = AsyncMock(return_value=None)

        result = await find_overlapping_decisions(
            storage,
            DecisionComponents(chosen="X"),
        )
        assert result == []


# ── DecisionOverlap ──


class TestDecisionOverlap:
    def test_to_dict(self) -> None:
        ov = DecisionOverlap(
            fiber_id="f1",
            content_preview="Chose X",
            overlap_score=0.756,
            relationship="confirms",
        )
        d = ov.to_dict()
        assert d["fiber_id"] == "f1"
        assert d["overlap_score"] == 0.756
        assert d["relationship"] == "confirms"


# ── EVOLVES_FROM synapse type ──


class TestSynapseType:
    def test_evolves_from_exists(self) -> None:
        from neural_memory.core.synapse import SynapseType

        assert SynapseType.EVOLVES_FROM == "evolves_from"
