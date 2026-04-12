"""Tests for proactive memory engine — hint selection and formatting.

Covers:
- ProactiveHint dataclass and serialization
- select_proactive_hints() core logic
- Budget enforcement (max hints, max chars)
- Deduplication against result neurons
- Cold-start (empty priming → no hints)
- Source attribution
- ProactiveConfig defaults and loading
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.priming import PrimingResult
from neural_memory.engine.proactive import (
    ProactiveHint,
    format_hints_for_response,
    select_proactive_hints,
)
from neural_memory.unified_config import ProactiveConfig

# ── ProactiveHint ─────────────────────────────────────────────────────


class TestProactiveHint:
    def test_to_dict_basic(self) -> None:
        hint = ProactiveHint(
            neuron_id="n1",
            content="test content",
            activation_level=0.5,
            source="topic",
        )
        d = hint.to_dict()
        assert d["content"] == "test content"
        assert d["source"] == "topic"
        assert d["activation"] == 0.5
        assert "type" not in d  # no neuron_type set

    def test_to_dict_with_type(self) -> None:
        hint = ProactiveHint(
            neuron_id="n1",
            content="test",
            activation_level=0.7,
            source="habit",
            neuron_type="decision",
        )
        d = hint.to_dict()
        assert d["type"] == "decision"

    def test_frozen(self) -> None:
        hint = ProactiveHint(
            neuron_id="n1",
            content="test",
            activation_level=0.5,
            source="cache",
        )
        with pytest.raises(AttributeError):
            hint.content = "changed"  # type: ignore[misc]


# ── format_hints_for_response ─────────────────────────────────────────


class TestFormatHints:
    def test_format_empty(self) -> None:
        assert format_hints_for_response([]) == []

    def test_format_multiple(self) -> None:
        hints = [
            ProactiveHint("n1", "content1", 0.8, "topic"),
            ProactiveHint("n2", "content2", 0.5, "habit", "insight"),
        ]
        result = format_hints_for_response(hints)
        assert len(result) == 2
        assert result[0]["content"] == "content1"
        assert result[1]["type"] == "insight"


# ── select_proactive_hints ────────────────────────────────────────────


def _make_neuron(neuron_id: str, content: str, neuron_type: str = "fact") -> MagicMock:
    """Create a mock neuron."""
    n = MagicMock()
    n.id = neuron_id
    n.content = content
    n.type = MagicMock(value=neuron_type)
    return n


def _make_storage(*neurons: MagicMock) -> AsyncMock:
    """Create a mock storage that returns neurons by ID."""
    neuron_map = {n.id: n for n in neurons}
    storage = AsyncMock()
    storage.get_neuron = AsyncMock(side_effect=lambda nid: neuron_map.get(nid))
    return storage


class TestSelectProactiveHints:
    @pytest.mark.asyncio
    async def test_empty_priming(self) -> None:
        """Empty priming result produces no hints."""
        result = await select_proactive_hints(
            PrimingResult.empty(),
            _make_storage(),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_basic_hint_selection(self) -> None:
        """Top activated neurons are selected as hints."""
        n1 = _make_neuron("n1", "first memory", "decision")
        n2 = _make_neuron("n2", "second memory", "insight")
        storage = _make_storage(n1, n2)

        priming = PrimingResult(
            activation_boosts={"n1": 0.8, "n2": 0.5},
            source_counts={"topic": 2},
            total_primed=2,
        )

        hints = await select_proactive_hints(priming, storage)
        assert len(hints) == 2
        assert hints[0].neuron_id == "n1"  # higher activation first
        assert hints[0].activation_level == 0.8
        assert hints[1].neuron_id == "n2"

    @pytest.mark.asyncio
    async def test_dedup_against_result(self) -> None:
        """Neurons already in result are excluded."""
        n1 = _make_neuron("n1", "first memory")
        n2 = _make_neuron("n2", "second memory")
        storage = _make_storage(n1, n2)

        priming = PrimingResult(
            activation_boosts={"n1": 0.8, "n2": 0.5},
            source_counts={"topic": 2},
            total_primed=2,
        )

        hints = await select_proactive_hints(
            priming,
            storage,
            result_neuron_ids={"n1"},
        )
        assert len(hints) == 1
        assert hints[0].neuron_id == "n2"

    @pytest.mark.asyncio
    async def test_max_hints_budget(self) -> None:
        """Respects max_hints limit."""
        neurons = [_make_neuron(f"n{i}", f"memory {i}") for i in range(5)]
        storage = _make_storage(*neurons)

        priming = PrimingResult(
            activation_boosts={f"n{i}": 0.9 - i * 0.1 for i in range(5)},
            source_counts={"cache": 5},
            total_primed=5,
        )

        hints = await select_proactive_hints(priming, storage, max_hints=2)
        assert len(hints) == 2

    @pytest.mark.asyncio
    async def test_max_chars_budget(self) -> None:
        """Respects max_chars limit — second hint is truncated or excluded."""
        n1 = _make_neuron("n1", "A" * 300)
        n2 = _make_neuron("n2", "B" * 300)
        storage = _make_storage(n1, n2)

        priming = PrimingResult(
            activation_boosts={"n1": 0.8, "n2": 0.7},
            source_counts={"topic": 2},
            total_primed=2,
        )

        hints = await select_proactive_hints(priming, storage, max_chars=350)
        # First hint takes 300 chars, leaving only 50 for second
        # Second hint gets truncated (300 → ~50 + "...")
        assert len(hints) >= 1
        assert len(hints[0].content) == 300  # first fits fully
        if len(hints) == 2:
            assert len(hints[1].content) <= 53  # truncated to ~50 + "..."

    @pytest.mark.asyncio
    async def test_min_activation_filter(self) -> None:
        """Neurons below min_activation are filtered out."""
        n1 = _make_neuron("n1", "strong")
        n2 = _make_neuron("n2", "weak")
        storage = _make_storage(n1, n2)

        priming = PrimingResult(
            activation_boosts={"n1": 0.5, "n2": 0.1},
            source_counts={"cache": 2},
            total_primed=2,
        )

        hints = await select_proactive_hints(priming, storage, min_activation=0.3)
        assert len(hints) == 1
        assert hints[0].neuron_id == "n1"

    @pytest.mark.asyncio
    async def test_missing_neuron_skipped(self) -> None:
        """Neurons not found in storage are silently skipped."""
        storage = _make_storage()  # empty storage

        priming = PrimingResult(
            activation_boosts={"missing": 0.8},
            source_counts={"topic": 1},
            total_primed=1,
        )

        hints = await select_proactive_hints(priming, storage)
        assert hints == []

    @pytest.mark.asyncio
    async def test_empty_content_skipped(self) -> None:
        """Neurons with empty content are skipped."""
        n1 = _make_neuron("n1", "")
        storage = _make_storage(n1)

        priming = PrimingResult(
            activation_boosts={"n1": 0.8},
            source_counts={"topic": 1},
            total_primed=1,
        )

        hints = await select_proactive_hints(priming, storage)
        assert hints == []


# ── ProactiveConfig ───────────────────────────────────────────────────


class TestProactiveConfig:
    def test_defaults(self) -> None:
        config = ProactiveConfig()
        assert config.enabled is True
        assert config.max_hints == 3
        assert config.max_hint_chars == 500
        assert config.min_activation == 0.3
        assert config.skip_high_confidence == 0.9

    def test_from_dict_partial(self) -> None:
        config = ProactiveConfig.from_dict({"enabled": False, "max_hints": 5})
        assert config.enabled is False
        assert config.max_hints == 5
        assert config.max_hint_chars == 500  # default

    def test_to_dict_roundtrip(self) -> None:
        original = ProactiveConfig(enabled=False, max_hints=5, min_activation=0.5)
        restored = ProactiveConfig.from_dict(original.to_dict())
        assert restored.enabled == original.enabled
        assert restored.max_hints == original.max_hints
        assert restored.min_activation == original.min_activation

    def test_from_empty_dict(self) -> None:
        config = ProactiveConfig.from_dict({})
        assert config.enabled is True
        assert config.max_hints == 3
