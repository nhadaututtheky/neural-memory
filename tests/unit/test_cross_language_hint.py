"""Tests for cross-language recall hint feature.

When recall returns 0 results and query language differs from brain majority,
a hint should suggest enabling embedding for cross-language recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from neural_memory.extraction.parser import detect_language

# ──────────────────── detect_language tests ────────────────────


class TestDetectLanguage:
    """Test the module-level detect_language function."""

    def test_english_text(self) -> None:
        assert detect_language("Fixed authentication bug in login module") == "en"

    def test_vietnamese_text_diacritics(self) -> None:
        assert detect_language("Lỗi xác thực trong module đăng nhập cần phải sửa ngay") == "vi"

    def test_vietnamese_text_keywords(self) -> None:
        assert detect_language("có lỗi trong login") == "vi"

    def test_short_english(self) -> None:
        assert detect_language("auth bug") == "en"

    def test_short_vietnamese(self) -> None:
        # Short text with Vietnamese keywords
        assert detect_language("lỗi của hệ thống") == "vi"

    def test_mixed_defaults_english(self) -> None:
        # No Vietnamese signals → defaults to English
        assert detect_language("fix bug") == "en"

    def test_empty_string(self) -> None:
        assert detect_language("") == "en"

    def test_code_only(self) -> None:
        assert detect_language("def main(): pass") == "en"


# ──────────────────── _check_cross_language_hint tests ────────────────────


@dataclass
class FakeRetrievalResult:
    """Minimal fake for RetrievalResult."""

    fibers_matched: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class FakeNeuron:
    """Minimal fake neuron with content."""

    id: str = "n1"
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeConfig:
    """Minimal fake BrainConfig."""

    embedding_enabled: bool = False


class FakeToolHandler:
    """Minimal stand-in that has _check_cross_language_hint."""

    async def get_storage(self) -> Any:
        return self._storage

    def __init__(self, neurons: list[FakeNeuron]) -> None:
        self._storage = AsyncMock()
        self._storage.find_neurons = AsyncMock(return_value=neurons)


async def _call_hint(
    handler: FakeToolHandler,
    query: str,
    result: FakeRetrievalResult,
    config: FakeConfig,
) -> str | None:
    """Import and call the actual method."""
    from neural_memory.mcp.tool_handlers import ToolHandler

    # Bind the real method to our fake handler
    bound = ToolHandler._check_cross_language_hint.__get__(handler, FakeToolHandler)
    return await bound(query, result, config)


class TestCrossLanguageHint:
    """Test _check_cross_language_hint behavior."""

    @pytest.mark.asyncio
    async def test_hint_shown_on_language_mismatch(self) -> None:
        """Vietnamese query, English brain, 0 results → hint shown."""
        neurons = [
            FakeNeuron(id=f"n{i}", content=f"Fixed bug number {i} in auth module") for i in range(5)
        ]
        handler = FakeToolHandler(neurons)
        result = FakeRetrievalResult(fibers_matched=[], confidence=0.0)
        config = FakeConfig(embedding_enabled=False)

        hint = await _call_hint(handler, "lỗi xác thực", result, config)

        assert hint is not None
        assert "Vietnamese" in hint
        assert "English" in hint
        assert "embedding" in hint.lower()

    @pytest.mark.asyncio
    async def test_no_hint_when_embedding_enabled(self) -> None:
        """No hint if embedding is already enabled."""
        neurons = [FakeNeuron(id=f"n{i}", content=f"Fixed bug {i}") for i in range(5)]
        handler = FakeToolHandler(neurons)
        result = FakeRetrievalResult(fibers_matched=[], confidence=0.0)
        config = FakeConfig(embedding_enabled=True)

        hint = await _call_hint(handler, "lỗi xác thực", result, config)

        assert hint is None

    @pytest.mark.asyncio
    async def test_no_hint_when_results_found(self) -> None:
        """No hint when recall found fibers with good confidence."""
        neurons = [FakeNeuron(id=f"n{i}", content=f"Fixed bug {i}") for i in range(5)]
        handler = FakeToolHandler(neurons)
        result = FakeRetrievalResult(fibers_matched=["f1", "f2"], confidence=0.5)
        config = FakeConfig(embedding_enabled=False)

        hint = await _call_hint(handler, "lỗi xác thực", result, config)

        assert hint is None

    @pytest.mark.asyncio
    async def test_no_hint_same_language(self) -> None:
        """No hint when query and brain are same language."""
        neurons = [FakeNeuron(id=f"n{i}", content=f"Fixed auth bug {i}") for i in range(5)]
        handler = FakeToolHandler(neurons)
        result = FakeRetrievalResult(fibers_matched=[], confidence=0.0)
        config = FakeConfig(embedding_enabled=False)

        hint = await _call_hint(handler, "auth bug", result, config)

        assert hint is None

    @pytest.mark.asyncio
    async def test_no_hint_too_few_neurons(self) -> None:
        """No hint when brain has fewer than 3 neurons (too early to tell)."""
        neurons = [FakeNeuron(id="n1", content="Hello")]
        handler = FakeToolHandler(neurons)
        result = FakeRetrievalResult(fibers_matched=[], confidence=0.0)
        config = FakeConfig(embedding_enabled=False)

        hint = await _call_hint(handler, "lỗi xác thực", result, config)

        assert hint is None

    @pytest.mark.asyncio
    async def test_hint_english_query_vietnamese_brain(self) -> None:
        """English query, Vietnamese brain → hint shown."""
        neurons = [
            FakeNeuron(id=f"n{i}", content=f"Lỗi xác thực trong module số {i}") for i in range(5)
        ]
        handler = FakeToolHandler(neurons)
        result = FakeRetrievalResult(fibers_matched=[], confidence=0.1)
        config = FakeConfig(embedding_enabled=False)

        hint = await _call_hint(handler, "authentication error", result, config)

        assert hint is not None
        assert "English" in hint
        assert "Vietnamese" in hint

    @pytest.mark.asyncio
    async def test_hint_includes_pip_install_when_not_installed(self) -> None:
        """Hint mentions pip install when sentence-transformers not installed."""
        neurons = [FakeNeuron(id=f"n{i}", content=f"Fixed bug {i} in module") for i in range(5)]
        handler = FakeToolHandler(neurons)
        result = FakeRetrievalResult(fibers_matched=[], confidence=0.0)
        config = FakeConfig(embedding_enabled=False)

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            hint = await _call_hint(handler, "lỗi xác thực", result, config)

        assert hint is not None
        assert "pip install" in hint

    @pytest.mark.asyncio
    async def test_hint_omits_pip_install_when_installed(self) -> None:
        """Hint omits pip install when sentence-transformers is available."""
        neurons = [FakeNeuron(id=f"n{i}", content=f"Fixed bug {i} in module") for i in range(5)]
        handler = FakeToolHandler(neurons)
        result = FakeRetrievalResult(fibers_matched=[], confidence=0.0)
        config = FakeConfig(embedding_enabled=False)

        # Mock sentence_transformers as importable
        import types

        fake_module = types.ModuleType("sentence_transformers")
        with patch.dict("sys.modules", {"sentence_transformers": fake_module}):
            hint = await _call_hint(handler, "lỗi xác thực", result, config)

        assert hint is not None
        assert "pip install" not in hint

    @pytest.mark.asyncio
    async def test_hint_low_confidence_triggers(self) -> None:
        """Hint shown when confidence is low (< 0.3) even with some fibers."""
        neurons = [FakeNeuron(id=f"n{i}", content=f"Fixed English bug {i}") for i in range(5)]
        handler = FakeToolHandler(neurons)
        # Has fibers but very low confidence
        result = FakeRetrievalResult(fibers_matched=[], confidence=0.1)
        config = FakeConfig(embedding_enabled=False)

        hint = await _call_hint(handler, "lỗi trong hệ thống", result, config)

        assert hint is not None

    @pytest.mark.asyncio
    async def test_storage_error_returns_none(self) -> None:
        """Gracefully handle storage errors."""
        handler = FakeToolHandler([])
        handler._storage.find_neurons = AsyncMock(side_effect=RuntimeError("db error"))
        result = FakeRetrievalResult(fibers_matched=[], confidence=0.0)
        config = FakeConfig(embedding_enabled=False)

        hint = await _call_hint(handler, "lỗi xác thực", result, config)

        assert hint is None
