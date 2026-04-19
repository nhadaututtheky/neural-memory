"""Tests for Vietnamese keyword extraction and language propagation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from neural_memory.extraction.keywords import (
    STOP_WORDS,
    STOP_WORDS_EN,
    STOP_WORDS_VI,
    WeightedKeyword,
    _detect_vietnamese,
    _get_stop_words,
    extract_keywords,
    extract_weighted_keywords,
)


class TestVietnameseDetection:
    """Tests for Vietnamese language detection."""

    def test_detect_vietnamese_text(self) -> None:
        assert _detect_vietnamese("Hôm nay trời đẹp quá") is True

    def test_detect_english_text(self) -> None:
        assert _detect_vietnamese("The weather is nice today") is False

    def test_detect_vietnamese_diacritics(self) -> None:
        # Characters unique to Vietnamese (not in French)
        assert _detect_vietnamese("ăâđêôơưắ") is True

    def test_detect_empty_text(self) -> None:
        assert _detect_vietnamese("") is False


class TestStopWords:
    """Tests for language-aware stop word selection."""

    def test_combined_stop_words(self) -> None:
        assert STOP_WORDS == STOP_WORDS_EN | STOP_WORDS_VI

    def test_english_stop_words(self) -> None:
        assert "the" in STOP_WORDS_EN
        assert "và" not in STOP_WORDS_EN

    def test_vietnamese_stop_words(self) -> None:
        assert "và" in STOP_WORDS_VI
        assert "the" not in STOP_WORDS_VI

    def test_get_stop_words_en(self) -> None:
        assert _get_stop_words("en", "") == STOP_WORDS_EN

    def test_get_stop_words_vi(self) -> None:
        assert _get_stop_words("vi", "") == STOP_WORDS_VI

    def test_get_stop_words_auto(self) -> None:
        assert _get_stop_words("auto", "") == STOP_WORDS


class TestVietnameseKeywordExtraction:
    """Tests for Vietnamese keyword extraction with pyvi integration."""

    def test_basic_vietnamese_keywords(self) -> None:
        """Test keyword extraction from Vietnamese text without pyvi."""
        text = "Hôm nay trời đẹp"
        keywords = extract_keywords(text, language="vi")
        assert len(keywords) > 0
        # "hôm" and "trời" should be extracted (not stop words)
        kw_texts = [k.lower() for k in keywords]
        assert "trời" in kw_texts or "đẹp" in kw_texts

    def test_language_parameter_backward_compat(self) -> None:
        """Test that extract_keywords works without language param."""
        keywords = extract_keywords("Hello world programming")
        assert len(keywords) > 0

    def test_language_param_weighted(self) -> None:
        """Test that extract_weighted_keywords accepts language param."""
        result = extract_weighted_keywords("Hello world programming", language="en")
        assert all(isinstance(kw, WeightedKeyword) for kw in result)

    @patch("neural_memory.extraction.keywords._tokenize_vietnamese")
    def test_pyvi_tokenization(self, mock_tokenize: MagicMock) -> None:
        """Test that pyvi tokenization produces compound words."""
        # Simulate pyvi joining compound words with underscores
        mock_tokenize.return_value = "học_sinh giỏi nhất trường"

        keywords = extract_keywords("học sinh giỏi nhất trường", language="vi")
        mock_tokenize.assert_called_once()

        # "học sinh" should appear as a single compound keyword
        assert "học sinh" in keywords

    @patch("neural_memory.extraction.keywords._tokenize_vietnamese")
    def test_pyvi_not_available_fallback(self, mock_tokenize: MagicMock) -> None:
        """Test graceful fallback when pyvi is not installed."""
        mock_tokenize.return_value = None

        keywords = extract_keywords("trời đẹp quá", language="vi")
        # Should still extract keywords via regex fallback
        assert len(keywords) > 0

    def test_english_language_explicit(self) -> None:
        """Test explicit English language uses English stop words only."""
        # "của" is a Vietnamese stop word but not English
        keywords = extract_keywords("của programming language", language="en")
        kw_lower = [k.lower() for k in keywords]
        assert "của" in kw_lower  # Not filtered by English stop words

    def test_vietnamese_language_explicit(self) -> None:
        """Test explicit Vietnamese language filters Vietnamese stop words."""
        keywords = extract_keywords("của lập trình ngôn ngữ", language="vi")
        kw_lower = [k.lower() for k in keywords]
        assert "của" not in kw_lower  # Filtered by Vietnamese stop words

    def test_auto_detection_vietnamese(self) -> None:
        """Test auto-detection identifies Vietnamese text."""
        # Text with Vietnamese diacritics should be detected as Vietnamese
        keywords = extract_keywords("Học sinh giỏi nhất trường đại học")
        # Should work without errors
        assert isinstance(keywords, list)

    @patch("neural_memory.extraction.keywords._tokenize_vietnamese")
    def test_compound_word_weight(self, mock_tokenize: MagicMock) -> None:
        """Test that compound words from pyvi get proper weights."""
        mock_tokenize.return_value = "máy_tính xách_tay hiện_đại"

        result = extract_weighted_keywords("máy tính xách tay hiện đại", language="vi")
        mock_tokenize.assert_called_once()

        keyword_texts = {kw.text for kw in result}
        assert "máy tính" in keyword_texts
        assert "xách tay" in keyword_texts

    def test_tokenize_does_not_leak_pyvi_warnings(self) -> None:
        """Issue #132: pyvi emits np.VisibleDeprecationWarning (UserWarning subclass).

        Narrow filters on DeprecationWarning miss it. The scoped suppression inside
        _tokenize_vietnamese must silence ALL warnings during the pyvi import path.
        """
        import warnings as warnings_mod

        try:
            import pyvi  # noqa: F401
        except ImportError:
            pytest.skip("pyvi not installed")

        from neural_memory.extraction.keywords import _tokenize_vietnamese

        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            _tokenize_vietnamese("Hôm nay trời đẹp quá")

        pyvi_warnings = [
            w
            for w in caught
            if "pyvi" in str(w.filename).lower() or "numpy" in str(w.message).lower()
        ]
        assert pyvi_warnings == [], (
            f"pyvi suppression leaked {len(pyvi_warnings)} warnings: "
            f"{[str(w.message) for w in pyvi_warnings]}"
        )


class TestEncoderLanguagePassthrough:
    """Tests for language parameter propagation through MemoryEncoder."""

    @pytest.fixture()
    async def storage(self) -> Any:
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.memory_store import InMemoryStorage

        store = InMemoryStorage()
        brain = Brain.create(name="test_vi", config=BrainConfig())
        await store.save_brain(brain)
        store.set_brain(brain.id)
        return store

    @pytest.mark.asyncio()
    async def test_encode_accepts_language(self, storage: Any) -> None:
        """Test that MemoryEncoder.encode() accepts language parameter."""
        from neural_memory.core.brain import BrainConfig
        from neural_memory.engine.encoder import MemoryEncoder

        encoder = MemoryEncoder(storage, BrainConfig())

        result = await encoder.encode(
            content="Hôm nay trời đẹp quá",
            language="vi",
        )
        assert result.fiber is not None

    @pytest.mark.asyncio()
    async def test_encode_default_language(self, storage: Any) -> None:
        """Test that encode() works with default language (auto)."""
        from neural_memory.core.brain import BrainConfig
        from neural_memory.engine.encoder import MemoryEncoder

        encoder = MemoryEncoder(storage, BrainConfig())

        result = await encoder.encode(content="The weather is nice today")
        assert result.fiber is not None

    @pytest.mark.asyncio()
    async def test_encode_english_explicit(self, storage: Any) -> None:
        """Test encoding with explicit English language."""
        from neural_memory.core.brain import BrainConfig
        from neural_memory.engine.encoder import MemoryEncoder

        encoder = MemoryEncoder(storage, BrainConfig())

        result = await encoder.encode(
            content="Python programming language is versatile",
            language="en",
        )
        assert result.fiber is not None
        assert len(result.neurons_created) > 0
