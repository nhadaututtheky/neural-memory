"""Tests for the encoding damage guard applied before a memory is written.

Content piped through a non-UTF-8 shell (Windows PowerShell 5.1 defaults
$OutputEncoding to ASCII) reaches storage silently damaged. The guard rejects
the write instead of persisting "?" where Chinese characters used to be.
"""

from __future__ import annotations

from neural_memory.safety.capture_hygiene import detect_encoding_damage


class TestDetectEncodingDamage:
    """The guard must catch real damage without flagging legitimate content."""

    def test_clean_chinese_passes(self) -> None:
        assert detect_encoding_damage("这是一条很正常的中文记忆内容") is None

    def test_clean_english_passes(self) -> None:
        assert detect_encoding_damage("We should refactor the auth module") is None

    def test_empty_passes(self) -> None:
        assert detect_encoding_damage("") is None

    def test_null_coalescing_operator_is_not_damage(self) -> None:
        """C#/JS '??' is common in code notes and must not be treated as damage."""
        assert detect_encoding_damage("var x = a ?? b; // C# null coalescing") is None

    def test_trailing_double_question_is_not_damage(self) -> None:
        assert detect_encoding_damage("Really?? That seems wrong to me") is None

    def test_question_mark_flood_is_damage(self) -> None:
        reason = detect_encoding_damage("????????????????")
        assert reason is not None
        assert "?" in reason

    def test_literal_unicode_escapes_are_damage(self) -> None:
        reason = detect_encoding_damage(r"内容 \u4e2d\u6587 没有被解码")
        assert reason is not None
        assert "uXXXX" in reason

    def test_single_literal_escape_is_not_damage(self) -> None:
        """A lone escape is plausible in technical notes about Unicode."""
        assert detect_encoding_damage(r"use \u200b for a zero-width space") is None

    def test_replacement_character_is_damage(self) -> None:
        reason = detect_encoding_damage("内容\ufffd损坏")
        assert reason is not None
        assert "U+FFFD" in reason
