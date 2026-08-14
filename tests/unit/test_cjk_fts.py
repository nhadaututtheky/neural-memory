"""Unit tests for CJK FTS helpers and query builders.

Covers utils/cjk.py (cjk_spaced / is_cjk_char) and the CJK-aware FTS5
MATCH expression builders shared by the SQLite storage paths.
"""

from __future__ import annotations

from neural_memory.storage.sql.mixins.fibers import _build_fts_query
from neural_memory.storage.sql.mixins.neurons import _build_fts_prefix_query
from neural_memory.utils.cjk import cjk_spaced, is_cjk_char


class TestCjkSpaced:
    def test_cjk_characters_get_spaced(self) -> None:
        assert cjk_spaced("起源世界") == " 起  源  世  界 "

    def test_latin_preserved(self) -> None:
        assert cjk_spaced("v2rayN") == "v2rayN"

    def test_mixed_cjk_latin(self) -> None:
        assert cjk_spaced("v2rayN代理") == "v2rayN 代  理 "

    def test_none_passthrough(self) -> None:
        assert cjk_spaced(None) is None

    def test_empty_string(self) -> None:
        assert cjk_spaced("") == ""


class TestIsCjkChar:
    def test_hanzi_true(self) -> None:
        assert is_cjk_char("起")
        assert is_cjk_char("世")

    def test_latin_and_digits_false(self) -> None:
        assert not is_cjk_char("a")
        assert not is_cjk_char("1")

    def test_multi_char_string_false(self) -> None:
        assert not is_cjk_char("起源")

    def test_extension_b_hanzi_true(self) -> None:
        assert is_cjk_char(chr(0x20000))
        assert is_cjk_char(chr(0x2A6DF))
        assert not is_cjk_char(chr(0x2A6E0))


class TestBuildFtsQuery:
    def test_english_unchanged(self) -> None:
        assert _build_fts_query("API design") == '"API" "design"'

    def test_chinese_short_word_phrased(self) -> None:
        assert _build_fts_query("起源") == '"起  源"'

    def test_chinese_multi_term(self) -> None:
        assert _build_fts_query("世界 电梯") == '"世  界" "电  梯"'

    def test_mixed_cjk_latin(self) -> None:
        assert _build_fts_query("v2rayN代理") == '"v2rayN 代  理"'

    def test_empty(self) -> None:
        assert _build_fts_query("") == '""'

    def test_chinese_with_double_quote_escaped(self) -> None:
        # Quotes inside a CJK piece must be doubled like the non-CJK
        # branch, otherwise the FTS5 phrase is unterminated.
        assert _build_fts_query('起源" OR *') == '"起  源 """ "OR" "*"'


class TestBuildFtsPrefixQuery:
    def test_english_prefix_unchanged(self) -> None:
        assert _build_fts_prefix_query("API des") == '"API" des*'

    def test_chinese_falls_back_to_phrase(self) -> None:
        assert _build_fts_prefix_query("起源") == '"起  源"'

    def test_chinese_multi_terms_fallback(self) -> None:
        assert _build_fts_prefix_query("世界 电梯") == '"世  界" "电  梯"'
