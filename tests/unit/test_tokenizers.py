"""Tests for the tokenizer abstraction used by the BM25 lexical index.

Item #1 from plan-tllr-learnings: pluggable tokenization so Vietnamese
content can be segmented properly while default English/code stays
whitespace-based. Vietnamese implementation is opt-in (lazy import).
"""

from __future__ import annotations

import pytest

from neural_memory.engine.tokenizers import (
    WhitespaceTokenizer,
    get_tokenizer,
)


def test_whitespace_tokenizer_splits_on_whitespace() -> None:
    tok = WhitespaceTokenizer()
    assert tok.tokenize("hello world foo") == ["hello", "world", "foo"]


def test_whitespace_tokenizer_lowercases() -> None:
    tok = WhitespaceTokenizer()
    assert tok.tokenize("Hello WORLD") == ["hello", "world"]


def test_whitespace_tokenizer_empty_input() -> None:
    assert WhitespaceTokenizer().tokenize("") == []


def test_whitespace_tokenizer_collapses_repeated_whitespace() -> None:
    tok = WhitespaceTokenizer()
    assert tok.tokenize("a  b\t\n c") == ["a", "b", "c"]


def test_whitespace_tokenizer_strips_punctuation() -> None:
    """Recall keyword queries usually omit punctuation — index must too."""
    tok = WhitespaceTokenizer()
    assert tok.tokenize("hello, world!") == ["hello", "world"]


def test_whitespace_tokenizer_keeps_numbers_and_underscores() -> None:
    """Code identifiers like `format_provenance_line` and `v4.55` should survive."""
    tok = WhitespaceTokenizer()
    assert tok.tokenize("format_provenance_line v4_55") == ["format_provenance_line", "v4_55"]


def test_whitespace_tokenizer_filters_empty_tokens() -> None:
    """After strip, no empty strings should leak into the token list."""
    tok = WhitespaceTokenizer()
    assert "" not in tok.tokenize(",,,,")


def test_get_tokenizer_returns_whitespace_by_default() -> None:
    tok = get_tokenizer("whitespace")
    assert isinstance(tok, WhitespaceTokenizer)


def test_get_tokenizer_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        get_tokenizer("unknown-language-xyz")


def test_get_tokenizer_vietnamese_raises_on_missing_dep() -> None:
    """`pyvi` is an optional extra; absence yields a helpful error, not crash."""
    pytest.importorskip_pyvi = None  # type: ignore[attr-defined]
    try:
        import pyvi  # noqa: F401

        pytest.skip("pyvi installed — skipping the missing-dep contract test")
    except ImportError:
        pass

    with pytest.raises(ImportError, match="pyvi"):
        get_tokenizer("vietnamese")
