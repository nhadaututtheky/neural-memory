"""Tokenizer abstraction for the BM25 lexical index.

The default WhitespaceTokenizer handles English/code well. Vietnamese
content benefits from word segmentation (compound words like "cà phê"
or "vượt đèn đỏ"); the optional `vietnamese` tokenizer routes through
`pyvi` and is registered lazily so the dep stays opt-in.

Stable contract: tokenizers return lower-cased token lists. Empty
strings are filtered out so they cannot leak into the BM25 frequency
counts.
"""

from __future__ import annotations

import re
from typing import Protocol


class Tokenizer(Protocol):
    """Minimal tokenizer protocol: text -> list[token]."""

    def tokenize(self, text: str) -> list[str]: ...


# Token character class — keeps letters, digits, and underscores.
# Drops punctuation so "hello, world!" tokenizes the same as "hello world".
# Backticks/brackets in code-style content also get split (intentional —
# BM25 hits should match on the identifier, not adjacent syntax).
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class WhitespaceTokenizer:
    """Lowercase + Unicode word-character tokenization.

    Suitable for English, code identifiers, and any language where word
    boundaries align with whitespace/punctuation. Vietnamese works
    syllable-by-syllable here — for compound-word matching, use the
    `vietnamese` tokenizer instead.
    """

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def get_tokenizer(name: str) -> Tokenizer:
    """Resolve a tokenizer by name.

    Args:
        name: ``"whitespace"`` (default) or ``"vietnamese"``.

    Raises:
        ValueError: If the name is unknown.
        ImportError: If ``"vietnamese"`` is requested but ``pyvi`` is not
            installed (it is an optional extra to keep the base install lean).
    """
    if name == "whitespace":
        return WhitespaceTokenizer()
    if name == "vietnamese":
        return _build_vietnamese_tokenizer()
    raise ValueError(f"Unknown tokenizer {name!r}. Supported: 'whitespace', 'vietnamese'.")


def _build_vietnamese_tokenizer() -> Tokenizer:
    try:
        from pyvi import ViTokenizer
    except ImportError as exc:
        raise ImportError(
            "Vietnamese tokenization requires the optional `pyvi` package. "
            "Install with: pip install neural-memory[nlp-vi]  # or: pip install pyvi"
        ) from exc

    class _VietnameseTokenizer:
        def tokenize(self, text: str) -> list[str]:
            if not text:
                return []
            # `tokenize` returns underscore-joined compound words; split on
            # whitespace and lowercase to align with the BM25 index contract.
            segmented = ViTokenizer.tokenize(text).lower()
            return [t for t in segmented.split() if t]

    return _VietnameseTokenizer()
