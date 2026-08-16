"""CJK-aware text helpers for FTS5 tokenization.

SQLite's ``unicode61`` tokenizer treats a run of CJK characters as a
single token, so an exact query like ``"起源"`` never matches content
``"起源世界"`` — both are one token and FTS5 exact-match does not
substring-match. Inserting spaces between CJK characters makes each
hanzi its own token, and quoting the spaced query as an FTS5 phrase
(``"起 源"``) restores substring-like matching for Chinese short words.

Used on both sides of the FTS pipeline:

- write path: the FTS sync triggers call the registered ``cjk_spaced``
  SQL function so the index stores spaced text (including the
  ``'delete'`` command, which must receive the same tokenization it was
  inserted with);
- read path: the FTS query builders apply the same spacing so the query
  phrase lines up with the indexed token sequence.
"""

from __future__ import annotations

# CJK ideograph blocks relevant for Chinese text (extension A + unified +
# compatibility). Python's str.isalpha() also covers CJK, but an explicit
# range check keeps the helper dependency-free and predictable.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x2B820, 0x2CEAF),  # Extension E
    (0x2CEB0, 0x2EBEF),  # Extension F
    (0x30000, 0x3134F),  # Extension G
)


def is_cjk_char(ch: str) -> bool:
    """Return True if *ch* is a single CJK ideograph."""
    if len(ch) != 1:
        return False
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def cjk_spaced(text: str | None) -> str | None:
    """Insert spaces around CJK characters so unicode61 tokenizes each hanzi separately.

    Non-CJK runs (latin, digits, whitespace) are preserved verbatim.
    ``None`` passes through untouched so the SQL function can be used on
    nullable columns (e.g. ``fibers.summary``).
    """
    if not text:
        return text
    out: list[str] = []
    for ch in text:
        if is_cjk_char(ch):
            out.append(" ")
            out.append(ch)
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)
