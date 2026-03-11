"""Citation format engine — generates citable references for recalled memories.

Supports multiple output formats (inline, footnote, full) and domain-specific
templates (legal, accounting, generic). Citations are built from Source metadata
and neuron provenance data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CitationFormat(StrEnum):
    """Output format for generated citations."""

    INLINE = "inline"  # [source/article] — compact, in-text
    FOOTNOTE = "footnote"  # ¹ Source Name, version, date
    FULL = "full"  # Complete reference with all metadata


@dataclass(frozen=True)
class CitationInput:
    """Data needed to generate a citation.

    Attributes:
        source_name: Name of the registered source.
        source_type: Type of source (law, contract, ledger, etc.).
        source_version: Version string of the source.
        effective_date: When the source became effective (ISO string or None).
        neuron_id: ID of the neuron being cited.
        metadata: Additional source metadata (article refs, categories, etc.).
    """

    source_name: str
    source_type: str = "document"
    source_version: str = ""
    effective_date: str | None = None
    neuron_id: str = ""
    metadata: dict[str, Any] | None = None


def format_citation(
    inp: CitationInput,
    fmt: CitationFormat | str = CitationFormat.INLINE,
) -> str:
    """Generate a citation string from source data.

    Args:
        inp: Citation input data.
        fmt: Desired output format.

    Returns:
        Formatted citation string.
    """
    fmt = CitationFormat(fmt)

    if fmt == CitationFormat.INLINE:
        return _format_inline(inp)
    if fmt == CitationFormat.FOOTNOTE:
        return _format_footnote(inp)
    return _format_full(inp)


# ── Format implementations ───────────────────────────────────


def _format_inline(inp: CitationInput) -> str:
    """Compact in-text citation: [source/detail]."""
    meta = inp.metadata or {}

    # Legal: [BLDS 2015, Điều 468]
    if inp.source_type == "law":
        article = meta.get("article", "")
        year = _extract_year(inp)
        parts = [inp.source_name]
        if year:
            parts[0] = f"{inp.source_name} {year}"
        if article:
            parts.append(f"Điều {article}")
        return f"[{', '.join(parts)}]"

    # Accounting/ledger: [Source, date, category]
    if inp.source_type == "ledger":
        parts = [inp.source_name]
        if inp.effective_date:
            parts.append(inp.effective_date[:10])
        category = meta.get("category", "")
        if category:
            parts.append(category)
        return f"[{', '.join(parts)}]"

    # Generic fallback — name and short neuron ID
    parts = [inp.source_name]
    if inp.neuron_id:
        parts.append(inp.neuron_id[:8])
    return f"[{', '.join(parts)}]"


def _format_footnote(inp: CitationInput) -> str:
    """Footnote-style citation: Source Name, version, date."""
    parts = [inp.source_name]
    if inp.source_version:
        parts.append(f"v{inp.source_version}")
    if inp.effective_date:
        parts.append(inp.effective_date[:10])
    elif not inp.source_version:
        # No version, no date — nothing extra to add
        pass
    return ", ".join(parts)


def _format_full(inp: CitationInput) -> str:
    """Complete reference with all available metadata."""
    meta = inp.metadata or {}
    lines = [f"Source: {inp.source_name}"]
    lines.append(f"Type: {inp.source_type}")

    if inp.source_version:
        lines.append(f"Version: {inp.source_version}")
    if inp.effective_date:
        lines.append(f"Effective: {inp.effective_date[:10]}")

    # Domain-specific fields
    article = meta.get("article")
    if article:
        lines.append(f"Article: {article}")
    category = meta.get("category")
    if category:
        lines.append(f"Category: {category}")

    if inp.neuron_id:
        lines.append(f"Neuron: {inp.neuron_id[:8]}")

    return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────


def _extract_year(inp: CitationInput) -> str:
    """Extract year from version or effective_date."""
    # Try version first (e.g. "2015", "2024-01")
    if inp.source_version and inp.source_version[:4].isdigit():
        return inp.source_version[:4]
    # Try effective_date
    if inp.effective_date and len(inp.effective_date) >= 4:
        return inp.effective_date[:4]
    return ""
