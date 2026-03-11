"""Tests for citation format engine (Phase 4)."""

from __future__ import annotations

import pytest

from neural_memory.engine.citation import (
    CitationFormat,
    CitationInput,
    format_citation,
)

# ──────────────────── Inline Format ────────────────────


class TestInlineCitation:
    """Inline citation format tests."""

    def test_generic_with_neuron_id(self) -> None:
        inp = CitationInput(
            source_name="Report Q4",
            source_type="document",
            neuron_id="abc12345-6789-0000-0000-000000000000",
        )
        result = format_citation(inp, CitationFormat.INLINE)
        assert result == "[Report Q4, abc12345]"

    def test_generic_without_neuron_id(self) -> None:
        inp = CitationInput(source_name="Manual Entry")
        result = format_citation(inp, CitationFormat.INLINE)
        assert result == "[Manual Entry]"

    def test_law_with_article(self) -> None:
        inp = CitationInput(
            source_name="BLDS",
            source_type="law",
            source_version="2015",
            metadata={"article": "468"},
        )
        result = format_citation(inp, CitationFormat.INLINE)
        assert result == "[BLDS 2015, Điều 468]"

    def test_law_without_article(self) -> None:
        inp = CitationInput(
            source_name="BLDS",
            source_type="law",
            source_version="2015",
        )
        result = format_citation(inp, CitationFormat.INLINE)
        assert result == "[BLDS 2015]"

    def test_law_with_effective_date_year(self) -> None:
        inp = CitationInput(
            source_name="Tax Code",
            source_type="law",
            effective_date="2020-06-01",
            metadata={"article": "10"},
        )
        result = format_citation(inp, CitationFormat.INLINE)
        assert result == "[Tax Code 2020, Điều 10]"

    def test_ledger_with_category(self) -> None:
        inp = CitationInput(
            source_name="Cafe Saigon",
            source_type="ledger",
            effective_date="2026-01-15",
            metadata={"category": "Salary"},
        )
        result = format_citation(inp, CitationFormat.INLINE)
        assert result == "[Cafe Saigon, 2026-01-15, Salary]"

    def test_ledger_without_category(self) -> None:
        inp = CitationInput(
            source_name="Shop A",
            source_type="ledger",
            effective_date="2026-03-01",
        )
        result = format_citation(inp, CitationFormat.INLINE)
        assert result == "[Shop A, 2026-03-01]"

    def test_string_format_accepted(self) -> None:
        """Verify string format names work."""
        inp = CitationInput(source_name="Test")
        result = format_citation(inp, "inline")
        assert result.startswith("[")


# ──────────────────── Footnote Format ────────────────────


class TestFootnoteCitation:
    """Footnote citation format tests."""

    def test_with_version_and_year(self) -> None:
        inp = CitationInput(
            source_name="Civil Code",
            source_version="2015",
        )
        result = format_citation(inp, CitationFormat.FOOTNOTE)
        assert result == "Civil Code, v2015"

    def test_with_date_no_version(self) -> None:
        inp = CitationInput(
            source_name="Meeting Notes",
            effective_date="2026-03-10T14:00:00",
        )
        result = format_citation(inp, CitationFormat.FOOTNOTE)
        assert result == "Meeting Notes, 2026-03-10"

    def test_name_only(self) -> None:
        inp = CitationInput(source_name="Quick Note")
        result = format_citation(inp, CitationFormat.FOOTNOTE)
        assert result == "Quick Note"

    def test_with_version_and_date(self) -> None:
        inp = CitationInput(
            source_name="API Docs",
            source_version="2.0",
            effective_date="2026-01-01",
        )
        result = format_citation(inp, CitationFormat.FOOTNOTE)
        assert result == "API Docs, v2.0, 2026-01-01"


# ──────────────────── Full Format ────────────────────


class TestFullCitation:
    """Full reference format tests."""

    def test_complete_reference(self) -> None:
        inp = CitationInput(
            source_name="BLDS",
            source_type="law",
            source_version="2015",
            effective_date="2015-01-01",
            neuron_id="abc12345-dead-beef",
            metadata={"article": "468"},
        )
        result = format_citation(inp, CitationFormat.FULL)
        assert "Source: BLDS" in result
        assert "Type: law" in result
        assert "Version: 2015" in result
        assert "Effective: 2015-01-01" in result
        assert "Article: 468" in result
        assert "Neuron: abc12345" in result

    def test_minimal_reference(self) -> None:
        inp = CitationInput(source_name="Note", source_type="manual")
        result = format_citation(inp, CitationFormat.FULL)
        assert "Source: Note" in result
        assert "Type: manual" in result
        assert "Version" not in result

    def test_with_category(self) -> None:
        inp = CitationInput(
            source_name="Ledger",
            source_type="ledger",
            metadata={"category": "Office Supplies"},
        )
        result = format_citation(inp, CitationFormat.FULL)
        assert "Category: Office Supplies" in result


# ──────────────────── CitationInput ────────────────────


class TestCitationInput:
    """CitationInput dataclass behavior."""

    def test_frozen(self) -> None:
        inp = CitationInput(source_name="Test")
        with pytest.raises(AttributeError):
            inp.source_name = "Mutated"  # type: ignore[misc]

    def test_defaults(self) -> None:
        inp = CitationInput(source_name="X")
        assert inp.source_type == "document"
        assert inp.source_version == ""
        assert inp.effective_date is None
        assert inp.neuron_id == ""
        assert inp.metadata is None


# ──────────────────── CitationFormat Enum ────────────────────


class TestCitationFormatEnum:
    """Verify enum values."""

    def test_all_values(self) -> None:
        assert CitationFormat.INLINE.value == "inline"
        assert CitationFormat.FOOTNOTE.value == "footnote"
        assert CitationFormat.FULL.value == "full"

    def test_from_string(self) -> None:
        assert CitationFormat("inline") == CitationFormat.INLINE
        assert CitationFormat("footnote") == CitationFormat.FOOTNOTE
        assert CitationFormat("full") == CitationFormat.FULL
