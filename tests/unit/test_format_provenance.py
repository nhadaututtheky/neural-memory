"""Tests for the provenance footer rendered in recall output.

Item #5 from plan-tllr-learnings: each recalled neuron gets a one-line
provenance footer so agents can judge trust without a separate
`nmem_provenance` round-trip.

Format (≤ 60 chars):
    [src=<source> · <YYYY-MM-DD> · conf=0.85]
"""

from __future__ import annotations

from datetime import datetime

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.format_provenance import (
    PROVENANCE_MAX_CHARS,
    format_provenance_line,
)


def _make_neuron(
    *,
    metadata: dict | None = None,
    created_at: datetime | None = None,
) -> Neuron:
    return Neuron(
        id="n1",
        type=NeuronType.CONCEPT,
        content="x",
        metadata=metadata or {},
        created_at=created_at or datetime(2026, 4, 12, 10, 0, 0),
    )


def test_format_provenance_line_all_fields_present() -> None:
    n = _make_neuron(metadata={"source": "workflow", "_confidence": 0.85})
    line = format_provenance_line(n)

    assert line.startswith("[")
    assert line.endswith("]")
    assert "workflow" in line
    assert "2026-04-12" in line
    assert "0.85" in line


def test_format_provenance_line_default_source_is_manual() -> None:
    n = _make_neuron(metadata={"_confidence": 0.7})
    line = format_provenance_line(n)
    assert "manual" in line


def test_format_provenance_line_falls_back_to_import_source() -> None:
    n = _make_neuron(metadata={"import_source": "mem0", "_confidence": 0.6})
    line = format_provenance_line(n)
    assert "mem0" in line


def test_format_provenance_line_default_confidence() -> None:
    """Neurons without _confidence default to 0.5 (Neuron.confidence property)."""
    n = _make_neuron(metadata={"source": "auto"})
    line = format_provenance_line(n)
    assert "0.50" in line  # always 2 decimals


def test_format_provenance_line_respects_max_chars() -> None:
    long_source = "a-very-very-very-long-source-name-that-overflows-everything"
    n = _make_neuron(metadata={"source": long_source, "_confidence": 0.9})
    line = format_provenance_line(n)
    assert len(line) <= PROVENANCE_MAX_CHARS


def test_format_provenance_line_includes_separator() -> None:
    n = _make_neuron(metadata={"source": "x", "_confidence": 0.5})
    line = format_provenance_line(n)
    # Three components separated — count separators
    assert line.count(" · ") == 2 or line.count("·") >= 2


def test_format_provenance_line_no_metadata() -> None:
    """Empty metadata dict — function must not raise."""
    n = _make_neuron(metadata={})
    line = format_provenance_line(n)
    assert "manual" in line
    assert "2026-04-12" in line


def test_format_provenance_line_iso_date_format() -> None:
    """Created_at always rendered as YYYY-MM-DD (no time, no timezone)."""
    n = _make_neuron(
        metadata={"source": "x"},
        created_at=datetime(2025, 1, 3, 23, 59, 59),
    )
    line = format_provenance_line(n)
    assert "2025-01-03" in line
    # Should NOT contain time or timezone markers
    assert "T" not in line
    assert "23:59" not in line


def test_format_provenance_line_custom_max_chars() -> None:
    n = _make_neuron(metadata={"source": "workflow", "_confidence": 0.85})
    line = format_provenance_line(n, max_chars=20)
    assert len(line) <= 20


# ── Defensive parsing fixes (review feedback) ────────────────────────


def test_canonical_source_key_wins_over_legacy() -> None:
    """`_source` (canonical) outranks `import_source` and `source` (legacy)."""
    n = _make_neuron(metadata={"_source": "canonical", "import_source": "imp", "source": "legacy"})
    line = format_provenance_line(n)
    assert "canonical" in line
    assert "imp" not in line
    assert "legacy" not in line


def test_import_source_used_when_canonical_missing() -> None:
    n = _make_neuron(metadata={"import_source": "mem0", "source": "ignored"})
    line = format_provenance_line(n)
    assert "mem0" in line


def test_nan_confidence_falls_back_to_default() -> None:
    """`_confidence=NaN` would render `conf=nan` — defend by clamping to default."""
    n = _make_neuron(metadata={"_source": "x", "_confidence": float("nan")})
    line = format_provenance_line(n)
    assert "nan" not in line.lower()
    assert "0.50" in line


def test_inf_confidence_falls_back_to_default() -> None:
    n = _make_neuron(metadata={"_source": "x", "_confidence": float("inf")})
    line = format_provenance_line(n)
    assert "inf" not in line.lower()
    assert "0.50" in line


def test_string_confidence_does_not_crash() -> None:
    """Non-numeric `_confidence` should not propagate ValueError into recall."""
    n = _make_neuron(metadata={"_source": "x", "_confidence": "high"})
    line = format_provenance_line(n)
    assert "0.50" in line


def test_negative_confidence_clamped_to_zero() -> None:
    n = _make_neuron(metadata={"_source": "x", "_confidence": -0.5})
    line = format_provenance_line(n)
    assert "0.00" in line


def test_oversized_confidence_clamped_to_one() -> None:
    n = _make_neuron(metadata={"_source": "x", "_confidence": 5.0})
    line = format_provenance_line(n)
    assert "1.00" in line


def test_source_with_newline_sanitized() -> None:
    n = _make_neuron(metadata={"_source": "line1\nline2"})
    line = format_provenance_line(n)
    assert "\n" not in line
    assert "line1 line2" in line


def test_source_with_brackets_replaced_to_parens() -> None:
    """Embedded `[...]` would corrupt automated parsing of the footer."""
    n = _make_neuron(metadata={"_source": "src[with]bracket"})
    line = format_provenance_line(n)
    # The outer brackets are the only square brackets in the rendered line.
    assert line.count("[") == 1
    assert line.count("]") == 1
    assert "src(with)bracket" in line


def test_whitespace_only_source_falls_back_to_default() -> None:
    n = _make_neuron(metadata={"_source": "   "})
    line = format_provenance_line(n)
    assert "manual" in line


def test_max_chars_zero_clamped_to_minimum() -> None:
    """`max_chars=0` would overflow without a guard — clamp up to a sane floor."""
    n = _make_neuron(metadata={"_source": "x", "_confidence": 0.5})
    line = format_provenance_line(n, max_chars=0)
    assert line.startswith("[")
    assert line.endswith("]")
    assert len(line) >= 12


def test_max_chars_one_clamped_to_minimum() -> None:
    n = _make_neuron(metadata={"_source": "x"})
    line = format_provenance_line(n, max_chars=1)
    assert line.startswith("[")
    assert line.endswith("]")
    assert len(line) >= 12
