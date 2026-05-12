"""Provenance footer for recall output.

Renders a compact one-line footer per recalled neuron so callers can judge
trust without a separate `nmem_provenance` round-trip.

Format (≤ 60 chars by default):
    [src=<source> · <YYYY-MM-DD> · conf=0.85]

Source resolution falls back through metadata keys, defaulting to "manual"
when no origin is recorded. Note: the canonical key is `_source`, populated
by `remember_handler` at write time. `import_source` is set by external
integrations (mem0 sync etc.). Other keys are tolerated as legacy
fallbacks but are not relied upon for normal `nmem_remember` flows.
"""

from __future__ import annotations

import math

from neural_memory.core.neuron import Neuron

PROVENANCE_MAX_CHARS: int = 60
"""Hard cap on rendered footer length to bound recall token cost."""

_MIN_MAX_CHARS: int = 12
"""Floor for max_chars — below this the bracket structure cannot fit."""

_SOURCE_KEYS: tuple[str, ...] = ("_source", "import_source", "source")
"""Resolution order: canonical first, then external-integration, then legacy."""

_DEFAULT_SOURCE: str = "manual"
_SEPARATOR: str = " · "
_DEFAULT_CONFIDENCE: float = 0.5


def _resolve_source(neuron: Neuron) -> str:
    """Find the first non-empty source string and sanitize it for display."""
    for key in _SOURCE_KEYS:
        value = neuron.metadata.get(key)
        if not value:
            continue
        cleaned = (
            str(value)
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("\t", " ")
            .replace("[", "(")
            .replace("]", ")")
            .strip()
        )
        if cleaned:
            return cleaned
    return _DEFAULT_SOURCE


def _resolve_confidence(neuron: Neuron) -> float:
    """Coerce `_confidence` metadata to a finite float in [0, 1].

    Returns the default (0.5) for missing, non-numeric, NaN, or out-of-range
    values rather than letting the recall pipeline crash on bad data.
    """
    raw = neuron.metadata.get("_confidence", _DEFAULT_CONFIDENCE)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_CONFIDENCE
    if math.isnan(value) or math.isinf(value):
        return _DEFAULT_CONFIDENCE
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def format_provenance_line(
    neuron: Neuron,
    *,
    max_chars: int = PROVENANCE_MAX_CHARS,
) -> str:
    """Render a compact provenance footer for a recalled neuron.

    Args:
        neuron: Neuron whose origin metadata to render.
        max_chars: Hard cap; output is truncated with `…` if it would exceed.
            Values below the structural minimum (12) are clamped up.

    Returns:
        Bracketed line such as ``[src=workflow · 2026-04-12 · conf=0.85]``.
    """
    max_chars = max(int(max_chars), _MIN_MAX_CHARS)

    source = _resolve_source(neuron)
    captured = neuron.created_at.strftime("%Y-%m-%d")
    confidence = f"{_resolve_confidence(neuron):.2f}"

    body = f"src={source}{_SEPARATOR}{captured}{_SEPARATOR}conf={confidence}"
    line = f"[{body}]"

    if len(line) <= max_chars:
        return line

    # Truncate the source first — date and confidence are fixed-width and
    # carry the most signal. Keep at least one character of source plus an
    # ellipsis so the field stays meaningful.
    overflow = len(line) - max_chars
    keep = max(1, len(source) - overflow - 1)
    truncated_source = source[:keep] + "…"
    body = f"src={truncated_source}{_SEPARATOR}{captured}{_SEPARATOR}conf={confidence}"
    line = f"[{body}]"

    if len(line) <= max_chars:
        return line

    # Fallback: hard-clip the entire line. Closing bracket preserved.
    return line[: max_chars - 1] + "]"
