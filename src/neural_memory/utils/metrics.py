"""Prometheus counters for silent-failure signals.

The base install has no prometheus-client dependency. Recording is a no-op
until that package is present (the ``server`` extra). ``GET /metrics`` on
the FastAPI server renders this registry.

Series stay low-cardinality: no brain names, no memory text, no file paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_PLAIN = "text/plain; version=0.0.4; charset=utf-8"
_DISABLED = b"# neural-memory metrics disabled. Install with: pip install 'neural-memory[server]'\n"

_COMMANDS = frozenset({"remember", "forget", "list"})
_DAMAGE_TYPES = frozenset({"question_mark_flood", "unicode_escape", "replacement_char"})
_EMPTY_REASONS = frozenset({"too_short", "cjk_unsupported", "no_trigger"})
_LANGUAGES = frozenset({"zh", "vi", "en", "ja", "ko"})

# The driver message when one commit lands while another statement is open.
_COLLISION_MARKER = "SQL statements in progress"


@dataclass(frozen=True)
class _Instruments:
    """Hold the process-local registry. The counters inside it stay mutable."""

    registry: Any
    collision: Any
    latency: Any
    encoding: Any
    empty: Any
    content_type: str
    generate_latest: Any


def _bounded(value: str, allowed: frozenset[str]) -> str:
    """Keep a label inside a fixed set so a caller cannot grow the series."""
    if value in allowed:
        return value
    return "other"


def _load_instruments() -> _Instruments | None:
    try:
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            Counter,
            Histogram,
            generate_latest,
        )
    except ImportError:
        return None

    registry = CollectorRegistry()
    # Counter names omit the ``_total`` suffix. The client adds it on export,
    # which is the name Prometheus scrapes.
    collision = Counter(
        "neural_memory_sqlite_write_collision",
        "SQLite commit collisions detected on the single-writer queue",
        ["dialect"],
        registry=registry,
    )
    latency = Histogram(
        "neural_memory_sqlite_write_latency_seconds",
        "Time spent waiting for the SQLite write queue and executing the write",
        ["dialect"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        registry=registry,
    )
    encoding = Counter(
        "neural_memory_cli_encoding_damage_rejected",
        "CLI input rejected because the text looks encoding-damaged",
        ["command", "damage_type"],
        registry=registry,
    )
    empty = Counter(
        "neural_memory_auto_capture_empty",
        "Auto-capture returned no memories",
        ["reason", "language"],
        registry=registry,
    )
    return _Instruments(
        registry=registry,
        collision=collision,
        latency=latency,
        encoding=encoding,
        empty=empty,
        content_type=str(CONTENT_TYPE_LATEST),
        generate_latest=generate_latest,
    )


_INSTRUMENTS = _load_instruments()


def metrics_enabled() -> bool:
    """Return whether prometheus-client is loaded in this process."""
    return _INSTRUMENTS is not None


def is_sqlite_write_collision(exc: BaseException) -> bool:
    """Return whether ``exc`` is the SQLite overlapping-commit error."""
    return _COLLISION_MARKER in str(exc)


def record_sqlite_write(elapsed_seconds: float, *, collision: bool) -> None:
    """Record one SQLite write. Failures here do not fail the write itself."""
    instruments = _INSTRUMENTS
    if instruments is None:
        return
    try:
        instruments.latency.labels(dialect="sqlite").observe(elapsed_seconds)
        if collision:
            instruments.collision.labels(dialect="sqlite").inc()
    except Exception:
        logger.warning("Failed to record SQLite write metric", exc_info=True)


def record_encoding_rejection(command: str, damage_type: str) -> None:
    """Count one CLI refusal of encoding-damaged input."""
    instruments = _INSTRUMENTS
    if instruments is None:
        return
    try:
        instruments.encoding.labels(
            command=_bounded(command, _COMMANDS),
            damage_type=_bounded(damage_type, _DAMAGE_TYPES),
        ).inc()
    except Exception:
        logger.warning("Failed to record encoding-damage metric", exc_info=True)


def record_empty_capture(reason: str, language: str) -> None:
    """Count one auto-capture that found nothing."""
    instruments = _INSTRUMENTS
    if instruments is None:
        return
    try:
        instruments.empty.labels(
            reason=_bounded(reason, _EMPTY_REASONS),
            language=_bounded(language, _LANGUAGES),
        ).inc()
    except Exception:
        logger.warning("Failed to record empty-capture metric", exc_info=True)


def render_metrics() -> tuple[bytes, str]:
    """Return Prometheus text and its content type.

    An install without prometheus-client still returns a short comment so
    ``GET /metrics`` can answer.
    """
    instruments = _INSTRUMENTS
    if instruments is None:
        return _DISABLED, _PLAIN
    try:
        body = instruments.generate_latest(instruments.registry)
    except Exception:
        logger.warning("Failed to render metrics", exc_info=True)
        return _DISABLED, _PLAIN
    if isinstance(body, str):
        body = body.encode("utf-8")
    return bytes(body), instruments.content_type


def metric_sample(name: str, labels: dict[str, str]) -> float:
    """Return one sample, or ``0`` when it is absent or metrics are disabled."""
    instruments = _INSTRUMENTS
    if instruments is None:
        return 0.0
    value = instruments.registry.get_sample_value(name, labels)
    if value is None:
        return 0.0
    return float(value)
