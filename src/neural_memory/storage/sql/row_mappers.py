"""Dialect-agnostic row-to-model conversion functions.

Works with both SQLite (aiosqlite.Row) and PostgreSQL (asyncpg.Record)
rows, since both support dict-style access via row["column"].

All functions accept an optional ``dialect`` parameter for explicit datetime
normalization, but default to auto-detection so callers don't need changes:
  - SQLite  → ISO format strings  (parsed via datetime.fromisoformat)
  - PG      → native datetime objects (tz→UTC normalized)
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from typing import Any

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import (
    Confidence,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
)
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.utils.timeutils import utcnow


def _get(row: Any, key: str, default: Any = None) -> Any:
    """Get value from a row (aiosqlite.Row or asyncpg.Record) with default."""
    try:
        val = row[key]
        return val if val is not None else default
    except (KeyError, TypeError):
        return default


def _normalize_dt(value: Any, dialect: Any = None) -> datetime | None:
    """Normalize a datetime value.

    When an explicit ``dialect`` is supplied, defer to ``dialect.normalize_dt``
    (the documented contract); otherwise auto-detect the backend format:

    - ``None``               → ``None``
    - ``datetime`` object    → strip/normalize tz to naive UTC
    - Anything else         → ``datetime.fromisoformat(str(value))`` (SQLite ISO strings)
    """
    if dialect is not None and hasattr(dialect, "normalize_dt"):
        normalized = dialect.normalize_dt(value)
        if isinstance(normalized, datetime) and normalized.tzinfo is not None:
            return normalized.astimezone(UTC).replace(tzinfo=None)
        return normalized if normalized is None or isinstance(normalized, datetime) else None
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value
    return datetime.fromisoformat(str(value))


def _resolve_row_dialect(arg1: Any, arg2: Any) -> tuple[Any, Any]:
    """Resolve the (row, dialect) pair from either call signature.

    Supports both ``fn(row)``, ``fn(row, dialect)`` and ``fn(dialect, row)``.
    Detects a Dialect by its ``.name`` attribute. Returns ``(row, dialect)``
    where ``dialect`` may be ``None`` (auto-detect datetime normalization).
    """
    if arg2 is None:
        return arg1, None
    if hasattr(arg1, "name"):
        # (dialect, row) order
        return arg2, arg1
    # (row, dialect) order — honor the dialect instead of dropping it.
    return arg1, arg2


def _parse_json_field(value: Any) -> Any:
    """Parse a JSON field that may be a string (SQLite) or already parsed (PG)."""
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def row_to_neuron(dialect_or_row: Any, row_or_dialect: Any = None) -> Neuron:
    """Convert database row to Neuron.

    Accepts ``(row)``, ``(row, dialect)`` or ``(dialect, row)``.
    """
    row, dialect = _resolve_row_dialect(dialect_or_row, row_or_dialect)
    return Neuron(
        id=str(row["id"]),
        type=NeuronType(row["type"]),
        content=str(row["content"]),
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        content_hash=int(_get(row, "content_hash", 0)),
        created_at=_normalize_dt(row["created_at"], dialect) or utcnow(),
        ephemeral=bool(_get(row, "ephemeral", False)),
    )


def row_to_neuron_state(dialect_or_row: Any, row_or_dialect: Any = None) -> NeuronState:
    """Convert database row to NeuronState.

    Accepts ``(row)``, ``(row, dialect)`` or ``(dialect, row)``.
    """
    row, dialect = _resolve_row_dialect(dialect_or_row, row_or_dialect)
    return NeuronState(
        neuron_id=str(row["neuron_id"]),
        activation_level=float(_get(row, "activation_level", 0.0)),
        access_frequency=int(_get(row, "access_frequency", 0)),
        last_activated=_normalize_dt(_get(row, "last_activated"), dialect),
        decay_rate=float(_get(row, "decay_rate", 0.1)),
        created_at=_normalize_dt(row["created_at"], dialect) or utcnow(),
        firing_threshold=float(_get(row, "firing_threshold", 0.3)),
        refractory_until=_normalize_dt(_get(row, "refractory_until"), dialect),
        refractory_period_ms=float(_get(row, "refractory_period_ms", 500.0)),
        homeostatic_target=float(_get(row, "homeostatic_target", 0.5)),
    )


def row_to_synapse(dialect_or_row: Any, row_or_dialect: Any = None) -> Synapse:
    """Convert database row to Synapse.

    Accepts ``(row)``, ``(row, dialect)`` or ``(dialect, row)``.
    """
    row, dialect = _resolve_row_dialect(dialect_or_row, row_or_dialect)
    return Synapse(
        id=str(row["id"]),
        source_id=str(row["source_id"]),
        target_id=str(row["target_id"]),
        type=SynapseType(row["type"]),
        weight=float(row["weight"]),
        direction=Direction(_get(row, "direction", "uni")),
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        reinforced_count=int(_get(row, "reinforced_count", 0)),
        last_activated=_normalize_dt(_get(row, "last_activated"), dialect),
        created_at=_normalize_dt(row["created_at"], dialect) or utcnow(),
    )


def row_to_fiber(dialect_or_row: Any, row_or_dialect: Any = None) -> Fiber:
    """Convert database row to Fiber.

    Accepts ``(row)``, ``(row, dialect)`` or ``(dialect, row)``.
    """
    row, dialect = _resolve_row_dialect(dialect_or_row, row_or_dialect)
    neuron_ids = _parse_json_field(row["neuron_ids"]) or []
    synapse_ids = _parse_json_field(row["synapse_ids"]) or []
    pathway = _parse_json_field(row["pathway"]) or []
    tags_raw = set(_parse_json_field(row["tags"]) or [])
    auto_tags = set(_parse_json_field(row["auto_tags"]) or [])
    agent_tags = set(_parse_json_field(row["agent_tags"]) or [])
    if not auto_tags and not agent_tags and tags_raw:
        agent_tags = tags_raw
    metadata = _parse_json_field(row["metadata"]) or {}

    return Fiber(
        id=str(row["id"]),
        neuron_ids=set(neuron_ids),
        synapse_ids=set(synapse_ids),
        anchor_neuron_id=str(row["anchor_neuron_id"]),
        pathway=pathway,
        conductivity=float(_get(row, "conductivity", 1.0)),
        last_conducted=_normalize_dt(_get(row, "last_conducted"), dialect),
        time_start=_normalize_dt(_get(row, "time_start"), dialect),
        time_end=_normalize_dt(_get(row, "time_end"), dialect),
        coherence=float(_get(row, "coherence", 0.0)),
        salience=float(_get(row, "salience", 0.0)),
        frequency=int(_get(row, "frequency", 0)),
        summary=_get(row, "summary"),
        essence=_get(row, "essence"),
        last_ghost_shown_at=_normalize_dt(_get(row, "last_ghost_shown_at"), dialect),
        auto_tags=auto_tags,
        agent_tags=agent_tags,
        metadata=metadata,
        compression_tier=int(_get(row, "compression_tier", 0)),
        pinned=bool(_get(row, "pinned", False)),
        created_at=_normalize_dt(row["created_at"], dialect) or utcnow(),
    )


def row_to_brain(dialect_or_row: Any, row_or_dialect: Any = None) -> Brain:
    """Convert database row to Brain.

    Loads every BrainConfig field stored in the row, filtered against the
    current dataclass schema (forward-compat for old saves, backward-compat
    for renamed fields). The previous hard-coded subset dropped any field
    added to BrainConfig after this function was written — see issue #168
    where ``bm25_enabled``/``high_signal_memory_boost`` silently reverted to
    defaults on every load.
    """
    row, dialect = _resolve_row_dialect(dialect_or_row, row_or_dialect)
    config_data = _parse_json_field(row["config"]) or {}
    valid_fields = {f.name for f in dataclasses.fields(BrainConfig)}
    filtered_config = {k: v for k, v in config_data.items() if k in valid_fields}
    config = BrainConfig(**filtered_config)

    shared_with = _parse_json_field(row["shared_with"]) or []
    return Brain(
        id=str(row["id"]),
        name=str(row["name"]),
        config=config,
        owner_id=row["owner_id"],
        is_public=bool(_get(row, "is_public", False)),
        shared_with=shared_with,
        created_at=_normalize_dt(row["created_at"], dialect) or utcnow(),
        updated_at=_normalize_dt(_get(row, "updated_at"), dialect) or utcnow(),
    )


def provenance_to_dict(provenance: Provenance) -> dict[str, object]:
    """Serialize Provenance to a JSON-compatible dict."""
    return {
        "source": provenance.source,
        "confidence": provenance.confidence.value,
        "verified": provenance.verified,
        "verified_at": (provenance.verified_at.isoformat() if provenance.verified_at else None),
        "created_by": provenance.created_by,
        "last_confirmed": (
            provenance.last_confirmed.isoformat() if provenance.last_confirmed else None
        ),
    }


def row_to_typed_memory(dialect_or_row: Any, row_or_dialect: Any = None) -> TypedMemory:
    """Convert database row to TypedMemory.

    Works with both SQLite (string provenance) and PostgreSQL (possibly native dict).
    """
    row, dialect = _resolve_row_dialect(dialect_or_row, row_or_dialect)
    prov_raw = row["provenance"]
    prov_data = _parse_json_field(prov_raw) or {}
    provenance = Provenance(
        source=prov_data.get("source", "unknown"),
        confidence=Confidence(prov_data.get("confidence", "medium")),
        verified=prov_data.get("verified", False),
        verified_at=(
            datetime.fromisoformat(prov_data["verified_at"])
            if prov_data.get("verified_at")
            else None
        ),
        created_by=prov_data.get("created_by", "unknown"),
        last_confirmed=(
            datetime.fromisoformat(prov_data["last_confirmed"])
            if prov_data.get("last_confirmed")
            else None
        ),
    )

    # trust_score, source, and tier added in schema v22/v37 — graceful fallback
    trust_score: float | None = None
    try:
        ts = row["trust_score"]
        if ts is not None:
            trust_score = float(ts)
    except (KeyError, TypeError, IndexError):
        pass

    source: str | None = None
    try:
        source = str(row["source"]) if row["source"] is not None else None
    except (KeyError, TypeError):
        pass

    tier: str = "warm"
    try:
        t = row["tier"]
        if t is not None:
            tier = str(t)
    except (KeyError, TypeError):
        pass

    tags_list = _parse_json_field(row["tags"]) or []
    meta_raw = row["metadata"]
    metadata = _parse_json_field(meta_raw) or {}

    return TypedMemory(
        fiber_id=str(row["fiber_id"]),
        memory_type=MemoryType(row["memory_type"]),
        priority=Priority(row["priority"]),
        provenance=provenance,
        expires_at=_normalize_dt(_get(row, "expires_at"), dialect),
        project_id=_get(row, "project_id"),
        tags=frozenset(tags_list),
        metadata=metadata,
        created_at=_normalize_dt(row["created_at"], dialect) or utcnow(),
        trust_score=trust_score,
        source=source,
        tier=tier,
    )


def _row_to_joined_synapse(dialect_or_row: Any, row_or_dialect: Any = None) -> Synapse:
    """Convert a joined row (s_ prefixed columns) to a Synapse.

    Used by graph traversal in synapses mixin.
    """
    row, dialect = _resolve_row_dialect(dialect_or_row, row_or_dialect)
    s_created = _get(row, "s_created_at") or _get(row, "s_created")
    created = _normalize_dt(s_created, dialect)
    return Synapse(
        id=str(row["s_id"]),
        source_id=str(row["source_id"]),
        target_id=str(row["target_id"]),
        type=SynapseType(row["s_type"]),
        weight=float(row["weight"]),
        direction=Direction(_get(row, "direction", "uni")),
        metadata=_parse_json_field(_get(row, "s_metadata")) or {},
        reinforced_count=int(_get(row, "reinforced_count", 0)),
        last_activated=_normalize_dt(_get(row, "s_last_activated"), dialect),
        created_at=created or utcnow(),
    )
